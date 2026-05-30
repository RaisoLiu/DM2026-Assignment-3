#!/usr/bin/env python3
"""Aggressive Generator candidate 3: 5-source HOS-12-optimized ensemble +
self-blend c2/c3 promote.

Mechanism:
  1. Blend SSL v1, SSL v2, centered_meta, catch22_lgbm, catch22_xgb at
     HOS-12-optimal weights (v2 dominates at 0.588).
  2. Viterbi smooth (alpha=1.0, beta=0.18).
  3. SELF-PROMOTE: where Viterbi-pred != c2 and blend_proba[c2] >= 0.30, top 30
     by blend c2 proba → switch to c2.
  4. Similarly for c3: top 15 above 0.55.

HOS-12 macro: 0.8006 (highest in this batch). c2 F1 = 0.442.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sequence_smoothing import (  # noqa: E402
    estimate_transition_model,
    viterbi_predict_by_user,
)


def align_proba(src, ref_fids):
    idx = {int(f): i for i, f in enumerate(src["file_id"].astype(int))}
    order = np.array([idx[int(f)] for f in ref_fids], dtype=int)
    return src["proba"][order].astype(float)


def main() -> None:
    out_csv = ROOT / "submissions/agg_5src_hos12_promote.csv"
    out_meta = ROOT / "submissions/agg_5src_hos12_promote_metadata.json"

    weights = {"ssl_v1": 0.011, "ssl_v2": 0.588, "centered_meta": 0.240, "catch22_lgbm": 0.075, "catch22_xgb": 0.086}
    alpha, beta = 1.0, 0.18

    # TEST
    t_we = np.load(ROOT / "artifacts/weighted_ensemble/test.npz", allow_pickle=True)
    t_v1 = np.load(ROOT / "artifacts/inception_full_ssl/test_proba_avg3.npz", allow_pickle=True)
    t_v2 = np.load(ROOT / "artifacts/inception_full_v2/test_avg.npz", allow_pickle=True)
    t_cm = np.load(ROOT / "artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz", allow_pickle=True)
    t_c22l = np.load(ROOT / "artifacts/catch22_full/test_proba_lgbm.npz", allow_pickle=True)
    t_c22x = np.load(ROOT / "artifacts/catch22_full/test_proba_xgb.npz", allow_pickle=True)

    ref_fids = t_we["file_id"].astype(int)
    classes = t_we["classes"].astype(int)

    p_v1 = align_proba(t_v1, ref_fids)
    p_v2 = align_proba(t_v2, ref_fids)
    p_cm = align_proba(t_cm, ref_fids)
    p_c22l = align_proba(t_c22l, ref_fids)
    p_c22x = align_proba(t_c22x, ref_fids)

    blended_test = (
        weights["ssl_v1"] * p_v1 + weights["ssl_v2"] * p_v2 + weights["centered_meta"] * p_cm
        + weights["catch22_lgbm"] * p_c22l + weights["catch22_xgb"] * p_c22x
    )
    blended_test = blended_test / blended_test.sum(axis=1, keepdims=True)

    # OOF for transitions
    o_we = np.load(ROOT / "artifacts/weighted_ensemble/oof.npz", allow_pickle=True)
    cm = np.load(ROOT / "artifacts/blend_search/oof_blend_centered_meta_round2_best.npz", allow_pickle=True)
    cw = cm["class_weights"].astype(float)
    oof_fids = o_we["file_id"].astype(int)
    oof_labels = o_we["label"].astype(int)
    oof_uids = o_we["user_id"].astype(str)
    tr, st = estimate_transition_model(y=oof_labels, classes=classes, file_ids=oof_fids, user_ids=oof_uids, alpha=alpha)

    test_uids = t_we["user_id"].astype(str)
    pred_test = viterbi_predict_by_user(
        proba=blended_test, classes=classes, file_ids=ref_fids, user_ids=test_uids,
        class_weights=cw, transition=tr, start=st, beta=beta, stay_bonus=0.0,
    )

    # SELF-PROMOTE c2 (top 30, thresh 0.30), c3 (top 15, thresh 0.55)
    pred_test = pred_test.copy()
    for k, top_n, thr in [(2, 30, 0.30), (3, 15, 0.55)]:
        p_k = blended_test[:, k]
        cands = np.where(pred_test != k)[0]
        cands_sorted = cands[np.argsort(-p_k[cands])]
        eligible = [int(i) for i in cands_sorted if p_k[i] >= thr][:top_n]
        pred_test[eligible] = k

    # Write
    out_df = pd.DataFrame({"Id": ref_fids.astype(int), "Label": pred_test.astype(int)})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    out_df = out_df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    out_df["Label"] = out_df["Label"].astype(int)
    out_df.to_csv(out_csv, index=False)

    sha = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    label_counts = out_df["Label"].value_counts().sort_index().to_dict()

    anchor = pd.read_csv(ROOT / "submissions/submission_h8_v3_sslv2_weighted.csv")
    anchor = anchor.set_index("Id").reindex(out_df["Id"])
    diff = int((anchor["Label"].to_numpy() != out_df["Label"].to_numpy()).sum())

    meta = {
        "mechanism": "5-source HOS-12-optimal blend + self-promote c2 (top 30, thresh 0.30) and c3 (top 15, thresh 0.55)",
        "untested_on_public": True,
        "weights": weights,
        "alpha": alpha,
        "beta": beta,
        "hos12_macro_f1": 0.8006,
        "hos12_per_class_f1": {2: 0.442, 3: 0.718, 4: 0.883, 5: 0.855},
        "diff_from_anchor_0p8248": diff,
        "label_counts": {int(k): int(v) for k, v in label_counts.items()},
        "sha256": sha,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_csv}")
    print(f"SHA: {sha}")
    print(f"Label counts: {label_counts}")
    print(f"Diff vs 0.8248 anchor: {diff}")


if __name__ == "__main__":
    main()
