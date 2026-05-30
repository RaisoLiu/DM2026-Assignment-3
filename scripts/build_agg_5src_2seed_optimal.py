#!/usr/bin/env python3
"""Aggressive Generator candidate 4: 5-source weighted ensemble with weights
optimized on a HOS-12 evaluation that uses ONLY the 2 SSL v2 seeds we also
have at test time (test_proba_seed2026 + test_proba_seed2027).

This is the FAIREST candidate because the HOS-12 score reflects exactly what
the test prediction is doing (no optimistic 5-seed OOF average vs 2-seed test).

Weights: v1=0.007, v2=0.488, cm=0.237, c22l=0.029, c22x=0.239
HOS-12 macro: 0.8005 (test-matched). c2 F1 = 0.422.
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
    out_csv = ROOT / "submissions/agg_5src_2seed_optimal.csv"
    out_meta = ROOT / "submissions/agg_5src_2seed_optimal_metadata.json"

    weights = {"ssl_v1": 0.007, "ssl_v2": 0.488, "centered_meta": 0.237, "catch22_lgbm": 0.029, "catch22_xgb": 0.239}
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
    tr, st = estimate_transition_model(
        y=o_we["label"].astype(int), classes=classes,
        file_ids=o_we["file_id"].astype(int), user_ids=o_we["user_id"].astype(str), alpha=alpha,
    )
    test_uids = t_we["user_id"].astype(str)
    pred_test = viterbi_predict_by_user(
        proba=blended_test, classes=classes, file_ids=ref_fids, user_ids=test_uids,
        class_weights=cw, transition=tr, start=st, beta=beta, stay_bonus=0.0,
    )

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
        "mechanism": "5-source weighted ensemble with weights optimized via test-matched 2-seed SSL v2 HOS-12 search; Viterbi smoothing",
        "untested_on_public": True,
        "weights": weights,
        "alpha": alpha,
        "beta": beta,
        "hos12_macro_f1": 0.8005,
        "hos12_per_class_f1": {0: 0.984, 1: 0.923, 2: 0.422, 3: 0.704, 4: 0.892, 5: 0.870},
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
