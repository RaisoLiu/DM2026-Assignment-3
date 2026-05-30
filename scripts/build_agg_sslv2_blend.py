#!/usr/bin/env python3
"""Aggressive Generator candidate 1: SSL v2 blended into the 4-source
weighted ensemble at lambda=0.35, Viterbi smoothed (alpha=1.0, beta=0.18).

HOS-12 macro: 0.7858 (target 0.7849).

This mechanism has NEVER been validated on public — the previous SSL v2 attempt
(submission_v9_sslv2_blend.csv, 0.8130 public) used a different weight stack
(40% SSL v2 + 45% cm_v + 10% catch22 + 5% inc_ssl_v1 + hybrid v3) and regressed
−0.0110. This time: blend with the existing 4-source ensemble (which uses
ssl_v1) at a smaller-but-meaningful 35% weight, NO hybrid layer.
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


def main() -> None:
    out_csv = ROOT / "submissions/agg_sslv2_blend035_viterbi.csv"
    out_meta = ROOT / "submissions/agg_sslv2_blend035_viterbi_metadata.json"

    # --- TEST-SPACE ---
    test_we = np.load(ROOT / "artifacts/weighted_ensemble/test.npz", allow_pickle=True)
    test_v2 = np.load(ROOT / "artifacts/inception_full_v2/test_avg.npz", allow_pickle=True)
    classes = test_we["classes"].astype(int)

    # Align by file_id
    we_fids = test_we["file_id"].astype(int)
    v2_idx = {int(f): i for i, f in enumerate(test_v2["file_id"].astype(int))}
    order = np.array([v2_idx[int(f)] for f in we_fids], dtype=int)
    v2_proba_test = test_v2["proba"][order].astype(float)
    we_proba_test = test_we["proba"].astype(float)

    lam = 0.35
    test_blended = (1 - lam) * we_proba_test + lam * v2_proba_test
    test_blended = test_blended / test_blended.sum(axis=1, keepdims=True)

    # --- OOF (for fitting Viterbi transitions on full train) ---
    oof_we = np.load(ROOT / "artifacts/weighted_ensemble/oof.npz", allow_pickle=True)
    oof_v2 = np.load(ROOT / "artifacts/inception_oof_ssl_v2/oof_inception_avg5.npz", allow_pickle=True)
    cm = np.load(ROOT / "artifacts/blend_search/oof_blend_centered_meta_round2_best.npz", allow_pickle=True)
    cw = cm["class_weights"].astype(float)

    oof_fids = oof_we["file_id"].astype(int)
    oof_labels = oof_we["label"].astype(int)
    oof_uids = oof_we["user_id"].astype(str)
    v2_idx_oof = {int(f): i for i, f in enumerate(oof_v2["file_id"].astype(int))}
    order_oof = np.array([v2_idx_oof[int(f)] for f in oof_fids], dtype=int)
    v2_proba_oof = oof_v2["proba"][order_oof].astype(float)
    we_proba_oof = oof_we["proba"].astype(float)

    # Estimate Viterbi on full training (test will use transitions tuned from train)
    alpha, beta = 1.0, 0.18
    tr, st = estimate_transition_model(
        y=oof_labels, classes=classes, file_ids=oof_fids, user_ids=oof_uids, alpha=alpha,
    )

    # Test users: use the test set's user_ids
    test_uids = test_we["user_id"].astype(str)
    pred_test = viterbi_predict_by_user(
        proba=test_blended, classes=classes, file_ids=we_fids, user_ids=test_uids,
        class_weights=cw, transition=tr, start=st, beta=beta, stay_bonus=0.0,
    )

    # Write submission, with sample ordering
    out_df = pd.DataFrame({"Id": we_fids.astype(int), "Label": pred_test.astype(int)})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    out_df = out_df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    out_df["Label"] = out_df["Label"].astype(int)
    out_df.to_csv(out_csv, index=False)

    sha = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    label_counts = out_df["Label"].value_counts().sort_index().to_dict()

    # Compare to 0.8248 anchor
    anchor = pd.read_csv(ROOT / "submissions/submission_h8_v3_sslv2_weighted.csv")
    anchor = anchor.set_index("Id").reindex(out_df["Id"])
    diff = int((anchor["Label"].to_numpy() != out_df["Label"].to_numpy()).sum())

    meta = {
        "mechanism": "ssl_v2 blended at lam=0.35 into 4-source weighted ensemble; Viterbi alpha=1.0 beta=0.18",
        "untested_on_public": True,
        "lambda": lam,
        "alpha": alpha,
        "beta": beta,
        "hos12_macro_f1": 0.7858,
        "hos12_per_class_f1": {2: 0.384, 5: 0.844, 3: 0.689},
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
