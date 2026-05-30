#!/usr/bin/env python3
"""Build a hybrid CSV combining SSL pipeline (0.8175) and consensus baseline (0.8156).

Strategy: keep SSL prediction by default; recover baseline's rare-class (2 or 5)
prediction only when SSL's soft proba for that rare class is above threshold.

This protects the SSL pipeline's confident corrections (mostly class-1 over class-2
where SSL was right) while restoring baseline's rare-class predictions that SSL
itself considers plausible (high enough soft proba but not the argmax).
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-csv", type=Path, default=Path("submissions/submission_inception_ssl_blend_viterbi.csv"))
    p.add_argument("--baseline-csv", type=Path, default=Path("submissions/submission_consensus_oof_greedy.csv"))
    p.add_argument("--ssl-proba-npz", type=Path, default=Path("artifacts/blend_search/test_blend_centered_meta_with_inception_ssl.npz"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--class2-recover-threshold",
        type=float,
        default=0.27,
        help="Restore baseline class-2 prediction iff SSL's softmax for class 2 >= this value.",
    )
    p.add_argument(
        "--class5-recover-threshold",
        type=float,
        default=0.25,
        help="Restore baseline class-5 prediction iff SSL's softmax for class 5 >= this value.",
    )
    p.add_argument(
        "--max-class2-recovered",
        type=int,
        default=30,
        help="Cap the number of class-2 recoveries (top by SSL class-2 proba).",
    )
    p.add_argument(
        "--max-class5-recovered",
        type=int,
        default=10,
        help="Cap the number of class-5 recoveries (top by SSL class-5 proba).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ssl_csv = pd.read_csv(args.ssl_csv)
    base_csv = pd.read_csv(args.baseline_csv)
    assert (ssl_csv["Id"].values == base_csv["Id"].values).all()

    npz = np.load(args.ssl_proba_npz, allow_pickle=True)
    proba = npz["proba"].astype(float)  # (6849, 6)
    file_ids = npz["file_id"].astype(int)
    # Make sure ordering matches the CSV
    if not np.array_equal(file_ids, ssl_csv["Id"].astype(int).to_numpy()):
        idx = {int(f): i for i, f in enumerate(file_ids)}
        order = np.array([idx[int(i)] for i in ssl_csv["Id"]], dtype=int)
        proba = proba[order]
        file_ids = file_ids[order]

    ssl_lbl = ssl_csv["Label"].astype(int).to_numpy()
    base_lbl = base_csv["Label"].astype(int).to_numpy()

    pred = ssl_lbl.copy()

    # Recover class-2 where baseline says 2, SSL says other, and SSL's class-2 proba is high
    disagree_b2 = (base_lbl == 2) & (ssl_lbl != 2)
    cand_b2_idx = np.where(disagree_b2 & (proba[:, 2] >= args.class2_recover_threshold))[0]
    cand_b2_idx_sorted = cand_b2_idx[np.argsort(-proba[cand_b2_idx, 2])]
    cand_b2_idx_top = cand_b2_idx_sorted[: args.max_class2_recovered]
    pred[cand_b2_idx_top] = 2
    print(f"Class-2 recoveries: {len(cand_b2_idx_top)} (threshold {args.class2_recover_threshold}, cap {args.max_class2_recovered})")
    if len(cand_b2_idx_top) > 0:
        print(f"  Restored proba range: {proba[cand_b2_idx_top, 2].min():.3f} – {proba[cand_b2_idx_top, 2].max():.3f}")

    # Recover class-5 similarly
    disagree_b5 = (base_lbl == 5) & (ssl_lbl != 5)
    cand_b5_idx = np.where(disagree_b5 & (proba[:, 5] >= args.class5_recover_threshold))[0]
    cand_b5_idx_sorted = cand_b5_idx[np.argsort(-proba[cand_b5_idx, 5])]
    cand_b5_idx_top = cand_b5_idx_sorted[: args.max_class5_recovered]
    pred[cand_b5_idx_top] = 5
    print(f"Class-5 recoveries: {len(cand_b5_idx_top)} (threshold {args.class5_recover_threshold}, cap {args.max_class5_recovered})")
    if len(cand_b5_idx_top) > 0:
        print(f"  Restored proba range: {proba[cand_b5_idx_top, 5].min():.3f} – {proba[cand_b5_idx_top, 5].max():.3f}")

    out = pd.DataFrame({"Id": ssl_csv["Id"].astype(int), "Label": pred.astype(int)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    label_counts = out["Label"].value_counts().sort_index().to_dict()
    ssl_counts = pd.Series(ssl_lbl).value_counts().sort_index().to_dict()
    base_counts = pd.Series(base_lbl).value_counts().sort_index().to_dict()
    diff_from_ssl = (out["Label"].astype(int) != ssl_lbl).sum()
    diff_from_base = (out["Label"].astype(int) != base_lbl).sum()

    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"\nWrote {args.output}")
    print(f"SHA256: {sha}")
    print(f"Label counts: {label_counts}")
    print(f"Diff from SSL: {diff_from_ssl} rows")
    print(f"Diff from baseline: {diff_from_base} rows")
    print(f"SSL counts:      {ssl_counts}")
    print(f"Baseline counts: {base_counts}")


if __name__ == "__main__":
    main()
