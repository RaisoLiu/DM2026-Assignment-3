#!/usr/bin/env python3
"""Hybrid recovery 3.0 — extends make_ssl_hybrid_submission.py with class-3 recovery.

Strategy: keep new-pipeline prediction by default; recover anchor's rare-class
(2, 3, or 5) prediction only when new-pipeline's *soft* proba for that rare class
exceeds a per-class threshold. Confidence-gated, low-volume — matches yesterday's
proven approach (0.8156→0.8240) extended to one more rare class.

Anchor at test time = yesterday's 0.8240 hybrid CSV (submission_ssl_hybrid_recover.csv).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--new-proba-npz",
        type=Path,
        required=True,
        help="npz with new-pipeline soft probas (centered ensemble), keys: proba, file_id",
    )
    p.add_argument(
        "--anchor-csv",
        type=Path,
        default=Path("submissions/submission_ssl_hybrid_recover.csv"),
        help="Anchor CSV — yesterday's 0.8240 hybrid (or any verified strong CSV).",
    )
    p.add_argument("--output", type=Path, required=True)
    # Thresholds
    p.add_argument("--c2-threshold", type=float, default=0.20)
    p.add_argument("--c3-threshold", type=float, default=0.25)
    p.add_argument("--c5-threshold", type=float, default=0.25)
    p.add_argument("--max-c2-recovered", type=int, default=35)
    p.add_argument("--max-c3-recovered", type=int, default=25)
    p.add_argument("--max-c5-recovered", type=int, default=10)
    return p.parse_args()


def recover(pred: np.ndarray, target_class: int, anchor_lbl: np.ndarray, new_proba: np.ndarray,
            threshold: float, cap: int, name: str) -> int:
    disagree = (anchor_lbl == target_class) & (pred != target_class)
    cand_idx = np.where(disagree & (new_proba[:, target_class] >= threshold))[0]
    cand_sorted = cand_idx[np.argsort(-new_proba[cand_idx, target_class])]
    cand_top = cand_sorted[:cap]
    pred[cand_top] = target_class
    n = len(cand_top)
    if n > 0:
        print(f"Class-{target_class} ({name}) recoveries: {n} (threshold {threshold:.3f}, cap {cap}) — proba range {new_proba[cand_top, target_class].min():.3f}–{new_proba[cand_top, target_class].max():.3f}")
    else:
        print(f"Class-{target_class} ({name}) recoveries: 0")
    return n


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.new_proba_npz, allow_pickle=True)
    new_proba = npz["proba"].astype(float)
    file_ids = npz["file_id"].astype(int)

    # The "new-pipeline labels" are simply argmax of new_proba
    new_pred = new_proba.argmax(axis=1)

    # Load anchor (yesterday's 0.8240)
    anchor_csv = pd.read_csv(args.anchor_csv)
    anchor_idx = {int(i): k for k, i in enumerate(anchor_csv["Id"])}
    anchor_lbl = np.array([anchor_csv["Label"].iloc[anchor_idx[int(f)]] for f in file_ids], dtype=int)

    pred = new_pred.copy()
    n_c2 = recover(pred, 2, anchor_lbl, new_proba, args.c2_threshold, args.max_c2_recovered, "rare burst")
    n_c3 = recover(pred, 3, anchor_lbl, new_proba, args.c3_threshold, args.max_c3_recovered, "rare activity")
    n_c5 = recover(pred, 5, anchor_lbl, new_proba, args.c5_threshold, args.max_c5_recovered, "rare burst")

    # Write CSV
    out = pd.DataFrame({"Id": file_ids.astype(int), "Label": pred.astype(int)})
    out.to_csv(args.output, index=False)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()

    label_counts = out["Label"].value_counts().sort_index().to_dict()
    new_counts = pd.Series(new_pred).value_counts().sort_index().to_dict()
    anchor_counts = pd.Series(anchor_lbl).value_counts().sort_index().to_dict()
    n_diff_new = int((pred != new_pred).sum())
    n_diff_anchor = int((pred != anchor_lbl).sum())

    print(f"\nWrote {args.output}")
    print(f"SHA256: {sha}")
    print(f"Label counts: {label_counts}")
    print(f"Diff from new (raw):    {n_diff_new} rows")
    print(f"Diff from anchor:       {n_diff_anchor} rows")
    print(f"New (raw) counts:    {new_counts}")
    print(f"Anchor counts:       {anchor_counts}")


if __name__ == "__main__":
    main()
