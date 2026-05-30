#!/usr/bin/env python3
"""Apply per-user softmax centering — zero-parameter test-time adaptation.

For each user, compute the per-class softmax mean across all of their windows;
subtract that user-prior from each window's softmax; clip negatives at 0 and
re-normalize. Removes per-user systematic bias without learned parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-user softmax centering.")
    parser.add_argument("--input", type=Path, required=True, help="npz with proba + user_id + file_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Centering strength: 1.0 = full subtract, 0.5 = half, 0 = identity.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="Min softmax value before renormalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.input, allow_pickle=True)
    proba = d["proba"].astype(np.float64)
    users = d["user_id"].astype(str)
    file_ids = d["file_id"].astype(int)
    n, c = proba.shape

    centered = proba.copy()
    unique_users = np.unique(users)
    print(f"Applying per-user centering across {len(unique_users)} users, alpha={args.alpha}")
    for u in unique_users:
        mask = users == u
        n_u = mask.sum()
        if n_u < 5:
            # Too few windows; skip to avoid noise dominance
            continue
        prior = proba[mask].mean(axis=0)  # (6,)
        centered[mask] = centered[mask] - args.alpha * prior

    # Clip and re-normalize
    centered = np.maximum(centered, args.floor)
    row_sum = centered.sum(axis=1, keepdims=True)
    centered = centered / np.maximum(row_sum, 1e-8)

    # Compare label distribution before/after
    pred_before = proba.argmax(axis=1)
    pred_after = centered.argmax(axis=1)
    n_changed = int((pred_before != pred_after).sum())
    print(f"Argmax changed for {n_changed}/{n} rows ({n_changed / n * 100:.2f}%)")
    print(f"Label counts before: {pd.Series(pred_before).value_counts().sort_index().to_dict()}")
    print(f"Label counts after:  {pd.Series(pred_after).value_counts().sort_index().to_dict()}")

    out = dict(d)
    out["proba"] = centered.astype(np.float32)
    out["alpha_centering"] = float(args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
