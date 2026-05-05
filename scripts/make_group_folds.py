#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed user-grouped folds.")
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_cache, usecols=["file_id", "user_id", "label"])
    y = frame["label"].astype(int).to_numpy()
    groups = frame["user_id"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_ids = np.full(len(frame), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(frame, y, groups), start=1):
        fold_ids[valid_idx] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("Some rows were not assigned to a fold")
    out = frame.copy()
    out["fold"] = fold_ids
    out.to_csv(args.output, index=False)
    summary = out.groupby("fold")["label"].value_counts().unstack(fill_value=0).sort_index()
    print(f"Wrote {args.output}")
    print(summary.to_string())


if __name__ == "__main__":
    main()

