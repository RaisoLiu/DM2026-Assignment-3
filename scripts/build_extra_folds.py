#!/usr/bin/env python3
"""Emit additional fold files for seeds 2028/2029/2030 (5-fold StratifiedGroupKFold)
and Leave-N-Users-Out splits for variance estimation.

Reuses the same 52-user training partition as artifacts/folds/sgkf_seed2026_train52.csv
(so the 8 held-out users are preserved for comparability with yesterday's 0.8240 baseline).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extra fold files for the 0.85 plan.")
    parser.add_argument(
        "--source-folds",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
        help="Existing 52-user fold CSV (any seed; we only need the row set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/folds"),
    )
    parser.add_argument(
        "--sgkf-seeds",
        default="2028,2029,2030",
    )
    parser.add_argument(
        "--lnuo-seeds",
        default="2026,2027,2028,2029,2030,2031",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lnuo-n-users", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(args.source_folds)
    if not {"file_id", "user_id", "label"}.issubset(src.columns):
        raise SystemExit(f"Missing required columns in {args.source_folds}")
    src = src[["file_id", "user_id", "label"]].copy()
    print(f"Loaded {len(src)} rows / {src['user_id'].nunique()} users from {args.source_folds}")

    # Repeated 5-fold StratifiedGroupKFold with extra seeds
    y = src["label"].astype(int).to_numpy()
    groups = src["user_id"].astype(str).to_numpy()
    for seed_str in [s.strip() for s in args.sgkf_seeds.split(",") if s.strip()]:
        seed = int(seed_str)
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        fold_ids = np.full(len(src), -1, dtype=int)
        for fold, (_, valid_idx) in enumerate(cv.split(src, y, groups), start=1):
            fold_ids[valid_idx] = fold
        out = src.copy()
        out["fold"] = fold_ids
        out_path = args.output_dir / f"sgkf_seed{seed}_train52.csv"
        out.to_csv(out_path, index=False)
        per_fold = out.groupby("fold")["label"].value_counts().unstack(fill_value=0).sort_index()
        print(f"\nSeed {seed} 5-fold → {out_path}")
        print(per_fold.to_string())

    # LNUO splits: each "fold" leaves out a random subset of N users
    unique_users = sorted(src["user_id"].astype(str).unique())
    print(f"\n--- LNUO splits ({args.lnuo_n_users} users left out per fold) ---")
    for seed_str in [s.strip() for s in args.lnuo_seeds.split(",") if s.strip()]:
        seed = int(seed_str)
        rng = np.random.default_rng(seed)
        held_users = sorted(rng.choice(unique_users, size=args.lnuo_n_users, replace=False).tolist())
        out = src.copy()
        out["fold"] = 0  # placeholder
        held_mask = out["user_id"].astype(str).isin(held_users)
        out.loc[held_mask, "fold"] = 1
        out_path = args.output_dir / f"lnuo{args.lnuo_n_users}_seed{seed}.csv"
        out.to_csv(out_path, index=False)
        print(
            f"  Seed {seed} → {out_path}: held out users = {held_users}, "
            f"{held_mask.sum()} rows ({held_mask.mean() * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
