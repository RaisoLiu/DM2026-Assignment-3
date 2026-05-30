#!/usr/bin/env python3
"""Build held-out 8-user split + 52-user repeated 5-fold.

The HOS is sampled by stratified greedy assignment on per-user class-2
incidence so the 8 users together hold at least the configured number of
class-2 windows (default 30). Remaining 52 users are split with
StratifiedGroupKFold for each requested seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HOS + repeated 5-fold splits.")
    parser.add_argument(
        "--source-folds",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026.csv"),
        help="Existing fold CSV with file_id,user_id,label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/folds"),
    )
    parser.add_argument("--holdout-users", type=int, default=8)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--min-c2-holdout", type=int, default=30)
    parser.add_argument("--cv-seeds", type=str, default="2026,2027")
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def stratify_users_by_c2(users_c2: pd.Series, n_holdout: int, rng: np.random.Generator) -> list[str]:
    """Stratified sample of users across class-2 incidence bands."""
    bands: dict[str, list[str]] = {
        "zero": users_c2[users_c2 == 0].index.tolist(),
        "low": users_c2[(users_c2 >= 1) & (users_c2 <= 2)].index.tolist(),
        "mid": users_c2[(users_c2 >= 3) & (users_c2 <= 5)].index.tolist(),
        "high": users_c2[users_c2 >= 6].index.tolist(),
    }
    total_users = sum(len(v) for v in bands.values())
    pick: list[str] = []
    leftover = n_holdout
    quotas: list[tuple[str, int]] = []
    for name, members in bands.items():
        if not members:
            continue
        share = max(1, round(n_holdout * len(members) / total_users))
        quotas.append((name, min(share, len(members))))
    extra = n_holdout - sum(q for _, q in quotas)
    if extra > 0:
        sorted_bands = sorted(quotas, key=lambda kv: len(bands[kv[0]]) - kv[1], reverse=True)
        for i in range(extra):
            name, q = sorted_bands[i % len(sorted_bands)]
            quotas[[idx for idx, (n, _) in enumerate(quotas) if n == name][0]] = (name, q + 1)
    elif extra < 0:
        sorted_bands = sorted(quotas, key=lambda kv: kv[1], reverse=True)
        for i in range(-extra):
            name, q = sorted_bands[i % len(sorted_bands)]
            if q > 1:
                quotas[[idx for idx, (n, _) in enumerate(quotas) if n == name][0]] = (name, q - 1)
    for name, quota in quotas:
        members = bands[name]
        chosen = list(rng.choice(members, size=min(quota, len(members)), replace=False))
        pick.extend(chosen)
        leftover -= len(chosen)
    if leftover > 0:
        pool = [u for u in users_c2.index if u not in set(pick)]
        pick.extend(rng.choice(pool, size=leftover, replace=False).tolist())
    elif leftover < 0:
        pick = pick[:n_holdout]
    return pick


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(args.source_folds)
    if not {"file_id", "user_id", "label"}.issubset(src.columns):
        raise SystemExit(f"Missing required columns in {args.source_folds}: {src.columns.tolist()}")

    user_class_counts = (
        src.groupby(["user_id", "label"]).size().unstack(fill_value=0).sort_index()
    )
    print(f"Loaded {len(src)} rows / {user_class_counts.shape[0]} users")
    print("Per-user class counts (head):")
    print(user_class_counts.head().to_string())

    rng = np.random.default_rng(args.holdout_seed)
    users_c2 = user_class_counts.get(2, pd.Series(0, index=user_class_counts.index)).rename("c2")

    for trial in range(50):
        holdout_users = stratify_users_by_c2(users_c2, args.holdout_users, rng)
        c2_total = int(users_c2.loc[holdout_users].sum())
        if c2_total >= args.min_c2_holdout:
            break
        print(f"Trial {trial}: c2_total={c2_total} below {args.min_c2_holdout}; reseeding")
    else:
        raise SystemExit("Failed to satisfy min-c2-holdout after 50 trials")

    holdout_users_sorted = sorted(holdout_users)
    print(f"\nHeld-out users ({len(holdout_users_sorted)}): {holdout_users_sorted}")
    print(f"Held-out class-2 windows: {c2_total}")
    hos = src[src["user_id"].isin(holdout_users_sorted)].copy().reset_index(drop=True)
    hos["fold"] = 0
    hos_path = args.output_dir / f"holdout{args.holdout_users}_seed{args.holdout_seed}.csv"
    hos.to_csv(hos_path, index=False)
    print(f"Wrote HOS to {hos_path}: {len(hos)} rows")
    hos_class_summary = hos["label"].value_counts().sort_index()
    print(f"HOS class counts:\n{hos_class_summary.to_string()}")

    train_part = src[~src["user_id"].isin(holdout_users_sorted)].copy().reset_index(drop=True)
    print(f"\n52-user training partition: {len(train_part)} rows / {train_part['user_id'].nunique()} users")

    cv_seeds = [int(s) for s in args.cv_seeds.split(",") if s.strip()]
    train_class_summary = train_part["label"].value_counts().sort_index()
    print(f"Train-CV class counts:\n{train_class_summary.to_string()}")

    for seed in cv_seeds:
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        y = train_part["label"].astype(int).to_numpy()
        groups = train_part["user_id"].astype(str).to_numpy()
        fold_ids = np.full(len(train_part), -1, dtype=int)
        for fold, (_, valid_idx) in enumerate(cv.split(train_part, y, groups), start=1):
            fold_ids[valid_idx] = fold
        if (fold_ids < 0).any():
            raise RuntimeError(f"Unassigned rows for seed {seed}")
        out = train_part.copy()
        out["fold"] = fold_ids
        out_path = args.output_dir / f"sgkf_seed{seed}_train{len(train_part['user_id'].unique())}.csv"
        out.to_csv(out_path, index=False)
        per_fold_classes = (
            out.groupby("fold")["label"].value_counts().unstack(fill_value=0).sort_index()
        )
        print(f"\nSeed {seed} → {out_path}")
        print(per_fold_classes.to_string())

    summary = {
        "holdout_users": holdout_users_sorted,
        "holdout_class2_count": c2_total,
        "holdout_class_counts": {int(k): int(v) for k, v in hos_class_summary.to_dict().items()},
        "train_users": sorted(train_part["user_id"].unique().tolist()),
        "n_train_rows": int(len(train_part)),
        "cv_seeds": cv_seeds,
        "n_splits": args.n_splits,
        "source_folds": str(args.source_folds),
    }
    summary_path = args.output_dir / f"holdout{args.holdout_users}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
