#!/usr/bin/env python3
"""Sample a fresh 12-user HOS that excludes the original 8-user HOS from
holdout8_seed2026.csv. Stratified on per-user class-2 incidence so the new HOS
contains at least 40 class-2 windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fresh 12-user HOS.")
    parser.add_argument(
        "--source-folds",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026.csv"),
        help="Full 60-user fold CSV (any seed; we use file_id/user_id/label).",
    )
    parser.add_argument(
        "--existing-holdout",
        type=Path,
        default=Path("artifacts/folds/holdout8_seed2026.csv"),
        help="Existing HOS to exclude.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/folds/holdout12_seed2027.csv"),
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--min-c2", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(args.source_folds)
    existing = pd.read_csv(args.existing_holdout)
    existing_users = set(existing["user_id"].astype(str).tolist())
    print(f"Existing HOS users ({len(existing_users)}): {sorted(existing_users)}")

    # Eligible users
    eligible = src[~src["user_id"].astype(str).isin(existing_users)].copy()
    print(f"Eligible users for fresh HOS: {eligible['user_id'].nunique()}")

    # Per-user class-2 incidence
    users_c2 = (
        eligible.groupby("user_id")["label"]
        .apply(lambda s: int((s == 2).sum()))
        .sort_values(ascending=False)
    )
    print(f"Per-user class-2 counts (head):\n{users_c2.head().to_string()}")

    # Stratified by class-2 incidence: bin then sample
    rng = np.random.default_rng(args.seed)
    for trial in range(50):
        bands = {
            "high": users_c2[users_c2 >= 6].index.tolist(),
            "mid": users_c2[(users_c2 >= 3) & (users_c2 <= 5)].index.tolist(),
            "low": users_c2[(users_c2 >= 1) & (users_c2 <= 2)].index.tolist(),
            "zero": users_c2[users_c2 == 0].index.tolist(),
        }
        total = sum(len(v) for v in bands.values())
        picks = []
        for name, members in bands.items():
            quota = max(1, round(args.n_users * len(members) / total)) if members else 0
            quota = min(quota, len(members))
            if quota > 0:
                chosen = list(rng.choice(members, size=quota, replace=False))
                picks.extend(chosen)
        # Pad or trim to n_users
        while len(picks) < args.n_users:
            remaining = [u for u in eligible["user_id"].astype(str).unique() if u not in picks]
            picks.append(str(rng.choice(remaining)))
        picks = picks[: args.n_users]

        # Check c2 count
        c2_total = sum(int(users_c2.loc[u]) for u in picks)
        if c2_total >= args.min_c2:
            break
        print(f"Trial {trial}: c2 total = {c2_total} (need {args.min_c2}); reseeding")
    else:
        raise SystemExit("Failed to find fresh HOS with sufficient class-2")

    picks_sorted = sorted(picks)
    print(f"\nFresh HOS users ({len(picks_sorted)}): {picks_sorted}")
    print(f"Total class-2 windows: {c2_total}")

    hos = src[src["user_id"].astype(str).isin(picks_sorted)].copy().reset_index(drop=True)
    hos["fold"] = 0
    hos.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(hos)} rows")
    class_counts = hos["label"].value_counts().sort_index().to_dict()
    print(f"HOS class counts: {class_counts}")

    summary = {
        "holdout_users": picks_sorted,
        "n_rows": int(len(hos)),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "excluded_users": sorted(existing_users),
        "seed": args.seed,
    }
    Path(str(args.output).replace(".csv", "_summary.json")).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
