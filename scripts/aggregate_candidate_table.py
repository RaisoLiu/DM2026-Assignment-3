#!/usr/bin/env python3
"""Aggregate worker-produced candidate CSVs into a Leader-decision table.

NOTE: Submissions contain TEST file_ids (11021-17869) while HOS-12 fold contains
TRAIN file_ids (held-out users). HOS-12 cannot be computed from submission CSV
directly — workers must compute HOS-12 from PROBAS on held-out train rows.
This script captures structural diffs only (vs anchor, c2 counts, label dist);
workers attach their own HOS-12 scores in their reports.

Usage:
    .venv/bin/python scripts/aggregate_candidate_table.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ANCHOR_0_8248 = Path("submissions/submission_h8_v3_sslv2_weighted.csv")
HOS12_FOLD = Path("artifacts/folds/holdout12_seed2027.csv")
PREFIXES = ("agg_", "cons_", "synth_")


def hos12_score(submission_csv: Path, hos12: pd.DataFrame) -> dict:
    """Submissions only have TEST file_ids; cannot score directly on HOS-12 train rows.
    Returns NaN to signal: workers must compute HOS-12 from probas, not CSVs."""
    return {"hos12_macro": float("nan"), "n_matched": 0,
            "hos12_c0": float("nan"), "hos12_c1": float("nan"), "hos12_c2": float("nan"),
            "hos12_c3": float("nan"), "hos12_c4": float("nan"), "hos12_c5": float("nan")}


def main() -> None:
    anchor = pd.read_csv(ANCHOR_0_8248).set_index("Id")["Label"].astype(int)
    hos12 = pd.read_csv(HOS12_FOLD)
    print(f"HOS-12 fold: {len(hos12)} rows from {hos12['user_id'].nunique()} users")

    rows = []
    sub_dir = Path("submissions")
    candidates = sorted(
        f for f in sub_dir.glob("*.csv")
        if any(f.name.startswith(p) for p in PREFIXES)
    )
    # Also include the strongest anchors for reference
    reference_anchors = [
        Path("submissions/submission_h8_v3_sslv2_weighted.csv"),
        Path("submissions/submission_h8_uncertain_refined.csv"),
        Path("submissions/submission_ssl_hybrid_recover.csv"),
    ]

    for f in reference_anchors + candidates:
        if not f.exists():
            continue
        sub = pd.read_csv(f).set_index("Id")["Label"].astype(int)
        if len(sub) != len(anchor):
            print(f"WARN {f.name}: size mismatch ({len(sub)} vs {len(anchor)})")
            continue
        diff = int((sub != anchor).sum())
        counts = sub.value_counts().sort_index()
        c2_count = int(counts.get(2, 0))
        hos = hos12_score(f, hos12)
        rows.append({
            "candidate": f.name,
            "is_anchor": f.name.startswith("submission_") and not any(f.name.startswith(f"submission_{p}") for p in PREFIXES),
            "diff_vs_0.8248": diff,
            "c2_count": c2_count,
            "hos12_macro": hos["hos12_macro"],
            "hos12_c2": hos.get("hos12_c2", float("nan")),
            "hos12_c4": hos.get("hos12_c4", float("nan")),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("hos12_macro", ascending=False)
    print()
    print("# Candidate decision table")
    print()
    print("| candidate | diff_vs_0.8248 | c2_count | HOS-12 macro | HOS-12 c2 | HOS-12 c4 |")
    print("|---|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        anchor_marker = " 📌" if r["is_anchor"] else ""
        print(f"| `{r['candidate']}`{anchor_marker} | {r['diff_vs_0.8248']} | {r['c2_count']} | {r['hos12_macro']:.4f} | {r['hos12_c2']:.3f} | {r['hos12_c4']:.3f} |")

    print()
    print(f"_{len(df)} candidates, {sum(df['is_anchor'])} anchor reference(s), {sum(~df['is_anchor'])} new candidate(s)_")


if __name__ == "__main__":
    main()
