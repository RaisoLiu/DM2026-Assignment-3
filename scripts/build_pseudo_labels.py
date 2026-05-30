#!/usr/bin/env python3
"""Build consensus pseudo-labels for test windows.

Takes three reference submission CSVs and emits pseudo-labels only for rows where
all three agree on the same label. The pseudo-label confidence_score is set to 1.0
when all three agree (default). Optionally, a 2-of-3 agreement set can also be
emitted with lower confidence.

Default sources (yesterday's best three): 0.8156, 0.8175, 0.8240 public CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consensus pseudo-labels for test.")
    parser.add_argument(
        "--source-csvs",
        nargs="+",
        default=[
            "submissions/submission_consensus_oof_greedy.csv",
            "submissions/submission_inception_ssl_blend_viterbi.csv",
            "submissions/submission_ssl_hybrid_recover.csv",
        ],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pseudo/pseudo_labels_consensus3.csv"),
    )
    parser.add_argument(
        "--min-agree",
        type=int,
        default=None,
        help="Minimum number of sources that must agree. Default = all sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load reference CSVs
    dfs = []
    for path in args.source_csvs:
        df = pd.read_csv(path)
        if not {"Id", "Label"}.issubset(df.columns):
            raise SystemExit(f"{path} missing Id/Label columns")
        dfs.append(df.set_index("Id")["Label"].astype(int))
        print(f"Loaded {path}: {len(df)} rows")

    # Align all on first DataFrame's Id index
    aligned = pd.concat(dfs, axis=1).dropna().astype(int)
    aligned.columns = [Path(p).stem for p in args.source_csvs]
    print(f"\nAligned: {len(aligned)} rows × {aligned.shape[1]} sources")

    # Per-row majority
    arr = aligned.to_numpy()
    n_sources = arr.shape[1]
    min_agree = args.min_agree if args.min_agree is not None else n_sources

    # Compute per-row mode and agreement count
    pseudo_labels = []
    n_pseudo = 0
    n_full_consensus = 0
    n_partial = 0
    n_disagree = 0
    for i, row in enumerate(arr):
        unique, counts = np.unique(row, return_counts=True)
        max_count = counts.max()
        if max_count == n_sources:
            n_full_consensus += 1
        elif max_count >= 2:
            n_partial += 1
        else:
            n_disagree += 1
        if max_count >= min_agree:
            mode_label = int(unique[counts.argmax()])
            pseudo_labels.append({
                "Id": int(aligned.index[i]),
                "Label": mode_label,
                "agree_count": int(max_count),
                "confidence_score": float(max_count) / n_sources,
            })
            n_pseudo += 1

    print(f"\nFull consensus ({n_sources}/{n_sources}): {n_full_consensus} ({n_full_consensus / len(arr) * 100:.1f}%)")
    print(f"Partial consensus ({n_sources - 1}/{n_sources} or {n_sources - 2}/{n_sources}): {n_partial}")
    print(f"No consensus (all different): {n_disagree}")
    print(f"\nPseudo-labels emitted (min_agree={min_agree}): {n_pseudo}")

    out_df = pd.DataFrame(pseudo_labels)
    if len(out_df) > 0:
        label_counts = out_df["Label"].value_counts().sort_index().to_dict()
        print(f"Pseudo-label class counts: {label_counts}")
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
