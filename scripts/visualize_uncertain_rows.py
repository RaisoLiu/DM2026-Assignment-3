"""Visualize the most-uncertain test rows from the HAR Kaggle pipeline.

Definition of "uncertain": rows where <=3 of 5 anchors agree on the label.

For each such row we plot the 6 sensor channels over the 300 timesteps and dump
the figure into ``artifacts/uncertain_viz/by_h8_label/<label>/<file_id>.png``.

Also writes ``artifacts/uncertain_viz/summary.csv`` with per-row metadata.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/raiso/playground/DM2026-Assignment-3-claude")
TEST_ROOT = ROOT / "data" / "raw" / "test"
OUT_ROOT = ROOT / "artifacts" / "uncertain_viz"
OUT_BY_LABEL = OUT_ROOT / "by_h8_label"
SUMMARY_CSV = OUT_ROOT / "summary.csv"

ANCHORS = [
    ("ssl_hybrid", ROOT / "submissions" / "submission_ssl_hybrid_recover.csv"),
    ("inception_blend", ROOT / "submissions" / "submission_inception_ssl_blend_viterbi.csv"),
    ("consensus_greedy", ROOT / "submissions" / "submission_consensus_oof_greedy.csv"),
    ("centered_meta", ROOT / "submissions" / "submission_centered_meta_viterbi_oof07693.csv"),
    ("lgbm63", ROOT / "submissions" / "submission_lgbm_leaves63_calibrated.csv"),
]
H8_CSV = ROOT / "submissions" / "submission_h8_v3_sslv2_weighted.csv"

CHANNELS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def load_anchor(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["Id"] = df["Id"].astype(int)
    df["Label"] = df["Label"].astype(int)
    return df.set_index("Id")["Label"]


def build_file_id_to_user() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for user_dir in sorted(TEST_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue
        for csv in user_dir.glob("*.csv"):
            try:
                fid = int(csv.stem)
            except ValueError:
                continue
            mapping[fid] = user_dir.name
    return mapping


def plot_row(file_id: int, user: str, df: pd.DataFrame, votes: dict[str, int],
             h8_label: int, agree_count: int, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, ch in zip(axes.flat, CHANNELS):
        ax.plot(df["index"].values, df[ch].values, lw=0.8)
        ax.set_title(ch)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 299)
    axes[-1, 0].set_xlabel("second")
    axes[-1, 1].set_xlabel("second")
    vote_str = " ".join(f"{k}={v}" for k, v in votes.items())
    fig.suptitle(
        f"file_id={file_id}  user={user}  H8={h8_label}  agree={agree_count}/5\n{vote_str}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_BY_LABEL.mkdir(parents=True, exist_ok=True)

    # Load anchors + H8.
    anchor_series = {name: load_anchor(path) for name, path in ANCHORS}
    h8_series = load_anchor(H8_CSV)

    # Common index from H8 (master test set).
    ids = sorted(h8_series.index.tolist())
    print(f"[info] total test rows: {len(ids)}")

    # Build matrix of anchor labels.
    anchor_matrix = pd.DataFrame({name: s for name, s in anchor_series.items()}).reindex(ids)
    anchor_matrix["h8"] = h8_series.reindex(ids)
    missing = anchor_matrix.isna().sum()
    print(f"[info] missing-per-anchor:\n{missing}")

    # Per-row max agreement among the 5 anchors.
    def max_agree(row: pd.Series) -> int:
        counts = Counter(int(v) for v in row if pd.notna(v))
        return max(counts.values()) if counts else 0

    def max_vote_class(row: pd.Series) -> int:
        counts = Counter(int(v) for v in row if pd.notna(v))
        if not counts:
            return -1
        # Tie-break by smallest class label for determinism.
        max_c = max(counts.values())
        return min(c for c, k in counts.items() if k == max_c)

    anchor_only = anchor_matrix[[name for name, _ in ANCHORS]]
    anchor_matrix["agree_count"] = anchor_only.apply(max_agree, axis=1)
    anchor_matrix["max_vote_class"] = anchor_only.apply(max_vote_class, axis=1)

    uncertain = anchor_matrix[anchor_matrix["agree_count"] <= 3].copy()
    print(f"[info] uncertain rows (agree <= 3/5): {len(uncertain)}")

    user_map = build_file_id_to_user()
    print(f"[info] file_id -> user map size: {len(user_map)}")

    # Build summary rows + plot per-row.
    summary_rows = []
    label_counts: Counter[int] = Counter()
    for fid, row in uncertain.iterrows():
        user = user_map.get(int(fid), "UNKNOWN")
        h8 = int(row["h8"])
        votes = {name: int(row[name]) for name, _ in ANCHORS}
        agree = int(row["agree_count"])
        max_vote = int(row["max_vote_class"])
        summary_rows.append(
            {
                "file_id": int(fid),
                "user": user,
                "H8_label": h8,
                **{f"anchor_{name}": votes[name] for name, _ in ANCHORS},
                "agree_count": agree,
                "max_vote_class": max_vote,
            }
        )
        # Locate the raw CSV.
        csv_path = TEST_ROOT / user / f"{int(fid)}.csv"
        if not csv_path.exists():
            print(f"[warn] missing test csv for file_id={fid} user={user}")
            continue
        df = pd.read_csv(csv_path)
        out_dir = OUT_BY_LABEL / f"class_{h8}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(fid)}.png"
        plot_row(int(fid), user, df, votes, h8, agree, out_path)
        label_counts[h8] += 1

    summary_df = pd.DataFrame(summary_rows).sort_values(["H8_label", "file_id"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[info] wrote summary: {SUMMARY_CSV} ({len(summary_df)} rows)")

    print("[info] files per H8 class folder:")
    for lab in sorted(label_counts):
        print(f"  class_{lab}: {label_counts[lab]}")

    print("[info] summary head:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
