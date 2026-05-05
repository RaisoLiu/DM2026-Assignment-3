#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import discover_records, load_window_csv
from dm2026_asg3.features import SIGNAL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append raw sequence landmark features to an existing feature cache.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=30)
    parser.add_argument("--n-segments", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_cache)
    records = discover_records(args.data_dir, args.split)
    rows = []
    for idx, record in enumerate(records, start=1):
        df = load_window_csv(record.path)
        row = extract_landmark_features(df, n_points=args.n_points, n_segments=args.n_segments)
        row["file_id"] = int(record.file_id)
        row["user_id"] = record.user_id
        rows.append(row)
        if idx % 1000 == 0:
            print(f"{args.split}: processed {idx}/{len(records)}", flush=True)
    extra = pd.DataFrame(rows)
    merged = base.merge(extra, on=["file_id", "user_id"], how="left", validate="one_to_one")
    feature_cols = [c for c in merged.columns if c not in {"file_id", "user_id", "split", "label"}]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    merged[feature_cols] = merged[feature_cols].fillna(merged[feature_cols].median(numeric_only=True)).fillna(0.0)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with shape {merged.shape}", flush=True)


def extract_landmark_features(df: pd.DataFrame, n_points: int, n_segments: int) -> dict[str, float]:
    series = make_series(df)
    out: dict[str, float] = {}
    for name, arr in series.items():
        arr = clean(arr)
        centered = arr - np.nanmean(arr)
        scaled = centered / (np.nanstd(arr) + 1e-8)
        for prefix, values in ((name, arr), (f"{name}_zshape", scaled)):
            sampled = resample(values, n_points)
            for idx, value in enumerate(sampled):
                out[f"lm_{prefix}_{idx:02d}"] = float(value)
        gradient = np.gradient(arr)
        for idx, value in enumerate(resample(gradient, max(8, n_points // 2))):
            out[f"lm_{name}_grad_{idx:02d}"] = float(value)
        for seg_idx, segment in enumerate(np.array_split(arr, n_segments), start=1):
            if len(segment) == 0:
                out[f"seg10_{name}_{seg_idx:02d}_mean"] = 0.0
                out[f"seg10_{name}_{seg_idx:02d}_std"] = 0.0
                out[f"seg10_{name}_{seg_idx:02d}_slope"] = 0.0
                continue
            x = np.linspace(-1.0, 1.0, len(segment))
            y = clean(segment)
            out[f"seg10_{name}_{seg_idx:02d}_mean"] = float(np.mean(y))
            out[f"seg10_{name}_{seg_idx:02d}_std"] = float(np.std(y))
            out[f"seg10_{name}_{seg_idx:02d}_slope"] = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0
    return out


def make_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    signals = {col: clean(df[col].to_numpy(dtype=float)) for col in SIGNAL_COLUMNS}
    mean_matrix = np.vstack([signals["mean_x"], signals["mean_y"], signals["mean_z"]]).T
    std_matrix = np.vstack([signals["std_x"], signals["std_y"], signals["std_z"]]).T
    mean_vm = np.sqrt(np.sum(mean_matrix**2, axis=1))
    std_vm = np.sqrt(np.sum(std_matrix**2, axis=1))
    orientation = np.zeros_like(mean_matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(mean_matrix, mean_vm[:, None], out=orientation, where=mean_vm[:, None] > 1e-12)
    out = dict(signals)
    out.update(
        {
            "mean_vm": mean_vm,
            "std_vm": std_vm,
            "gravity_abs_delta": np.abs(mean_vm - 1.0),
            "orient_x": orientation[:, 0],
            "orient_y": orientation[:, 1],
            "orient_z": orientation[:, 2],
        }
    )
    return out


def clean(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def resample(values: np.ndarray, n_points: int) -> np.ndarray:
    values = clean(values)
    if len(values) == 1:
        return np.full(n_points, float(values[0]))
    src = np.linspace(0.0, 1.0, len(values))
    dst = np.linspace(0.0, 1.0, n_points)
    return np.interp(dst, src, values)


if __name__ == "__main__":
    main()
