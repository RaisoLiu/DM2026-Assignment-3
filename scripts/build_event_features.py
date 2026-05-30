#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import load_window_csv, normalize_id
from dm2026_asg3.features import SIGNAL_COLUMNS


META_COLS = {"file_id", "user_id", "split", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append event/transition features to a feature cache.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_cache)
    rows = []
    paths = sorted((args.data_dir / args.split).rglob("*.csv"))
    for idx, path in enumerate(paths, start=1):
        df = load_window_csv(path)
        row = extract_event_features(df)
        row["file_id"] = infer_file_id(df, path)
        row["user_id"] = path.parent.name
        rows.append(row)
        if idx % 1000 == 0:
            print(f"{args.split}: processed {idx}/{len(paths)}", flush=True)
    extra = pd.DataFrame(rows)
    merged = base.merge(extra, on=["file_id", "user_id"], how="left", validate="one_to_one")
    feature_cols = [c for c in merged.columns if c not in META_COLS]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    merged[feature_cols] = merged[feature_cols].fillna(merged[feature_cols].median(numeric_only=True)).fillna(0.0)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with shape {merged.shape}", flush=True)


def extract_event_features(df: pd.DataFrame) -> dict[str, float]:
    signals = {col: clean(df[col].to_numpy(dtype=float)) for col in SIGNAL_COLUMNS}
    mean_matrix = np.vstack([signals["mean_x"], signals["mean_y"], signals["mean_z"]]).T
    std_matrix = np.vstack([signals["std_x"], signals["std_y"], signals["std_z"]]).T
    mean_vm = np.linalg.norm(mean_matrix, axis=1)
    std_vm = np.linalg.norm(std_matrix, axis=1)
    orientation = np.zeros_like(mean_matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(mean_matrix, mean_vm[:, None], out=orientation, where=mean_vm[:, None] > 1e-12)
    orientation = np.nan_to_num(orientation, nan=0.0, posinf=0.0, neginf=0.0)

    pitch = np.arctan2(mean_matrix[:, 0], np.sqrt(mean_matrix[:, 1] ** 2 + mean_matrix[:, 2] ** 2))
    roll = np.arctan2(mean_matrix[:, 1], np.sqrt(mean_matrix[:, 0] ** 2 + mean_matrix[:, 2] ** 2))
    xy_angle = np.arctan2(mean_matrix[:, 1], mean_matrix[:, 0])
    mean_step = vector_step_magnitude(mean_matrix)
    std_step = vector_step_magnitude(std_matrix)
    orient_step = vector_angle_step(orientation)

    out: dict[str, float] = {}
    series = {
        "ev_mean_step": mean_step,
        "ev_std_step": std_step,
        "ev_orient_step": orient_step,
        "ev_mean_vm": mean_vm,
        "ev_std_vm": std_vm,
        "ev_pitch": pitch,
        "ev_roll": roll,
        "ev_xy_angle": unwrap_angle(xy_angle),
    }
    for axis, idx in zip(("x", "y", "z"), range(3), strict=True):
        series[f"ev_orient_{axis}"] = orientation[:, idx]
    for name, arr in series.items():
        out.update(event_summary(name, arr))

    out.update(vector_endpoint_features("ev_mean_vec", mean_matrix))
    out.update(vector_endpoint_features("ev_std_vec", std_matrix))
    out.update(vector_endpoint_features("ev_orient_vec", orientation))
    out.update(window_contrast_features("ev_mean_vm", mean_vm))
    out.update(window_contrast_features("ev_std_vm", std_vm))
    out.update(window_contrast_features("ev_orient_step", orient_step))
    out.update(window_contrast_features("ev_mean_step", mean_step))
    out.update(coupling_features(mean_vm, std_vm, mean_step, orient_step))
    return out


def infer_file_id(df: pd.DataFrame, path: Path) -> int:
    if "file_id" in df.columns:
        values = df["file_id"].dropna().unique()
        if len(values) == 1:
            return int(normalize_id(values[0]))
    return int(normalize_id(path.stem))


def clean(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def unwrap_angle(values: np.ndarray) -> np.ndarray:
    return np.unwrap(clean(values))


def vector_step_magnitude(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.zeros(1, dtype=float)
    return np.linalg.norm(np.diff(matrix, axis=0), axis=1)


def vector_angle_step(unit_vectors: np.ndarray) -> np.ndarray:
    if len(unit_vectors) < 2:
        return np.zeros(1, dtype=float)
    dots = np.sum(unit_vectors[1:] * unit_vectors[:-1], axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def event_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = clean(values)
    if arr.size == 0:
        arr = np.zeros(1, dtype=float)
    diff = np.diff(arr) if len(arr) > 1 else np.zeros(1, dtype=float)
    abs_diff = np.abs(diff)
    centered = arr - arr.mean()
    x = np.linspace(-1.0, 1.0, len(arr))
    slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) > 1 else 0.0
    peaks = peak_features(arr)
    out = {
        f"{prefix}_trend_slope": slope,
        f"{prefix}_endpoint_delta": endpoint_delta(arr, 20),
        f"{prefix}_endpoint_delta60": endpoint_delta(arr, 60),
        f"{prefix}_max_abs_centered": float(np.max(np.abs(centered))),
        f"{prefix}_q95_abs_centered": float(np.quantile(np.abs(centered), 0.95)),
        f"{prefix}_top3_abs_centered_mean": topk_mean(np.abs(centered), 3),
        f"{prefix}_top10_abs_centered_mean": topk_mean(np.abs(centered), 10),
        f"{prefix}_diff_q90": float(np.quantile(abs_diff, 0.90)),
        f"{prefix}_diff_q95": float(np.quantile(abs_diff, 0.95)),
        f"{prefix}_diff_top3_mean": topk_mean(abs_diff, 3),
        f"{prefix}_diff_top10_mean": topk_mean(abs_diff, 10),
        f"{prefix}_diff_burst_ratio": float(topk_mean(abs_diff, 10) / (np.mean(abs_diff) + 1e-12)),
        f"{prefix}_autocorr_lag3": autocorr(arr, 3),
        f"{prefix}_autocorr_lag15": autocorr(arr, 15),
        f"{prefix}_autocorr_lag30": autocorr(arr, 30),
        f"{prefix}_longest_high_q90": longest_fraction(arr >= np.quantile(arr, 0.90)),
        f"{prefix}_longest_low_q10": longest_fraction(arr <= np.quantile(arr, 0.10)),
    }
    out.update({f"{prefix}_{key}": value for key, value in peaks.items()})
    return out


def peak_features(values: np.ndarray) -> dict[str, float]:
    arr = clean(values)
    centered = arr - np.median(arr)
    scale = np.median(np.abs(centered)) + 1e-8
    peaks, props = find_peaks(centered, prominence=scale)
    neg_peaks, neg_props = find_peaks(-centered, prominence=scale)
    pos_prom = props.get("prominences", np.zeros(0, dtype=float))
    neg_prom = neg_props.get("prominences", np.zeros(0, dtype=float))
    return {
        "pos_peak_count": float(len(peaks)),
        "neg_peak_count": float(len(neg_peaks)),
        "pos_peak_prom_max": float(pos_prom.max()) if len(pos_prom) else 0.0,
        "neg_peak_prom_max": float(neg_prom.max()) if len(neg_prom) else 0.0,
        "pos_peak_prom_mean": float(pos_prom.mean()) if len(pos_prom) else 0.0,
        "neg_peak_prom_mean": float(neg_prom.mean()) if len(neg_prom) else 0.0,
    }


def endpoint_delta(values: np.ndarray, width: int) -> float:
    arr = clean(values)
    width = min(width, max(1, len(arr) // 2))
    return float(arr[-width:].mean() - arr[:width].mean())


def topk_mean(values: np.ndarray, k: int) -> float:
    arr = clean(values)
    k = min(k, len(arr))
    if k <= 0:
        return 0.0
    return float(np.partition(arr, -k)[-k:].mean())


def autocorr(values: np.ndarray, lag: int) -> float:
    arr = clean(values)
    if len(arr) <= lag:
        return 0.0
    left = arr[:-lag] - arr[:-lag].mean()
    right = arr[lag:] - arr[lag:].mean()
    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(left * right) / denom)


def longest_fraction(mask: np.ndarray) -> float:
    if len(mask) == 0:
        return 0.0
    best = cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / len(mask))


def vector_endpoint_features(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for width in (10, 30, 60):
        width = min(width, max(1, len(matrix) // 2))
        delta = matrix[-width:].mean(axis=0) - matrix[:width].mean(axis=0)
        out[f"{prefix}_endpoint{width}_norm"] = float(np.linalg.norm(delta))
        for axis, value in zip(("x", "y", "z"), delta, strict=True):
            out[f"{prefix}_endpoint{width}_{axis}"] = float(value)
    return out


def window_contrast_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = clean(values)
    out: dict[str, float] = {}
    for n_segments in (4, 6, 10):
        means = np.array([seg.mean() if len(seg) else 0.0 for seg in np.array_split(arr, n_segments)])
        out[f"{prefix}_seg{n_segments}_max_minus_min"] = float(means.max() - means.min())
        out[f"{prefix}_seg{n_segments}_argmax"] = float(np.argmax(means) / max(1, n_segments - 1))
        out[f"{prefix}_seg{n_segments}_argmin"] = float(np.argmin(means) / max(1, n_segments - 1))
        out[f"{prefix}_seg{n_segments}_last_minus_first"] = float(means[-1] - means[0])
    return out


def coupling_features(mean_vm: np.ndarray, std_vm: np.ndarray, mean_step: np.ndarray, orient_step: np.ndarray) -> dict[str, float]:
    n = min(len(mean_step), len(orient_step), len(std_vm) - 1)
    if n <= 1:
        return {
            "ev_std_step_corr": 0.0,
            "ev_meanstep_orientstep_corr": 0.0,
            "ev_std_high_orient_high_fraction": 0.0,
        }
    std_tail = clean(std_vm[1 : n + 1])
    mean_step = clean(mean_step[:n])
    orient_step = clean(orient_step[:n])
    return {
        "ev_std_step_corr": corr(std_tail, mean_step),
        "ev_meanstep_orientstep_corr": corr(mean_step, orient_step),
        "ev_std_high_orient_high_fraction": float(
            np.mean((std_tail >= np.quantile(std_tail, 0.80)) & (orient_step >= np.quantile(orient_step, 0.80)))
        ),
        "ev_meanvm_stdvm_corr": corr(clean(mean_vm), clean(std_vm)),
    }


def corr(left: np.ndarray, right: np.ndarray) -> float:
    n = min(len(left), len(right))
    if n <= 1:
        return 0.0
    left = clean(left[:n])
    right = clean(right[:n])
    left = left - left.mean()
    right = right - right.mean()
    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(left * right) / denom)


if __name__ == "__main__":
    main()
