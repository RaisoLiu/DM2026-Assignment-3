#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import discover_records, load_window_csv
from dm2026_asg3.features import SIGNAL_COLUMNS


META_COLS = {"file_id", "user_id", "split", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append spectral and wavelet sequence features to a feature cache.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_cache)
    records = discover_records(args.data_dir, args.split)
    rows = []
    for idx, record in enumerate(records, start=1):
        df = load_window_csv(record.path)
        row = extract_spectral_wavelet_features(df)
        row["file_id"] = int(record.file_id)
        row["user_id"] = record.user_id
        rows.append(row)
        if idx % 1000 == 0:
            print(f"{args.split}: processed {idx}/{len(records)}", flush=True)
    extra = pd.DataFrame(rows)
    merged = base.merge(extra, on=["file_id", "user_id"], how="left", validate="one_to_one")
    feature_cols = [c for c in merged.columns if c not in META_COLS]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    merged[feature_cols] = merged[feature_cols].fillna(merged[feature_cols].median(numeric_only=True)).fillna(0.0)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with shape {merged.shape}", flush=True)


def extract_spectral_wavelet_features(df: pd.DataFrame) -> dict[str, float]:
    signals = {col: clean(df[col].to_numpy(dtype=float)) for col in SIGNAL_COLUMNS}
    mean_matrix = np.vstack([signals["mean_x"], signals["mean_y"], signals["mean_z"]]).T
    std_matrix = np.vstack([signals["std_x"], signals["std_y"], signals["std_z"]]).T
    mean_vm = np.linalg.norm(mean_matrix, axis=1)
    std_vm = np.linalg.norm(std_matrix, axis=1)
    orientation = np.zeros_like(mean_matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(mean_matrix, mean_vm[:, None], out=orientation, where=mean_vm[:, None] > 1e-12)
    orientation = np.nan_to_num(orientation, nan=0.0, posinf=0.0, neginf=0.0)
    mean_step = step_norm(mean_matrix)
    std_step = step_norm(std_matrix)
    orient_step = angle_step(orientation)

    series: dict[str, np.ndarray] = {
        "sw_mean_vm": mean_vm,
        "sw_std_vm": std_vm,
        "sw_gravity_delta": np.abs(mean_vm - 1.0),
        "sw_mean_step": mean_step,
        "sw_std_step": std_step,
        "sw_orient_step": orient_step,
    }
    for col in SIGNAL_COLUMNS:
        series[f"sw_{col}"] = signals[col]
    for axis, idx in zip(("x", "y", "z"), range(3), strict=True):
        series[f"sw_orient_{axis}"] = orientation[:, idx]

    out: dict[str, float] = {}
    compact_names = {
        "sw_mean_vm",
        "sw_std_vm",
        "sw_gravity_delta",
        "sw_mean_step",
        "sw_std_step",
        "sw_orient_step",
        "sw_mean_x",
        "sw_mean_y",
        "sw_mean_z",
        "sw_std_x",
        "sw_std_y",
        "sw_std_z",
        "sw_orient_x",
        "sw_orient_y",
        "sw_orient_z",
    }
    for name in sorted(compact_names):
        arr = series[name]
        out.update(spectral_features(name, arr))
        out.update(wavelet_features(name, arr))
        out.update(multiscale_residual_features(name, arr))
    out.update(lag_correlation_features("sw_mean", signals["mean_x"], signals["mean_y"], signals["mean_z"]))
    out.update(lag_correlation_features("sw_std", signals["std_x"], signals["std_y"], signals["std_z"]))
    out.update(lag_correlation_features("sw_orient", orientation[:, 0], orientation[:, 1], orientation[:, 2]))
    return out


def clean(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def step_norm(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.zeros(1, dtype=float)
    return np.linalg.norm(np.diff(matrix, axis=0), axis=1)


def angle_step(unit_vectors: np.ndarray) -> np.ndarray:
    if len(unit_vectors) < 2:
        return np.zeros(1, dtype=float)
    dots = np.sum(unit_vectors[1:] * unit_vectors[:-1], axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def spectral_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = clean(values)
    if len(arr) < 8:
        return {f"{prefix}_welch_empty": 0.0}
    centered = arr - arr.mean()
    freqs, power = welch(centered, fs=1.0, nperseg=min(128, len(arr)), noverlap=min(64, max(0, len(arr) // 4)))
    power = np.maximum(power, 0.0)
    if len(power) > 1:
        freqs = freqs[1:]
        power = power[1:]
    total = float(power.sum())
    norm = power / (total + 1e-12)
    cdf = np.cumsum(norm)
    dom_idx = int(np.argmax(power)) if len(power) else 0
    entropy = -float(np.sum(norm * np.log(norm + 1e-12)) / np.log(len(norm) + 1e-12)) if len(norm) > 1 else 0.0
    centroid = float(np.sum(freqs * norm)) if len(freqs) else 0.0
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12)) if len(power) else 0.0
    out = {
        f"{prefix}_welch_total": total,
        f"{prefix}_welch_dom_freq": float(freqs[dom_idx]) if len(freqs) else 0.0,
        f"{prefix}_welch_centroid": centroid,
        f"{prefix}_welch_entropy": entropy,
        f"{prefix}_welch_flatness": flatness,
        f"{prefix}_welch_roll50": rolloff(freqs, cdf, 0.50),
        f"{prefix}_welch_roll85": rolloff(freqs, cdf, 0.85),
        f"{prefix}_welch_roll95": rolloff(freqs, cdf, 0.95),
    }
    bands = ((0.00, 0.025), (0.025, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50))
    band_values = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        value = float(power[mask].sum() / (total + 1e-12))
        out[f"{prefix}_welch_band_{int(lo * 1000):03d}_{int(hi * 1000):03d}"] = value
        band_values.append(value)
    out[f"{prefix}_welch_low_high_ratio"] = float((band_values[0] + band_values[1]) / (band_values[-1] + band_values[-2] + 1e-12))
    out[f"{prefix}_welch_mid_high_ratio"] = float((band_values[2] + band_values[3]) / (band_values[-1] + 1e-12))
    return out


def rolloff(freqs: np.ndarray, cdf: np.ndarray, threshold: float) -> float:
    if len(freqs) == 0:
        return 0.0
    idx = int(np.searchsorted(cdf, threshold, side="left"))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def wavelet_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = clean(values)
    arr = arr - arr.mean()
    out: dict[str, float] = {}
    for scale in (2, 4, 8, 16, 32):
        kernel = ricker_kernel(scale)
        conv = np.convolve(arr, kernel, mode="same")
        abs_conv = np.abs(conv)
        out[f"{prefix}_wav_s{scale}_std"] = float(conv.std(ddof=0))
        out[f"{prefix}_wav_s{scale}_energy"] = float(np.mean(conv**2))
        out[f"{prefix}_wav_s{scale}_abs_mean"] = float(abs_conv.mean())
        out[f"{prefix}_wav_s{scale}_abs_q95"] = float(np.quantile(abs_conv, 0.95))
        out[f"{prefix}_wav_s{scale}_abs_max"] = float(abs_conv.max())
        out[f"{prefix}_wav_s{scale}_abs_top5"] = topk_mean(abs_conv, 5)
        out[f"{prefix}_wav_s{scale}_argmax"] = float(np.argmax(abs_conv) / max(1, len(abs_conv) - 1))
        seg_means = np.array([np.mean(seg) if len(seg) else 0.0 for seg in np.array_split(abs_conv, 5)])
        out[f"{prefix}_wav_s{scale}_seg_max_minus_min"] = float(seg_means.max() - seg_means.min())
        out[f"{prefix}_wav_s{scale}_seg_argmax"] = float(np.argmax(seg_means) / 4.0)
    return out


def ricker_kernel(scale: int) -> np.ndarray:
    points = int(max(9, scale * 10))
    if points % 2 == 0:
        points += 1
    x = np.arange(points, dtype=float) - (points - 1) / 2.0
    a2 = float(scale * scale)
    kernel = (1.0 - (x**2 / a2)) * np.exp(-(x**2) / (2.0 * a2))
    kernel -= kernel.mean()
    norm = np.sqrt(np.sum(kernel**2))
    return kernel / (norm + 1e-12)


def multiscale_residual_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = clean(values)
    out: dict[str, float] = {}
    for width in (5, 11, 21, 51):
        smooth = moving_average(arr, width)
        residual = arr - smooth
        abs_res = np.abs(residual)
        out[f"{prefix}_resid_w{width}_std"] = float(residual.std(ddof=0))
        out[f"{prefix}_resid_w{width}_abs_mean"] = float(abs_res.mean())
        out[f"{prefix}_resid_w{width}_abs_q95"] = float(np.quantile(abs_res, 0.95))
        out[f"{prefix}_resid_w{width}_range"] = float(smooth.max() - smooth.min())
        out[f"{prefix}_resid_w{width}_endpoint"] = endpoint_delta(smooth, max(5, width // 2))
        out[f"{prefix}_resid_w{width}_roughness_ratio"] = float(abs_res.mean() / (np.std(smooth) + 1e-12))
    return out


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.full(width, 1.0 / width, dtype=float)
    pad = width // 2
    padded = np.pad(values, (pad, width - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def endpoint_delta(values: np.ndarray, width: int) -> float:
    width = min(width, max(1, len(values) // 2))
    return float(values[-width:].mean() - values[:width].mean())


def topk_mean(values: np.ndarray, k: int) -> float:
    k = min(k, len(values))
    if k <= 0:
        return 0.0
    return float(np.partition(values, -k)[-k:].mean())


def lag_correlation_features(prefix: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    arrays = {"x": clean(x), "y": clean(y), "z": clean(z)}
    out: dict[str, float] = {}
    for left, right in (("x", "y"), ("x", "z"), ("y", "z")):
        vals = []
        for lag in (-20, -10, -5, 0, 5, 10, 20):
            value = lag_corr(arrays[left], arrays[right], lag)
            out[f"{prefix}_{left}{right}_lag{lag:+03d}_corr"] = value
            vals.append(value)
        vals_arr = np.asarray(vals, dtype=float)
        out[f"{prefix}_{left}{right}_lagcorr_absmax"] = float(np.max(np.abs(vals_arr)))
        out[f"{prefix}_{left}{right}_lagcorr_argabsmax"] = float((-20, -10, -5, 0, 5, 10, 20)[int(np.argmax(np.abs(vals_arr)))])
    return out


def lag_corr(left: np.ndarray, right: np.ndarray, lag: int) -> float:
    if lag > 0:
        a = left[:-lag]
        b = right[lag:]
    elif lag < 0:
        a = left[-lag:]
        b = right[:lag]
    else:
        a = left
        b = right
    if len(a) < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


if __name__ == "__main__":
    main()
