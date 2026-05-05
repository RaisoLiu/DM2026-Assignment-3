from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = ("mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z")
MEAN_COLUMNS = ("mean_x", "mean_y", "mean_z")
STD_COLUMNS = ("std_x", "std_y", "std_z")


def _safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(1, dtype=float)
    s = pd.Series(arr)
    s = s.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=float)


def _summary(prefix: str, arr: np.ndarray) -> dict[str, float]:
    arr = _safe_array(arr)
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    centered = arr - arr.mean()
    std = float(arr.std(ddof=0))
    if std > 1e-12:
        skew = float(np.mean((centered / std) ** 3))
        kurt = float(np.mean((centered / std) ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": std,
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_median": float(q50),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(arr.max() - arr.min()),
        f"{prefix}_rms": float(np.sqrt(np.mean(arr**2))),
        f"{prefix}_skew": skew,
        f"{prefix}_kurtosis": kurt,
    }


def _diff_features(prefix: str, arr: np.ndarray) -> dict[str, float]:
    arr = _safe_array(arr)
    if len(arr) < 2:
        diff = np.zeros(1, dtype=float)
    else:
        diff = np.diff(arr)
    signs = np.sign(diff)
    zero_crossings = np.sum(signs[1:] * signs[:-1] < 0) if len(signs) > 1 else 0
    return {
        f"{prefix}_diff_mean": float(diff.mean()),
        f"{prefix}_diff_std": float(diff.std(ddof=0)),
        f"{prefix}_diff_abs_mean": float(np.abs(diff).mean()),
        f"{prefix}_diff_abs_max": float(np.abs(diff).max()),
        f"{prefix}_diff_energy": float(np.mean(diff**2)),
        f"{prefix}_diff_zero_crossings": float(zero_crossings),
    }


def _fft_features(prefix: str, arr: np.ndarray) -> dict[str, float]:
    arr = _safe_array(arr)
    if len(arr) < 4:
        return {
            f"{prefix}_fft_total_power": 0.0,
            f"{prefix}_fft_dom_freq": 0.0,
            f"{prefix}_fft_entropy": 0.0,
            f"{prefix}_fft_centroid": 0.0,
            f"{prefix}_fft_band_000_005": 0.0,
            f"{prefix}_fft_band_005_015": 0.0,
            f"{prefix}_fft_band_015_030": 0.0,
            f"{prefix}_fft_band_030_050": 0.0,
        }
    arr = arr - arr.mean()
    spectrum = np.fft.rfft(arr)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(arr), d=1.0)
    power = power[1:]
    freqs = freqs[1:]
    total = float(power.sum())
    if total <= 1e-12:
        norm = np.full_like(power, 1.0 / len(power))
    else:
        norm = power / total
    dom_idx = int(np.argmax(power)) if len(power) else 0
    entropy = -float(np.sum(norm * np.log(norm + 1e-12)) / np.log(len(norm) + 1e-12))
    out = {
        f"{prefix}_fft_total_power": total,
        f"{prefix}_fft_dom_freq": float(freqs[dom_idx]) if len(freqs) else 0.0,
        f"{prefix}_fft_entropy": entropy,
        f"{prefix}_fft_centroid": float(np.sum(freqs * norm)) if len(freqs) else 0.0,
    }
    bands = ((0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50))
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        key = f"{prefix}_fft_band_{int(lo * 100):03d}_{int(hi * 100):03d}"
        out[key] = float(power[mask].sum() / (total + 1e-12))
    return out


def _segment_features(prefix: str, arr: np.ndarray, n_segments: int = 5) -> dict[str, float]:
    arr = _safe_array(arr)
    out: dict[str, float] = {}
    for idx, segment in enumerate(np.array_split(arr, n_segments)):
        seg_prefix = f"{prefix}_seg{idx + 1}"
        out[f"{seg_prefix}_mean"] = float(segment.mean()) if len(segment) else 0.0
        out[f"{seg_prefix}_std"] = float(segment.std(ddof=0)) if len(segment) else 0.0
        out[f"{seg_prefix}_min"] = float(segment.min()) if len(segment) else 0.0
        out[f"{seg_prefix}_max"] = float(segment.max()) if len(segment) else 0.0
    return out


def _rolling_features(prefix: str, arr: np.ndarray, windows: tuple[int, ...] = (5, 15, 30)) -> dict[str, float]:
    arr = _safe_array(arr)
    out: dict[str, float] = {}
    series = pd.Series(arr)
    for window in windows:
        rolled_mean = series.rolling(window, min_periods=max(2, window // 2)).mean().dropna()
        rolled_std = series.rolling(window, min_periods=max(2, window // 2)).std(ddof=0).dropna()
        if rolled_mean.empty:
            out[f"{prefix}_roll{window}_mean_std"] = 0.0
            out[f"{prefix}_roll{window}_std_mean"] = 0.0
        else:
            out[f"{prefix}_roll{window}_mean_std"] = float(rolled_mean.std(ddof=0))
            out[f"{prefix}_roll{window}_std_mean"] = float(rolled_std.fillna(0.0).mean())
    return out


def extract_window_features(df: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}
    missing_count = int(df[list(SIGNAL_COLUMNS)].isna().sum().sum())
    features["n_rows"] = float(len(df))
    features["missing_signal_count"] = float(missing_count)
    if "index" in df.columns:
        idx = _safe_array(df["index"])
        features["index_min"] = float(idx.min())
        features["index_max"] = float(idx.max())
        features["index_gap_count"] = float(max(0, int(idx.max() - idx.min() + 1 - len(np.unique(idx)))))
    means = {col: _safe_array(df[col]) for col in MEAN_COLUMNS}
    stds = {col: _safe_array(df[col]) for col in STD_COLUMNS}
    mean_matrix = np.vstack([means[c] for c in MEAN_COLUMNS]).T
    std_matrix = np.vstack([stds[c] for c in STD_COLUMNS]).T
    mean_vm = np.sqrt(np.sum(mean_matrix**2, axis=1))
    std_vm = np.sqrt(np.sum(std_matrix**2, axis=1))
    features.update(_summary("mean_vm", mean_vm))
    features.update(_summary("std_vm", std_vm))
    features.update(_summary("gravity_abs_delta", np.abs(mean_vm - 1.0)))
    for col in SIGNAL_COLUMNS:
        arr = _safe_array(df[col])
        features.update(_summary(col, arr))
        features.update(_diff_features(col, arr))
    for prefix, arr in (
        ("mean_vm", mean_vm),
        ("std_vm", std_vm),
        ("mean_x", means["mean_x"]),
        ("mean_y", means["mean_y"]),
        ("mean_z", means["mean_z"]),
    ):
        features.update(_diff_features(prefix, arr))
        features.update(_fft_features(prefix, arr))
        features.update(_segment_features(prefix, arr))
        features.update(_rolling_features(prefix, arr))
    orientation = np.zeros_like(mean_matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(mean_matrix, mean_vm[:, None], out=orientation, where=mean_vm[:, None] > 1e-12)
    orientation = np.nan_to_num(orientation, nan=0.0, posinf=0.0, neginf=0.0)
    for axis, idx in zip(("x", "y", "z"), range(3), strict=True):
        features.update(_summary(f"orient_{axis}", orientation[:, idx]))
    corr = np.corrcoef(mean_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    for i, a in enumerate(("x", "y", "z")):
        for j, b in enumerate(("x", "y", "z")):
            if i < j:
                features[f"corr_mean_{a}{b}"] = float(corr[i, j])
    cov = np.cov(mean_matrix.T)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    for i, a in enumerate(("x", "y", "z")):
        features[f"cov_mean_{a}{a}"] = float(cov[i, i])
    features["mean_vm_to_std_vm_ratio"] = float(mean_vm.mean() / (std_vm.mean() + 1e-12))
    features["dynamic_to_static_energy"] = float(np.mean(std_matrix**2) / (np.mean(mean_matrix**2) + 1e-12))
    return features


def build_feature_frame(records, load_window_csv) -> pd.DataFrame:
    rows = []
    for record in records:
        df = load_window_csv(record.path)
        row = extract_window_features(df)
        row["file_id"] = record.file_id
        row["user_id"] = record.user_id
        row["split"] = record.split
        if record.label is not None:
            row["label"] = int(record.label)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in frame.columns if c not in {"file_id", "user_id", "split", "label"}]
    frame[feature_cols] = frame[feature_cols].fillna(frame[feature_cols].median(numeric_only=True)).fillna(0.0)
    return frame
