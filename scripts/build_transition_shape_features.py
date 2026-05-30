#!/usr/bin/env python3
"""Hand-crafted transition-shape features per 300-row × 6-channel window.

Targets class-2 (transient bursts) by capturing within-window dynamics:
  - Boundary asymmetry: first-30 vs last-30 mean/std/range deltas
  - Mid-window analysis: split into 3 thirds, compare variance/mean
  - Peak detection: position and magnitude of max-abs in each channel
  - Autocorrelation at lags 1, 5, 10, 30
  - Within-channel monotonicity / trend
  - Cross-channel correlation changes (first half vs second half)
  - Frequency-domain features: dominant FFT bin, energy in low/mid/high bands

Output: feature DataFrame compatible with LGBM/XGB training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis


CHANNEL_NAMES = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract transition-shape features.")
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("artifacts/sequence/train_sequences.npz"),
        help="Input sequence cache (x, y, users, file_ids).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/features_transition/train_transition.csv"),
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        default=False,
        help="Include label column if present in input.",
    )
    return parser.parse_args()


def extract_features(x: np.ndarray) -> dict[str, float]:
    """Extract transition-shape features from a (6, 300) window."""
    feats = {}
    T = x.shape[1]
    boundary_n = 30  # first/last 30 rows
    third = T // 3

    for c, name in enumerate(CHANNEL_NAMES):
        ch = x[c]  # (300,)

        # Boundary asymmetry
        first = ch[:boundary_n]
        last = ch[-boundary_n:]
        feats[f"{name}_first_mean"] = float(first.mean())
        feats[f"{name}_last_mean"] = float(last.mean())
        feats[f"{name}_first_std"] = float(first.std())
        feats[f"{name}_last_std"] = float(last.std())
        feats[f"{name}_first_last_mean_delta"] = float(first.mean() - last.mean())
        feats[f"{name}_first_last_std_delta"] = float(first.std() - last.std())
        feats[f"{name}_first_last_range_delta"] = float((first.max() - first.min()) - (last.max() - last.min()))

        # Mid-window thirds
        t1 = ch[:third]
        t2 = ch[third:2 * third]
        t3 = ch[2 * third:]
        feats[f"{name}_t1_var"] = float(t1.var())
        feats[f"{name}_t2_var"] = float(t2.var())
        feats[f"{name}_t3_var"] = float(t3.var())
        feats[f"{name}_var_range"] = float(max(t1.var(), t2.var(), t3.var()) - min(t1.var(), t2.var(), t3.var()))

        # Peak detection
        abs_ch = np.abs(ch - ch.mean())
        peak_idx = int(np.argmax(abs_ch))
        feats[f"{name}_peak_pos_norm"] = float(peak_idx / T)
        feats[f"{name}_peak_magnitude"] = float(abs_ch[peak_idx])
        feats[f"{name}_max"] = float(ch.max())
        feats[f"{name}_min"] = float(ch.min())
        feats[f"{name}_range"] = float(ch.max() - ch.min())

        # Autocorrelation at lags
        ch_centered = ch - ch.mean()
        denom = (ch_centered ** 2).sum()
        for lag in (1, 5, 10, 30):
            if denom > 1e-9:
                ac = float((ch_centered[:-lag] * ch_centered[lag:]).sum() / denom)
            else:
                ac = 0.0
            feats[f"{name}_ac_lag{lag}"] = ac

        # Skew / kurtosis
        if ch.std() > 1e-9:
            feats[f"{name}_skew"] = float(skew(ch))
            feats[f"{name}_kurt"] = float(kurtosis(ch))
        else:
            feats[f"{name}_skew"] = 0.0
            feats[f"{name}_kurt"] = 0.0

        # Monotonic trend: count of sign changes in diff
        d = np.diff(ch)
        sign_changes = int(((d[:-1] * d[1:]) < 0).sum())
        feats[f"{name}_sign_changes"] = sign_changes
        feats[f"{name}_diff_mean"] = float(d.mean())
        feats[f"{name}_diff_std"] = float(d.std())
        feats[f"{name}_diff_max_abs"] = float(np.abs(d).max())

        # FFT energy bands
        fft = np.fft.rfft(ch - ch.mean())
        power = np.abs(fft) ** 2
        total = power.sum() + 1e-9
        n_freq = len(power)
        low = power[:n_freq // 4].sum() / total
        mid = power[n_freq // 4 : 3 * n_freq // 4].sum() / total
        high = power[3 * n_freq // 4 :].sum() / total
        feats[f"{name}_fft_low"] = float(low)
        feats[f"{name}_fft_mid"] = float(mid)
        feats[f"{name}_fft_high"] = float(high)
        dom = int(np.argmax(power))
        feats[f"{name}_fft_dominant_bin_norm"] = float(dom / n_freq)

    # Cross-channel features (compute on first 3 (mean) channels)
    for i, ci in enumerate(["mean_x", "mean_y", "mean_z"]):
        for j, cj in enumerate(["mean_x", "mean_y", "mean_z"]):
            if i >= j:
                continue
            ch_i = x[i]
            ch_j = x[j]
            # First-half vs second-half correlation
            half = T // 2
            r1 = np.corrcoef(ch_i[:half], ch_j[:half])[0, 1] if ch_i[:half].std() > 1e-9 and ch_j[:half].std() > 1e-9 else 0.0
            r2 = np.corrcoef(ch_i[half:], ch_j[half:])[0, 1] if ch_i[half:].std() > 1e-9 and ch_j[half:].std() > 1e-9 else 0.0
            feats[f"corr_{ci}_{cj}_first_half"] = float(r1) if not np.isnan(r1) else 0.0
            feats[f"corr_{ci}_{cj}_second_half"] = float(r2) if not np.isnan(r2) else 0.0
            feats[f"corr_{ci}_{cj}_delta"] = float((r1 - r2)) if not (np.isnan(r1) or np.isnan(r2)) else 0.0

    # Magnitude norm trace (sqrt of sum of squares across mean channels)
    mag = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    feats["mag_max"] = float(mag.max())
    feats["mag_min"] = float(mag.min())
    feats["mag_range"] = float(mag.max() - mag.min())
    feats["mag_peak_pos_norm"] = float(np.argmax(mag) / T)
    mag_d = np.diff(mag)
    feats["mag_diff_max"] = float(np.abs(mag_d).max())
    feats["mag_diff_mean"] = float(mag_d.mean())

    return feats


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.sequences, allow_pickle=True)
    x = d["x"].astype(np.float32)
    file_ids = d["file_ids"].astype(int)
    users = d["users"].astype(str)
    has_labels = "y" in d.files
    y = d["y"].astype(int) if has_labels else None

    print(f"Loaded {len(x)} sequences from {args.sequences}")

    rows = []
    for i in range(len(x)):
        feats = extract_features(x[i])
        feats["file_id"] = int(file_ids[i])
        feats["user_id"] = str(users[i])
        if has_labels and args.with_labels:
            feats["label"] = int(y[i])
        rows.append(feats)
        if (i + 1) % 2000 == 0:
            print(f"  processed {i+1}/{len(x)}")

    df = pd.DataFrame(rows)
    # Reorder so id columns are first
    cols = ["file_id", "user_id"] + ([c for c in df.columns if c not in ("file_id", "user_id", "label")] + (["label"] if "label" in df.columns else []))
    df = df[cols]
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: shape {df.shape}")


if __name__ == "__main__":
    main()
