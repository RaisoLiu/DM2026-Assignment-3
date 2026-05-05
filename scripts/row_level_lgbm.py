#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from dm2026_asg3.modeling import tune_probability_class_weights


SIGNAL_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Row-level LGBM with file-level aggregation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/row_level"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/row_lgbm"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--row-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, files = load_or_build(args.data_dir, args.cache_dir)
    if args.row_stride > 1:
        rows = rows[rows["index"] % args.row_stride == 0].reset_index(drop=True)
        print(f"Using row stride {args.row_stride}: {len(rows)} rows", flush=True)
    feature_cols = [c for c in rows.columns if c not in {"file_id", "user_id", "label"}]
    file_y = files["label"].astype(int).to_numpy()
    file_groups = files["user_id"].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(file_y)))
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    file_oof = np.zeros((len(files), len(classes)), dtype=float)
    file_index = {fid: i for i, fid in enumerate(files["file_id"])}
    folds = []
    for fold, (tr_file_idx, va_file_idx) in enumerate(cv.split(files, file_y, file_groups), start=1):
        tr_ids = set(files.iloc[tr_file_idx]["file_id"])
        va_ids = set(files.iloc[va_file_idx]["file_id"])
        tr_mask = rows["file_id"].isin(tr_ids).to_numpy()
        va_mask = rows["file_id"].isin(va_ids).to_numpy()
        X_tr = rows.loc[tr_mask, feature_cols]
        y_tr = rows.loc[tr_mask, "label"].astype(int).to_numpy()
        X_va = rows.loc[va_mask, feature_cols]
        va_file_ids = rows.loc[va_mask, "file_id"].to_numpy()
        sample_weight = compute_sample_weight("balanced", y_tr)
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            n_estimators=args.n_estimators,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=200,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=0.2,
            random_state=args.seed + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        print(f"fold {fold}: train rows={len(X_tr)} valid rows={len(X_va)}", flush=True)
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        row_proba = model.predict_proba(X_va)
        agg = aggregate_by_file(row_proba, va_file_ids, classes)
        for fid, proba in agg.items():
            file_oof[file_index[fid]] = proba
        pred = classes[np.argmax(file_oof[va_file_idx], axis=1)]
        score = float(f1_score(file_y[va_file_idx], pred, average="macro"))
        folds.append(score)
        print(f"fold {fold}: file macro-F1={score:.5f}", flush=True)
    base_pred = classes[np.argmax(file_oof, axis=1)]
    base = float(f1_score(file_y, base_pred, average="macro"))
    cal = tune_probability_class_weights(file_oof, file_y, classes, n_passes=6)
    print("base", base, "calibrated", cal["macro_f1"], "folds", folds, flush=True)
    print(classification_report(file_y, cal["pred"], digits=4, zero_division=0), flush=True)
    np.savez_compressed(args.output_dir / "oof_row_lgbm.npz", proba=file_oof, label=file_y, classes=classes)
    pd.DataFrame({"model": ["row_lgbm"], "base_macro_f1": [base], "calibrated_macro_f1": [cal["macro_f1"]], "fold_scores": [",".join(f"{x:.5f}" for x in folds)]}).to_csv(args.output_dir / "cv_metrics.csv", index=False)


def load_or_build(data_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_path = cache_dir / "train_rows.pkl"
    file_path = cache_dir / "train_files.csv"
    if row_path.exists() and file_path.exists():
        return pd.read_pickle(row_path), pd.read_csv(file_path)
    rows = []
    files = []
    for path in sorted((data_dir / "train").rglob("*.csv")):
        df = pd.read_csv(path).sort_values("index").reset_index(drop=True)
        fid = int(df["file_id"].iloc[0])
        label = int(df["label"].iloc[0])
        user = path.parent.name
        row = df[["index", *SIGNAL_COLS]].copy()
        row["file_id"] = fid
        row["user_id"] = user
        row["label"] = label
        add_row_features(row)
        rows.append(row)
        files.append({"file_id": fid, "user_id": user, "label": label})
    rows_df = pd.concat(rows, ignore_index=True)
    files_df = pd.DataFrame(files)
    rows_df.to_pickle(row_path)
    files_df.to_csv(file_path, index=False)
    return rows_df, files_df


def add_row_features(row: pd.DataFrame) -> None:
    mx = row["mean_x"].to_numpy()
    my = row["mean_y"].to_numpy()
    mz = row["mean_z"].to_numpy()
    sx = row["std_x"].to_numpy()
    sy = row["std_y"].to_numpy()
    sz = row["std_z"].to_numpy()
    row["mean_vm"] = np.sqrt(mx * mx + my * my + mz * mz)
    row["std_vm"] = np.sqrt(sx * sx + sy * sy + sz * sz)
    row["gravity_delta"] = np.abs(row["mean_vm"] - 1.0)
    row["t_norm"] = row["index"] / 299.0
    for col in SIGNAL_COLS + ["mean_vm", "std_vm"]:
        row[f"{col}_diff1"] = row[col].diff().fillna(0.0)
        row[f"{col}_roll5"] = row[col].rolling(5, center=True, min_periods=1).mean()
        row[f"{col}_roll15"] = row[col].rolling(15, center=True, min_periods=1).mean()


def aggregate_by_file(row_proba: np.ndarray, file_ids: np.ndarray, classes: np.ndarray) -> dict[int, np.ndarray]:
    df = pd.DataFrame(row_proba, columns=[f"p_{c}" for c in classes])
    df["file_id"] = file_ids
    means = df.groupby("file_id").mean()
    # Blend mean and high-percentile probabilities so rare transient classes can survive averaging.
    q90 = df.groupby("file_id").quantile(0.90)
    out = {}
    for fid in means.index:
        p = 0.75 * means.loc[fid].to_numpy() + 0.25 * q90.loc[fid].to_numpy()
        out[int(fid)] = p / p.sum()
    return out


if __name__ == "__main__":
    main()
