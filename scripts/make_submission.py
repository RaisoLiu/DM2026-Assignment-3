#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import discover_records, load_sample_submission, load_window_csv, normalize_id, validate_training_records
from dm2026_asg3.features import build_feature_frame
from dm2026_asg3.modeling import feature_columns, make_model_specs, predict_ensemble, train_full_models
from dm2026_asg3.modeling import predict_model_proba, select_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final models and create Kaggle submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_ensemble.csv"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fast", action="store_true", help="Use smaller models for a quick smoke run.")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    parser.add_argument("--prediction-strategy", default="auto", choices=["auto", "ensemble", "best_cv_model"])
    parser.add_argument("--rebuild-features", action="store_true", help="Ignore cached feature CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    sample = load_sample_submission(args.data_dir)
    train_frame = load_or_build_features(args, "train")
    test_frame = load_or_build_features(args, "test")
    selected_feature_path = args.experiment_dir / "selected_features.txt"
    if selected_feature_path.exists():
        features = [line.strip() for line in selected_feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Using {len(features)} selected features from {selected_feature_path}")
    else:
        features = feature_columns(train_frame)
    missing = [col for col in features if col not in test_frame.columns]
    for col in missing:
        test_frame[col] = 0.0
    X_train = train_frame[features]
    y = train_frame["label"].astype(int).to_numpy()
    X_test = test_frame[features]
    classes = np.array(sorted(np.unique(y)))
    model_specs = select_model_specs(make_model_specs(args.seed, fast=args.fast), args.models)
    models = train_full_models(model_specs, X_train, y)
    proba, class_weights, strategy_name = choose_prediction_proba(args, models, X_test, classes)
    pred = classes[np.argmax(proba * class_weights.reshape(1, -1), axis=1)]
    pred_by_id = pd.DataFrame({"_join_id": test_frame["file_id"].map(normalize_id).astype(str), "Label": pred})
    submission = sample[["Id"]].copy()
    submission["_join_id"] = submission["Id"].map(normalize_id).astype(str)
    submission = submission.merge(pred_by_id, on="_join_id", how="left").drop(columns=["_join_id"])
    if submission["Label"].isna().any():
        missing_ids = submission.loc[submission["Label"].isna(), "Id"].head().tolist()
        raise ValueError(f"Missing predictions for sample_submission IDs: {missing_ids}")
    submission["Label"] = submission["Label"].astype(int)
    submission.to_csv(args.output, index=False)
    joblib.dump(
        {
            "models": models,
            "features": features,
            "classes": classes,
            "prediction_strategy": strategy_name,
            "class_weights": class_weights,
            "test_file_ids": test_frame["file_id"].tolist(),
        },
        args.experiment_dir / "submission_models.joblib",
    )
    print(f"Wrote {args.output} with {len(submission)} rows.")


def load_or_build_features(args: argparse.Namespace, split: str) -> pd.DataFrame:
    cache_path = args.feature_cache_dir / f"{split}_features.csv"
    if cache_path.exists() and not args.rebuild_features:
        print(f"Loading cached {split} features from {cache_path}")
        return pd.read_csv(cache_path)
    records = discover_records(args.data_dir, split)
    if split == "train":
        validate_training_records(records)
    elif not records:
        raise FileNotFoundError("No test CSV files found under data/raw/test")
    frame = build_feature_frame(records, load_window_csv)
    frame.to_csv(cache_path, index=False)
    print(f"Cached {split} features to {cache_path}")
    return frame


def choose_prediction_proba(args: argparse.Namespace, models: dict[str, object], X_test: pd.DataFrame, classes: np.ndarray):
    weights = np.ones(len(classes), dtype=float)
    if args.prediction_strategy == "ensemble":
        return predict_ensemble(models, X_test, classes), weights, "ensemble"
    calibration_path = args.experiment_dir / "decision_calibration.csv"
    if args.prediction_strategy == "auto" and calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        for _, row in calibration.sort_values("calibrated_macro_f1", ascending=False).iterrows():
            name = str(row["model"])
            if name == "soft_vote_ensemble":
                proba = predict_ensemble(models, X_test, classes)
            elif name in models:
                proba = predict_model_proba(models[name], X_test, classes)
            else:
                continue
            weights = np.array([float(row[f"class_{int(cls)}_weight"]) for cls in classes], dtype=float)
            print(f"Using calibrated prediction strategy: {name}")
            return proba, weights, f"{name}_calibrated"
    if args.prediction_strategy == "best_cv_model" and calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        for _, row in calibration.sort_values("base_macro_f1", ascending=False).iterrows():
            name = str(row["model"])
            if name in models:
                print(f"Using best CV single model without calibration: {name}")
                return predict_model_proba(models[name], X_test, classes), weights, name
    return predict_ensemble(models, X_test, classes), weights, "ensemble_fallback"


if __name__ == "__main__":
    main()
