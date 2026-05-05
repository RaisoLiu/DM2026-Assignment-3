#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from explore_models import META_COLS, add_context_features, aligned_proba, make_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explored boosting model on all train data and write a Kaggle CSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--test-feature-cache", type=Path, default=Path("artifacts/features/test_features.csv"))
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model key from scripts/explore_models.py, e.g. lgbm_leaves63 or xgb_base.")
    parser.add_argument("--context", default="position", choices=["base", "position", "rolling", "position_rolling", "user_norm", "position_user_norm"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-weight-mode", choices=["balanced", "none"], default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = add_context_features(pd.read_csv(args.train_feature_cache), args.context)
    test = add_context_features(pd.read_csv(args.test_feature_cache), args.context)
    features = [c for c in train.columns if c not in META_COLS]
    for col in [c for c in features if c not in test.columns]:
        test[col] = 0.0

    X_train = train[features]
    y = train["label"].astype(int).to_numpy()
    X_test = test[features]
    classes = np.array(sorted(np.unique(y)))

    model_specs = make_models(args.seed)
    if args.model not in model_specs:
        raise ValueError(f"Unknown model {args.model}. Available: {sorted(model_specs)}")
    class_weights = load_class_weights(args.results_csv, args.model)

    model = model_specs[args.model]
    sample_weight = compute_sample_weight("balanced", y) if args.sample_weight_mode == "balanced" else None
    print(f"Training {args.model} on X={X_train.shape}; test={X_test.shape}; context={args.context}")
    if sample_weight is None:
        model.fit(X_train, y)
    else:
        model.fit(X_train, y, sample_weight=sample_weight)

    proba = aligned_proba(model, X_test, classes)
    pred = classes[np.argmax(proba * class_weights.reshape(1, -1), axis=1)]
    submission = merge_submission(args.data_dir, test, pred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} rows={len(submission)}")
    print(submission["Label"].value_counts().sort_index().to_dict())


def load_class_weights(results_csv: Path, model_name: str) -> np.ndarray:
    results = pd.read_csv(results_csv)
    row = results.loc[results["name"] == model_name]
    if row.empty:
        raise ValueError(f"Model {model_name} not found in {results_csv}")
    return np.array(ast.literal_eval(str(row.iloc[0]["class_weights"])), dtype=float)


def merge_submission(data_dir: Path, test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    pred_by_id = pd.DataFrame({"_join_id": test["file_id"].map(normalize_id).astype(str), "Label": pred})
    sample = load_sample_submission(data_dir)
    submission = sample[["Id"]].copy()
    submission["_join_id"] = submission["Id"].map(normalize_id).astype(str)
    submission = submission.merge(pred_by_id, on="_join_id", how="left").drop(columns=["_join_id"])
    if submission["Label"].isna().any():
        missing_ids = submission.loc[submission["Label"].isna(), "Id"].head().tolist()
        raise ValueError(f"Missing predictions for sample_submission IDs: {missing_ids}")
    submission["Label"] = submission["Label"].astype(int)
    return submission


if __name__ == "__main__":
    main()
