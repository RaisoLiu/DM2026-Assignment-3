#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from explore_models import META_COLS, add_context_features, aligned_proba, fit_model, make_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train explored models on all train data and write a blended Kaggle CSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--test-feature-cache", type=Path, default=Path("artifacts/features/test_features.csv"))
    parser.add_argument("--context", default="position", choices=["base", "position", "rolling", "position_rolling", "user_norm", "position_user_norm"])
    parser.add_argument(
        "--model-weights",
        required=True,
        help="Comma-separated model weights, e.g. lgbm_leaves63=0.55,xgb_base=0.45.",
    )
    parser.add_argument(
        "--class-weights",
        required=True,
        help="Comma-separated class decision weights in class order, e.g. 0.5,0.6,3.0,1.0,1.1,1.0.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-proba", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-weight-mode", choices=["balanced", "none"], default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_weights = parse_model_weights(args.model_weights)
    class_weights = parse_class_weights(args.class_weights)

    train = add_context_features(pd.read_csv(args.train_feature_cache), args.context)
    test = add_context_features(pd.read_csv(args.test_feature_cache), args.context)
    features = [c for c in train.columns if c not in META_COLS]
    for col in [c for c in features if c not in test.columns]:
        test[col] = 0.0

    X_train = train[features]
    y = train["label"].astype(int).to_numpy()
    X_test = test[features]
    classes = np.array(sorted(np.unique(y)))
    if len(class_weights) != len(classes):
        raise ValueError(f"Expected {len(classes)} class weights, got {len(class_weights)}")

    model_specs = make_models(args.seed)
    unknown = [name for name in model_weights if name not in model_specs]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Available: {sorted(model_specs)}")

    sample_weight = compute_sample_weight("balanced", y) if args.sample_weight_mode == "balanced" else None
    blended = np.zeros((len(X_test), len(classes)), dtype=float)
    per_model = {}
    print(
        f"context={args.context}; train={X_train.shape}; test={X_test.shape}; "
        f"models={model_weights}; sample_weight_mode={args.sample_weight_mode}",
        flush=True,
    )
    for name, weight in model_weights.items():
        print(f"Training {name} with blend weight {weight:.6f}", flush=True)
        model = clone(model_specs[name])
        fit_model(model, X_train, y, sample_weight)
        proba = aligned_proba(model, X_test, classes)
        blended += weight * proba
        per_model[f"{name}_proba"] = proba

    pred = classes[np.argmax(blended * class_weights.reshape(1, -1), axis=1)]
    submission = merge_submission(args.data_dir, test, pred)
    validate_submission(args.data_dir, submission)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    digest = sha256_file(args.output)
    counts = submission["Label"].value_counts().sort_index().to_dict()
    print(f"Wrote {args.output} rows={len(submission)} sha256={digest}")
    print(f"label_counts={counts}")

    if args.save_proba is not None:
        args.save_proba.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_proba,
            proba=blended,
            classes=classes,
            file_id=test["file_id"].to_numpy(),
            model_names=np.array(list(model_weights.keys()), dtype=object),
            model_weights=np.array(list(model_weights.values()), dtype=float),
            class_weights=class_weights,
            **per_model,
        )
        print(f"Saved probabilities to {args.save_proba}")


def parse_model_weights(text: str) -> dict[str, float]:
    pairs = [part.strip() for part in text.split(",") if part.strip()]
    if not pairs:
        raise ValueError("--model-weights cannot be empty")
    weights: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid model weight {pair!r}; expected name=value")
        name, value = pair.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid model weight {pair!r}; empty model name")
        weights[name] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Model weights must sum to a positive value")
    return {name: weight / total for name, weight in weights.items()}


def parse_class_weights(text: str) -> np.ndarray:
    weights = np.array([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)
    if weights.size == 0 or np.any(weights <= 0):
        raise ValueError("Class weights must be positive")
    return weights


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


def validate_submission(data_dir: Path, submission: pd.DataFrame) -> None:
    sample = load_sample_submission(data_dir)
    if list(submission.columns) != ["Id", "Label"]:
        raise ValueError(f"Invalid columns: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, got {len(submission)}")
    if submission["Id"].duplicated().any():
        raise ValueError("Duplicate Id values in submission")
    if not submission["Id"].equals(sample["Id"]):
        raise ValueError("Submission Id order does not match sample_submission.csv")
    labels = set(submission["Label"].astype(int).unique())
    if not labels.issubset({0, 1, 2, 3, 4, 5}):
        raise ValueError(f"Invalid labels: {sorted(labels)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
