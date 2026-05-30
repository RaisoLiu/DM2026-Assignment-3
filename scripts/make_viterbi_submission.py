#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from dm2026_asg3.modeling import tune_probability_class_weights
from evaluate_sequence_smoothing import estimate_transition_model, tune_viterbi_params, viterbi_predict_by_user
from explore_models import META_COLS, add_context_features, aligned_proba, fit_model, make_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full model and write a Viterbi-smoothed Kaggle CSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--test-feature-cache", type=Path, default=Path("artifacts/features/test_features.csv"))
    parser.add_argument("--oof-npz", type=Path, required=True)
    parser.add_argument("--model", default="xgb_base")
    parser.add_argument("--context", default="position", choices=["base", "position", "rolling", "position_rolling", "user_norm", "position_user_norm"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-proba", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--calibration-passes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = add_context_features(pd.read_csv(args.train_feature_cache), args.context)
    test = add_context_features(pd.read_csv(args.test_feature_cache), args.context)
    features = [col for col in train.columns if col not in META_COLS]
    for col in [col for col in features if col not in test.columns]:
        test[col] = 0.0

    X_train = train[features]
    y = train["label"].astype(int).to_numpy()
    X_test = test[features]
    classes = np.array(sorted(np.unique(y)))

    oof = np.load(args.oof_npz, allow_pickle=True)
    oof_proba = oof["proba"].astype(float)
    oof_y = oof["label"].astype(int)
    oof_classes = oof["classes"].astype(int)
    oof_file_id = oof["file_id"].astype(int)
    oof_user_id = oof["user_id"].astype(str)
    if not np.array_equal(classes, oof_classes):
        raise ValueError(f"Class order mismatch: train={classes.tolist()} oof={oof_classes.tolist()}")

    tuned = tune_probability_class_weights(oof_proba, oof_y, classes, n_passes=args.calibration_passes)
    class_weights = tuned["weights"].astype(float)
    best_params = tune_viterbi_params(
        proba=oof_proba,
        y=oof_y,
        classes=classes,
        file_ids=oof_file_id,
        user_ids=oof_user_id,
        class_weights=class_weights,
    )
    oof_transition, oof_start = estimate_transition_model(
        y=oof_y,
        classes=classes,
        file_ids=oof_file_id,
        user_ids=oof_user_id,
        alpha=best_params["alpha"],
    )
    oof_pred = viterbi_predict_by_user(
        proba=oof_proba,
        classes=classes,
        file_ids=oof_file_id,
        user_ids=oof_user_id,
        class_weights=class_weights,
        transition=oof_transition,
        start=oof_start,
        beta=best_params["beta"],
    )
    oof_viterbi_macro_f1 = float(f1_score(oof_y, oof_pred, average="macro"))

    model_specs = make_models(args.seed)
    if args.model not in model_specs:
        raise ValueError(f"Unknown model {args.model}. Available: {sorted(model_specs)}")
    model = model_specs[args.model]
    sample_weight = compute_sample_weight("balanced", y) if args.sample_weight_mode == "balanced" else None
    print(f"Training {args.model} on X={X_train.shape}; test={X_test.shape}; context={args.context}", flush=True)
    fit_model(model, X_train, y, sample_weight)

    test_proba = aligned_proba(model, X_test, classes)
    transition, start = estimate_transition_model(
        y=y,
        classes=classes,
        file_ids=train["file_id"].astype(int).to_numpy(),
        user_ids=train["user_id"].astype(str).to_numpy(),
        alpha=best_params["alpha"],
    )
    pred = viterbi_predict_by_user(
        proba=test_proba,
        classes=classes,
        file_ids=test["file_id"].astype(int).to_numpy(),
        user_ids=test["user_id"].astype(str).to_numpy(),
        class_weights=class_weights,
        transition=transition,
        start=start,
        beta=best_params["beta"],
    )

    submission = merge_submission(args.data_dir, test["file_id"].to_numpy(), pred)
    validate_submission(args.data_dir, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    digest = sha256_file(args.output)
    label_counts = {int(k): int(v) for k, v in submission["Label"].value_counts().sort_index().items()}

    metadata = {
        "model": args.model,
        "context": args.context,
        "sample_weight_mode": args.sample_weight_mode,
        "oof_npz": str(args.oof_npz),
        "oof_global_class_calibrated_macro_f1": float(tuned["macro_f1"]),
        "oof_global_viterbi_macro_f1": oof_viterbi_macro_f1,
        "fold_trained_reference_macro_f1": 0.7496352119149347,
        "class_weights": [float(value) for value in class_weights],
        "alpha": float(best_params["alpha"]),
        "beta": float(best_params["beta"]),
        "train_macro_f1_for_viterbi_params": float(best_params["train_macro_f1"]),
        "label_counts": label_counts,
        "sha256": digest,
        "output": str(args.output),
    }
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_proba is not None:
        args.save_proba.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_proba,
            proba=test_proba,
            classes=classes,
            file_id=test["file_id"].to_numpy(),
            user_id=test["user_id"].astype(str).to_numpy(),
            pred=pred,
            class_weights=class_weights,
            transition=transition,
            start=start,
            alpha=float(best_params["alpha"]),
            beta=float(best_params["beta"]),
        )

    print(f"Wrote {args.output} rows={len(submission)} sha256={digest}")
    print(f"oof_global_viterbi_macro_f1={oof_viterbi_macro_f1:.6f}")
    print(f"alpha={best_params['alpha']} beta={best_params['beta']}")
    print(f"label_counts={label_counts}")


def merge_submission(data_dir: Path, file_id: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    pred_by_id = pd.DataFrame({"_join_id": pd.Series(file_id).map(normalize_id).astype(str), "Label": pred})
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
    if not labels.issubset(set(range(6))):
        raise ValueError(f"Invalid labels: {sorted(labels)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
