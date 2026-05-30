#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.transformations.collection.feature_based import Catch22
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.modeling import tune_probability_class_weights
from evaluate_aeon_rocket import make_representation
from explore_models import make_default_fold_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Catch22 sequence features with grouped OOF.")
    parser.add_argument("--sequence-cache", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/catch22_oof"))
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--representation", choices=["raw", "augmented"], default="augmented")
    parser.add_argument("--models", default="lgbm_c22,xgb_c22")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--catch24", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.sequence_cache, allow_pickle=True)
    y = data["y"].astype(int)
    users = data["users"].astype(str)
    file_ids = data["file_ids"].astype(int)
    classes = np.array(sorted(np.unique(y)), dtype=int)
    if args.feature_cache is not None and args.feature_cache.exists():
        features = pd.read_csv(args.feature_cache)
        x_features = features.drop(columns=["file_id", "user_id", "label"], errors="ignore")
        print(f"Loaded Catch22 features {args.feature_cache}: {x_features.shape}", flush=True)
    else:
        x = make_representation(data["x"].astype(np.float32), args.representation)
        print(f"Extracting Catch22: representation={args.representation}; x={x.shape}", flush=True)
        transformer = Catch22(catch24=args.catch24, replace_nans=True, n_jobs=1)
        values = transformer.fit_transform(x)
        x_features = pd.DataFrame(values, columns=[f"c22_{idx:03d}" for idx in range(values.shape[1])])
        features = x_features.copy()
        features.insert(0, "file_id", file_ids)
        features.insert(1, "user_id", users)
        features["label"] = y
        if args.feature_cache is not None:
            args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.feature_cache, index=False)
            print(f"Saved Catch22 features {args.feature_cache}: {x_features.shape}", flush=True)

    fold_ids = make_default_fold_ids(x_features, y, users, n_splits=args.n_splits, seed=args.seed)
    model_specs = make_models(args.seed)
    selected = [name.strip() for name in args.models.split(",") if name.strip()]
    rows = []
    for name in selected:
        if name not in model_specs:
            raise ValueError(f"Unknown model {name}. Available: {sorted(model_specs)}")
        print(f"\n=== {name} ===", flush=True)
        oof, fold_scores = run_cv(model_specs[name], x_features, y, classes, fold_ids)
        base_pred = classes[np.argmax(oof, axis=1)]
        base_score = float(f1_score(y, base_pred, average="macro"))
        tuned = tune_probability_class_weights(oof, y, classes, n_passes=6)
        print(classification_report(y, tuned["pred"], digits=4, zero_division=0), flush=True)
        row = {
            "name": name,
            "base_macro_f1": base_score,
            "calibrated_macro_f1": float(tuned["macro_f1"]),
            "fold_scores": fold_scores,
            "class_weights": [float(v) for v in tuned["weights"]],
        }
        rows.append(row)
        stem = f"oof_catch22_{args.representation}_{name}"
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            proba=oof,
            classes=classes,
            label=y,
            fold=fold_ids,
            file_id=file_ids,
            user_id=users,
        )
        (args.output_dir / f"{stem}_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary = pd.DataFrame(rows).sort_values("calibrated_macro_f1", ascending=False)
    summary.to_csv(args.output_dir / f"results_catch22_{args.representation}.csv", index=False)
    print("\nSummary")
    print(summary.to_string(index=False), flush=True)


def make_models(seed: int) -> dict[str, object]:
    return {
        "lgbm_c22": LGBMClassifier(
            objective="multiclass",
            num_class=6,
            n_estimators=700,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.04,
            reg_lambda=0.4,
            random_state=seed + 101,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb_c22": XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=420,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2.5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.04,
            reg_lambda=0.4,
            random_state=seed + 102,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        ),
        "hgb_c22": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.03,
                        max_leaf_nodes=31,
                        l2_regularization=0.08,
                        random_state=seed + 103,
                    ),
                ),
            ]
        ),
        "extra_c22": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features=0.5,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed + 104,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def run_cv(model, x_features: pd.DataFrame, y: np.ndarray, classes: np.ndarray, fold_ids: np.ndarray):
    oof = np.zeros((len(y), len(classes)), dtype=float)
    fold_scores: list[float] = []
    for fold in sorted(np.unique(fold_ids)):
        tr = np.flatnonzero(fold_ids != fold)
        va = np.flatnonzero(fold_ids == fold)
        m = clone(model)
        sample_weight = compute_sample_weight("balanced", y[tr])
        try:
            m.fit(x_features.iloc[tr], y[tr], sample_weight=sample_weight)
        except (TypeError, ValueError):
            try:
                m.fit(x_features.iloc[tr], y[tr], model__sample_weight=sample_weight)
            except (TypeError, ValueError):
                m.fit(x_features.iloc[tr], y[tr])
        proba = aligned_proba(m, x_features.iloc[va], classes)
        oof[va] = proba
        pred = classes[np.argmax(proba, axis=1)]
        score = float(f1_score(y[va], pred, average="macro"))
        fold_scores.append(score)
        print(f"fold {fold}: {score:.5f}", flush=True)
    return oof, fold_scores


def aligned_proba(model, x_valid: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x_valid)
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    else:
        model_classes = model.named_steps["model"].classes_
    aligned = np.zeros((len(x_valid), len(classes)), dtype=float)
    for idx, cls in enumerate(model_classes.astype(int)):
        aligned[:, np.where(classes == cls)[0][0]] = raw[:, idx]
    return aligned


if __name__ == "__main__":
    main()
