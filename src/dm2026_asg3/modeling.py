from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class CVResult:
    name: str
    macro_f1: float
    fold_scores: list[float]
    oof_proba: np.ndarray
    oof_pred: np.ndarray
    models: list[object]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"file_id", "user_id", "split", "label"}
    return [c for c in frame.columns if c not in excluded]


def make_model_specs(seed: int, fast: bool = False) -> dict[str, object]:
    n_tree = 180 if fast else 800
    rf_tree = 160 if fast else 600
    hgb_iter = 90 if fast else 450
    return {
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=n_tree,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=rf_tree,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed + 7,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=hgb_iter,
                        learning_rate=0.035,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=seed + 13,
                    ),
                ),
            ]
        ),
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed + 19,
                    ),
                ),
            ]
        ),
    }


def select_model_specs(model_specs: dict[str, object], names: str) -> dict[str, object]:
    if names == "all":
        return model_specs
    requested = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in requested if name not in model_specs]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Available: {sorted(model_specs)}")
    return {name: model_specs[name] for name in requested}


def make_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    unique_groups = np.unique(groups)
    n_classes = len(np.unique(y))
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two user groups are needed for group-aware CV.")
    class_counts = np.bincount(y.astype(int))
    min_class_count = class_counts[class_counts > 0].min() if class_counts.size else 0
    if min_class_count >= n_splits and len(unique_groups) >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if len(unique_groups) >= n_splits:
        return GroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=seed)


def _fit_model(model, X_train, y_train, sample_weight):
    try:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    except TypeError:
        model.fit(X_train, y_train)
    return model


def _aligned_proba(model, X, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    model_classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    for idx, cls in enumerate(model_classes):
        aligned[:, np.where(classes == cls)[0][0]] = raw[:, idx]
    return aligned


def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    n_splits: int,
    seed: int,
) -> CVResult:
    cv = make_cv(y, groups, n_splits=n_splits, seed=seed)
    oof_proba = np.zeros((len(X), len(classes)), dtype=float)
    fold_scores: list[float] = []
    models: list[object] = []
    split_iter = cv.split(X, y, groups) if isinstance(cv, (GroupKFold, StratifiedGroupKFold)) else cv.split(X, y)
    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        fold_model = clone(model)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y[train_idx])
        _fit_model(fold_model, X.iloc[train_idx], y[train_idx], sample_weight)
        proba = _aligned_proba(fold_model, X.iloc[valid_idx], classes)
        oof_proba[valid_idx] = proba
        pred = classes[np.argmax(proba, axis=1)]
        score = f1_score(y[valid_idx], pred, average="macro")
        fold_scores.append(float(score))
        models.append(fold_model)
        print(f"{name} fold {fold_idx}: macro-F1={score:.5f}")
    oof_pred = classes[np.argmax(oof_proba, axis=1)]
    macro_f1 = f1_score(y, oof_pred, average="macro")
    return CVResult(
        name=name,
        macro_f1=float(macro_f1),
        fold_scores=fold_scores,
        oof_proba=oof_proba,
        oof_pred=oof_pred,
        models=models,
    )


def cross_validate_ensemble(results: list[CVResult], y: np.ndarray, classes: np.ndarray) -> dict[str, object]:
    avg = np.mean([r.oof_proba for r in results], axis=0)
    pred = classes[np.argmax(avg, axis=1)]
    return {
        "name": "soft_vote_ensemble",
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "oof_pred": pred,
        "oof_proba": avg,
    }


def predict_from_weighted_proba(proba: np.ndarray, classes: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    weighted = proba * class_weights.reshape(1, -1)
    return classes[np.argmax(weighted, axis=1)]


def tune_probability_class_weights(
    proba: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    n_passes: int = 5,
) -> dict[str, object]:
    candidate_factors = np.array([0.25, 0.35, 0.50, 0.70, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0, 5.5, 7.0])
    weights = np.ones(len(classes), dtype=float)
    best_pred = predict_from_weighted_proba(proba, classes, weights)
    best_score = float(f1_score(y, best_pred, average="macro"))
    for _ in range(n_passes):
        improved = False
        for class_idx in range(len(classes)):
            current_best = best_score
            current_weight = weights[class_idx]
            for factor in candidate_factors:
                trial = weights.copy()
                trial[class_idx] = current_weight * factor
                trial = trial / np.exp(np.mean(np.log(trial + 1e-12)))
                pred = predict_from_weighted_proba(proba, classes, trial)
                score = float(f1_score(y, pred, average="macro"))
                if score > current_best + 1e-8:
                    current_best = score
                    weights = trial
                    best_score = score
                    improved = True
        if not improved:
            break
    return {
        "weights": weights,
        "macro_f1": best_score,
        "pred": predict_from_weighted_proba(proba, classes, weights),
    }


def calibrate_cv_predictions(results: list[CVResult], ensemble: dict[str, object], y: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    rows = []
    candidates = [(result.name, result.macro_f1, result.oof_proba) for result in results]
    candidates.append((str(ensemble["name"]), float(ensemble["macro_f1"]), ensemble["oof_proba"]))
    for name, base_score, proba in candidates:
        tuned = tune_probability_class_weights(proba, y, classes)
        row = {
            "model": name,
            "base_macro_f1": base_score,
            "calibrated_macro_f1": tuned["macro_f1"],
        }
        for cls, weight in zip(classes, tuned["weights"], strict=True):
            row[f"class_{int(cls)}_weight"] = float(weight)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("calibrated_macro_f1", ascending=False)


def train_full_models(model_specs: dict[str, object], X: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    trained = {}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    for name, model in model_specs.items():
        full_model = clone(model)
        _fit_model(full_model, X, y, sample_weight)
        trained[name] = full_model
    return trained


def predict_ensemble(models: dict[str, object], X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    probas = [_aligned_proba(model, X, classes) for model in models.values()]
    return np.mean(probas, axis=0)


def predict_model_proba(model, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    return _aligned_proba(model, X, classes)


def save_diagnostics(
    output_dir: Path,
    y: np.ndarray,
    classes: np.ndarray,
    results: list[CVResult],
    ensemble: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.name,
                "macro_f1": result.macro_f1,
                "fold_scores": ",".join(f"{v:.5f}" for v in result.fold_scores),
            }
        )
    rows.append({"model": ensemble["name"], "macro_f1": ensemble["macro_f1"], "fold_scores": ""})
    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    report = classification_report(y, ensemble["oof_pred"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    cm = confusion_matrix(y, ensemble["oof_pred"], labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output_dir / "confusion_matrix.csv")
    joblib.dump({"classes": classes, "metrics": metrics.to_dict(orient="records")}, output_dir / "metrics.joblib")
