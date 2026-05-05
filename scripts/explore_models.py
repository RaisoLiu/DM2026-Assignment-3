#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.modeling import tune_probability_class_weights


META_COLS = {"file_id", "user_id", "split", "label"}


@dataclass
class ExperimentResult:
    name: str
    base_macro_f1: float
    calibrated_macro_f1: float
    fold_scores: list[float]
    class_weights: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore grouped-CV HAR models.")
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_search"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--folds-file", type=Path, default=None)
    parser.add_argument("--sample-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument(
        "--context",
        default="base",
        choices=["base", "position", "rolling", "position_rolling", "user_norm", "position_user_norm"],
    )
    parser.add_argument("--models", default="hgb_fast,lgbm_base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_cache)
    frame = add_context_features(frame, args.context)
    features = [c for c in frame.columns if c not in META_COLS]
    X = frame[features]
    y = frame["label"].astype(int).to_numpy()
    groups = frame["user_id"].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(y)))
    fold_ids = load_fold_ids(frame, args.folds_file)
    if fold_ids is None:
        fold_ids = make_default_fold_ids(X, y, groups, n_splits=args.n_splits, seed=args.seed)
    print(f"context={args.context}; X={X.shape}; users={len(np.unique(groups))}; classes={classes.tolist()}", flush=True)
    model_specs = make_models(args.seed)
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []
    for name in selected:
        if name not in model_specs:
            raise ValueError(f"Unknown model {name}. Available: {sorted(model_specs)}")
        print(f"\n=== {name} ===", flush=True)
        result, oof_pred, calibrated_pred, oof_proba = run_cv(
            name=name,
            model=model_specs[name],
            X=X,
            y=y,
            groups=groups,
            classes=classes,
            n_splits=args.n_splits,
            seed=args.seed,
            fold_ids=fold_ids,
            sample_weight_mode=args.sample_weight_mode,
        )
        results.append(result)
        pd.DataFrame(
            {
                "file_id": frame["file_id"],
                "user_id": frame["user_id"],
                "label": y,
                "fold": fold_ids,
                f"{name}_pred": oof_pred,
                f"{name}_calibrated_pred": calibrated_pred,
            }
        ).to_csv(args.output_dir / f"oof_{args.context}_{name}.csv", index=False)
        np.savez_compressed(
            args.output_dir / f"oof_{args.context}_{name}.npz",
            proba=oof_proba,
            classes=classes,
            label=y,
            fold=fold_ids,
            file_id=frame["file_id"].to_numpy(),
            user_id=frame["user_id"].astype(str).to_numpy(),
        )
    out = pd.DataFrame([r.__dict__ for r in results]).sort_values("calibrated_macro_f1", ascending=False)
    out.to_csv(args.output_dir / f"results_{args.context}.csv", index=False)
    (args.output_dir / f"results_{args.context}.json").write_text(json.dumps(out.to_dict(orient="records"), indent=2), encoding="utf-8")
    print("\nSummary")
    print(out.to_string(index=False), flush=True)


def add_context_features(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    frame = frame.sort_values(["user_id", "file_id"]).reset_index(drop=True).copy()
    if context in {"position", "position_rolling"}:
        add_position_features(frame)
    if context == "position_user_norm":
        add_position_features(frame)
    if context in {"user_norm", "position_user_norm"}:
        frame = add_user_norm_features(frame)
    if context in {"rolling", "position_rolling"}:
        base_cols = choose_context_base_columns(frame)
        additions = {}
        for col in base_cols:
            grouped = frame.groupby("user_id", sort=False)[col]
            prev1 = grouped.shift(1)
            next1 = grouped.shift(-1)
            additions[f"{col}_prev1_delta"] = (frame[col] - prev1).fillna(0.0)
            additions[f"{col}_next1_delta"] = (next1 - frame[col]).fillna(0.0)
            for window in (3, 5):
                roll = grouped.transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
                additions[f"{col}_roll{window}_user_mean"] = roll
                additions[f"{col}_roll{window}_user_delta"] = frame[col] - roll
        frame = pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_fold_ids(frame: pd.DataFrame, folds_file: Path | None) -> np.ndarray | None:
    if folds_file is None:
        return None
    folds = pd.read_csv(folds_file, usecols=["file_id", "fold"])
    mapping = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    fold_ids = np.array([mapping[int(file_id)] for file_id in frame["file_id"]], dtype=int)
    if np.any(fold_ids < 1):
        raise ValueError(f"Invalid fold ids loaded from {folds_file}")
    return fold_ids


def make_default_fold_ids(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.full(len(X), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        fold_ids[valid_idx] = fold
    if np.any(fold_ids < 1):
        raise RuntimeError("Some rows were not assigned to a fold")
    return fold_ids


def add_position_features(frame: pd.DataFrame) -> None:
    sizes = frame.groupby("user_id")["file_id"].transform("size").astype(float)
    idx = frame.groupby("user_id").cumcount().astype(float)
    denom = (sizes - 1).clip(lower=1)
    pos = idx / denom
    frame["seq_pos_norm"] = pos
    frame["seq_rev_pos_norm"] = 1.0 - pos
    frame["seq_edge_distance"] = np.minimum(pos, 1.0 - pos)
    frame["seq_pos_sin"] = np.sin(2 * np.pi * pos)
    frame["seq_pos_cos"] = np.cos(2 * np.pi * pos)
    frame["seq_len"] = sizes


def add_user_norm_features(frame: pd.DataFrame) -> pd.DataFrame:
    cols = choose_user_norm_columns(frame)
    additions = {}
    grouped = frame.groupby("user_id", sort=False)
    for col in cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0.0, np.nan)
        additions[f"{col}_user_z"] = ((frame[col] - mean) / std).fillna(0.0)
        additions[f"{col}_user_rank"] = grouped[col].rank(pct=True).fillna(0.5)
    if not additions:
        return frame
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def choose_user_norm_columns(frame: pd.DataFrame) -> list[str]:
    priority_tokens = (
        "mean_vm",
        "std_vm",
        "gravity_abs_delta",
        "mean_x",
        "mean_y",
        "mean_z",
        "std_x",
        "std_y",
        "std_z",
        "orient_",
        "corr_mean_",
        "dynamic_to_static",
    )
    cols = [c for c in frame.columns if c not in META_COLS and c.startswith(priority_tokens)]
    return cols[:220]


def choose_context_base_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "mean_vm_",
        "std_vm_",
        "gravity_abs_delta_",
        "mean_x_",
        "mean_y_",
        "mean_z_",
        "std_x_",
        "std_y_",
        "std_z_",
    )
    suffixes = ("_mean", "_std", "_q10", "_q90", "_range", "_rms", "_diff_abs_mean", "_fft_entropy")
    cols = []
    for col in frame.columns:
        if col in META_COLS:
            continue
        if col.startswith(prefixes) and (col.endswith(suffixes) or "_fft_band_" in col):
            cols.append(col)
    return cols[:90]


def make_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "hgb_fast": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.035,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hgb_deeper": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.025,
                        max_leaf_nodes=63,
                        min_samples_leaf=15,
                        l2_regularization=0.03,
                        random_state=seed + 1,
                    ),
                ),
            ]
        ),
        "extra_trees_500": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features=0.45,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed + 2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_500": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features=0.45,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed + 3,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models.update(
            {
                "lgbm_base": LGBMClassifier(
                    objective="multiclass",
                    num_class=6,
                    n_estimators=900,
                    learning_rate=0.025,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,
                    reg_lambda=0.2,
                    random_state=seed + 10,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                "lgbm_leaves63": LGBMClassifier(
                    objective="multiclass",
                    num_class=6,
                    n_estimators=900,
                    learning_rate=0.02,
                    num_leaves=63,
                    min_child_samples=12,
                    subsample=0.9,
                    colsample_bytree=0.75,
                    reg_alpha=0.02,
                    reg_lambda=0.15,
                    random_state=seed + 11,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            }
        )
    except Exception as exc:
        print(f"LightGBM unavailable: {exc}", flush=True)
    try:
        from xgboost import XGBClassifier

        models["xgb_base"] = XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=450,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.02,
            reg_lambda=0.2,
            random_state=seed + 20,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        )
    except Exception as exc:
        print(f"XGBoost unavailable: {exc}", flush=True)
    try:
        from catboost import CatBoostClassifier

        models["catboost_base"] = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=900,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed + 30,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        models["catboost_depth8"] = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=700,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=4.0,
            random_seed=seed + 31,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    except Exception as exc:
        print(f"CatBoost unavailable: {exc}", flush=True)
    return models


def fit_model(model, X_train, y_train, sample_weight):
    if sample_weight is None:
        model.fit(X_train, y_train)
        return
    try:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    except TypeError:
        model.fit(X_train, y_train, sample_weight=sample_weight)


def aligned_proba(model, X, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    elif hasattr(model, "named_steps"):
        model_classes = model.named_steps["model"].classes_
    else:
        model_classes = classes
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    for idx, cls in enumerate(model_classes):
        aligned[:, np.where(classes == cls)[0][0]] = raw[:, idx]
    return aligned


def run_cv(
    name: str,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    n_splits: int,
    seed: int,
    fold_ids: np.ndarray,
    sample_weight_mode: str,
):
    oof = np.zeros((len(X), len(classes)), dtype=float)
    fold_scores: list[float] = []
    for fold in sorted(np.unique(fold_ids)):
        tr = np.flatnonzero(fold_ids != fold)
        va = np.flatnonzero(fold_ids == fold)
        m = clone(model)
        sample_weight = None
        if sample_weight_mode == "balanced":
            sample_weight = compute_sample_weight("balanced", y[tr])
        fit_model(m, X.iloc[tr], y[tr], sample_weight)
        proba = aligned_proba(m, X.iloc[va], classes)
        oof[va] = proba
        pred = classes[np.argmax(proba, axis=1)]
        score = float(f1_score(y[va], pred, average="macro"))
        fold_scores.append(score)
        print(f"{name} fold {fold}: {score:.5f}", flush=True)
    base_pred = classes[np.argmax(oof, axis=1)]
    base_score = float(f1_score(y, base_pred, average="macro"))
    tuned = tune_probability_class_weights(oof, y, classes, n_passes=6)
    calibrated_pred = tuned["pred"]
    print(classification_report(y, calibrated_pred, digits=4, zero_division=0), flush=True)
    result = ExperimentResult(
        name=name,
        base_macro_f1=base_score,
        calibrated_macro_f1=float(tuned["macro_f1"]),
        fold_scores=fold_scores,
        class_weights=[float(v) for v in tuned["weights"]],
    )
    return result, base_pred, calibrated_pred, oof


if __name__ == "__main__":
    main()
