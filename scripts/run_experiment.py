#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import discover_records, load_window_csv, validate_training_records
from dm2026_asg3.features import build_feature_frame
from dm2026_asg3.modeling import (
    calibrate_cv_predictions,
    cross_validate_ensemble,
    cross_validate_model,
    feature_columns,
    make_model_specs,
    select_model_specs,
    save_diagnostics,
    train_full_models,
)
from dm2026_asg3.reporting import write_metrics_json, write_report_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAR experiments for DM2026 Assignment 3.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fast", action="store_true", help="Use smaller models for a quick smoke run.")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    parser.add_argument("--rebuild-features", action="store_true", help="Ignore cached feature CSV files.")
    parser.add_argument("--run-feature-ablations", action="store_true", help="Run feature-family ablations.")
    parser.add_argument("--ablation-model", default="hist_gradient_boosting", help="Model used for feature-family ablations.")
    parser.add_argument(
        "--exclude-feature-families",
        default="",
        help="Comma-separated feature families to remove before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    frame = load_or_build_train_features(args)
    frame.to_csv(args.output_dir / "train_features.csv", index=False)
    write_dataset_summary(args.output_dir, frame)
    features = feature_columns(frame)
    features = apply_feature_family_exclusions(features, args.exclude_feature_families)
    (args.output_dir / "selected_features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    X = frame[features]
    y = frame["label"].astype(int).to_numpy()
    groups = frame["user_id"].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(y)))
    print(f"Feature matrix: {X.shape[0]} windows x {X.shape[1]} features")
    print(f"Classes: {classes.tolist()}; users: {len(np.unique(groups))}")
    model_specs = select_model_specs(make_model_specs(args.seed, fast=args.fast), args.models)
    results = []
    for name, model in model_specs.items():
        print(f"\nRunning {name}")
        results.append(
            cross_validate_model(
                name=name,
                model=model,
                X=X,
                y=y,
                groups=groups,
                classes=classes,
                n_splits=args.n_splits,
                seed=args.seed,
            )
        )
    ensemble = cross_validate_ensemble(results, y, classes)
    print(f"\nEnsemble macro-F1={ensemble['macro_f1']:.5f}")
    save_diagnostics(args.output_dir, y, classes, results, ensemble)
    metrics = pd.read_csv(args.output_dir / "cv_metrics.csv")
    calibration = calibrate_cv_predictions(results, ensemble, y, classes)
    calibration.to_csv(args.output_dir / "decision_calibration.csv", index=False)
    write_calibrated_diagnostics(args.output_dir, calibration, results, ensemble, y, classes)
    metrics = append_calibrated_metrics(metrics, calibration)
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    ablation = make_ablation_table(metrics)
    if args.run_feature_ablations:
        ablation_model = select_model_specs(make_model_specs(args.seed, fast=True), args.ablation_model)
        ablation_name, ablation_estimator = next(iter(ablation_model.items()))
        print(f"Using {ablation_name} for feature-family ablations")
        feature_ablation = run_feature_ablations(
            ablation_estimator,
            X=X,
            y=y,
            groups=groups,
            classes=classes,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        feature_ablation.to_csv(args.output_dir / "feature_ablation_results.csv", index=False)
        ablation = pd.concat([ablation, feature_ablation], ignore_index=True)
    ablation.to_csv(args.output_dir / "ablation_results.csv", index=False)
    write_metrics_json(
        args.output_dir,
        metrics,
        metadata={
            "seed": args.seed,
            "n_splits": args.n_splits,
            "n_windows": int(len(frame)),
            "n_features": int(len(features)),
            "n_users": int(len(np.unique(groups))),
        },
    )
    write_report_tables(args.output_dir, metrics, ablation)
    full_models = train_full_models(model_specs, X, y)
    import joblib

    joblib.dump(
        {
            "models": full_models,
            "classes": classes,
            "features": features,
            "seed": args.seed,
        },
        args.output_dir / "full_models.joblib",
    )
    print(f"\nArtifacts written to {args.output_dir}")


def load_or_build_train_features(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = args.feature_cache_dir / "train_features.csv"
    if cache_path.exists() and not args.rebuild_features:
        print(f"Loading cached train features from {cache_path}")
        return pd.read_csv(cache_path)
    train_records = discover_records(args.data_dir, "train")
    validate_training_records(train_records)
    print(f"Discovered {len(train_records)} training windows.")
    frame = build_feature_frame(train_records, load_window_csv)
    frame.to_csv(cache_path, index=False)
    print(f"Cached train features to {cache_path}")
    return frame


def write_dataset_summary(output_dir: Path, frame: pd.DataFrame) -> None:
    label_counts = frame["label"].astype(int).value_counts().sort_index().rename_axis("label").reset_index(name="count")
    label_counts["fraction"] = label_counts["count"] / label_counts["count"].sum()
    label_counts.to_csv(output_dir / "class_distribution.csv", index=False)
    user_counts = frame.groupby("user_id").size().rename("n_windows").reset_index()
    user_counts.to_csv(output_dir / "user_window_counts.csv", index=False)
    summary = pd.DataFrame(
        [
            {"statistic": "n_windows", "value": len(frame)},
            {"statistic": "n_users", "value": frame["user_id"].nunique()},
            {"statistic": "n_classes", "value": frame["label"].nunique()},
            {"statistic": "n_features", "value": len(feature_columns(frame))},
            {"statistic": "min_windows_per_user", "value": int(user_counts["n_windows"].min())},
            {"statistic": "max_windows_per_user", "value": int(user_counts["n_windows"].max())},
            {"statistic": "minority_class_count", "value": int(label_counts["count"].min())},
            {"statistic": "majority_class_count", "value": int(label_counts["count"].max())},
        ]
    )
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)


def append_calibrated_metrics(metrics: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    calibrated_rows = []
    for _, row in calibration.iterrows():
        calibrated_rows.append(
            {
                "model": f"{row['model']}_calibrated",
                "macro_f1": row["calibrated_macro_f1"],
                "fold_scores": "",
            }
        )
    return (
        pd.concat([metrics, pd.DataFrame(calibrated_rows)], ignore_index=True)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )


def write_calibrated_diagnostics(
    output_dir: Path,
    calibration: pd.DataFrame,
    results,
    ensemble: dict[str, object],
    y: np.ndarray,
    classes: np.ndarray,
) -> None:
    best = calibration.sort_values("calibrated_macro_f1", ascending=False).iloc[0]
    proba_by_name = {result.name: result.oof_proba for result in results}
    proba_by_name[str(ensemble["name"])] = ensemble["oof_proba"]
    model_name = str(best["model"])
    proba = proba_by_name[model_name]
    weights = np.array([float(best[f"class_{int(cls)}_weight"]) for cls in classes], dtype=float)
    pred = classes[np.argmax(proba * weights.reshape(1, -1), axis=1)]
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / "calibrated_classification_report.csv")
    cm = confusion_matrix(y, pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output_dir / "calibrated_confusion_matrix.csv")
    payload = {
        "model": model_name,
        "calibrated_macro_f1": float(best["calibrated_macro_f1"]),
        "base_macro_f1": float(best["base_macro_f1"]),
        "class_weights": {str(int(cls)): float(weight) for cls, weight in zip(classes, weights, strict=True)},
    }
    (output_dir / "selected_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def feature_family_masks(features: list[str]) -> dict[str, list[str]]:
    return {
        "gravity_magnitude": [
            c
            for c in features
            if c.startswith(("mean_vm", "std_vm", "gravity_abs_delta"))
            or c in {"mean_vm_to_std_vm_ratio", "dynamic_to_static_energy"}
        ],
        "temporal_derivatives": [c for c in features if "_diff_" in c],
        "segment_rolling": [c for c in features if "_seg" in c or "_roll" in c],
        "frequency": [c for c in features if "_fft_" in c],
        "orientation_cross_axis": [
            c for c in features if c.startswith(("orient_", "corr_mean_", "cov_mean_"))
        ],
    }


def apply_feature_family_exclusions(features: list[str], exclude_feature_families: str) -> list[str]:
    requested = [name.strip().replace("minus_", "") for name in exclude_feature_families.split(",") if name.strip()]
    if not requested:
        return features
    masks = feature_family_masks(features)
    unknown = [name for name in requested if name not in masks]
    if unknown:
        raise ValueError(f"Unknown feature families: {unknown}. Available: {sorted(masks)}")
    removed = set()
    for name in requested:
        removed.update(masks[name])
    kept = [c for c in features if c not in removed]
    print(f"Excluding feature families {requested}: removed {len(removed)} features, kept {len(kept)}")
    return kept


def run_feature_ablations(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    print("\nRunning feature-family ablations")
    full = cross_validate_model(
        name="ablation_full_model",
        model=model,
        X=X,
        y=y,
        groups=groups,
        classes=classes,
        n_splits=n_splits,
        seed=seed,
    )
    rows = [
        {
            "design": "Ablation full feature set",
            "cv_macro_f1": full.macro_f1,
            "delta_from_full": 0.0,
            "n_features": X.shape[1],
            "interpretation": "Reference for feature-family removal tests.",
        }
    ]
    for family, removed in feature_family_masks(list(X.columns)).items():
        kept = [c for c in X.columns if c not in set(removed)]
        if not removed or not kept:
            continue
        result = cross_validate_model(
            name=family,
            model=model,
            X=X[kept],
            y=y,
            groups=groups,
            classes=classes,
            n_splits=n_splits,
            seed=seed,
        )
        rows.append(
            {
                "design": family.replace("_", " "),
                "cv_macro_f1": result.macro_f1,
                "delta_from_full": result.macro_f1 - full.macro_f1,
                "n_features": len(kept),
                "interpretation": f"Removed {len(removed)} related features to test their contribution.",
            }
        )
    return pd.DataFrame(rows)


def make_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[metrics["model"] == "logistic_l2", "macro_f1"]
    ensemble = metrics.loc[metrics["model"] == "soft_vote_ensemble", "macro_f1"]
    rows = []
    if not baseline.empty:
        rows.append(
                {
                    "design": "Linear baseline with engineered window features",
                    "cv_macro_f1": float(baseline.iloc[0]),
                    "delta_from_full": pd.NA,
                    "n_features": pd.NA,
                    "interpretation": "Checks how much separability comes from feature design alone.",
                }
            )
    for name in ("hist_gradient_boosting", "random_forest", "extra_trees"):
        score = metrics.loc[metrics["model"] == name, "macro_f1"]
        if not score.empty:
            rows.append(
                {
                    "design": name,
                    "cv_macro_f1": float(score.iloc[0]),
                    "delta_from_full": pd.NA,
                    "n_features": pd.NA,
                    "interpretation": "Nonlinear tree model on the same aligned features.",
                }
            )
    if not ensemble.empty:
        rows.append(
            {
                "design": "Soft-vote ensemble",
                "cv_macro_f1": float(ensemble.iloc[0]),
                "delta_from_full": pd.NA,
                "n_features": pd.NA,
                "interpretation": "Tests whether complementary model errors improve macro-F1.",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
