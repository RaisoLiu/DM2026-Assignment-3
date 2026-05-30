#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.modeling import tune_probability_class_weights
from explore_models import META_COLS, add_context_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fold-fair stackers on saved OOF probabilities.")
    parser.add_argument("--input", action="append", required=True, help="name=path/to/oof.npz; repeatable")
    parser.add_argument("--folds-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="logreg_l2,lgbm_small,extra_trees")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-passes", type=int, default=8)
    parser.add_argument("--sample-weight-mode", choices=["none", "balanced"], default="none")
    parser.add_argument("--include-log", action="store_true")
    parser.add_argument("--include-aggregate", action="store_true")
    parser.add_argument("--include-seq-context", action="store_true")
    parser.add_argument("--neighbor-shifts", default="", help="Comma-separated user-sequence shifts, e.g. -2,-1,1,2.")
    parser.add_argument("--neighbor-rolling", default="", help="Comma-separated centered rolling window sizes for meta features.")
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--feature-context", default="base", choices=["base", "position", "rolling", "position_rolling", "user_norm", "position_user_norm"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names, bundles = load_inputs(args.input)
    first = bundles[0]
    folds = align_folds(first["file_id"], args.folds_file)
    y = first["label"]
    classes = first["classes"]
    x = build_meta_features(
        bundles,
        first["file_id"],
        first["user_id"],
        include_log=args.include_log,
        include_aggregate=args.include_aggregate,
        include_seq_context=args.include_seq_context,
        neighbor_shifts=parse_int_list(args.neighbor_shifts),
        neighbor_rolling=parse_int_list(args.neighbor_rolling),
    )
    if args.feature_cache is not None:
        x = np.concatenate([x, load_feature_cache(args.feature_cache, args.feature_context, first["file_id"])], axis=1)
    print(
        f"inputs={names}; x={x.shape}; rows={len(y)}; classes={classes.tolist()}; folds={sorted(np.unique(folds).tolist())}",
        flush=True,
    )
    rows = []
    specs = make_models(args.seed)
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if model_name not in specs:
            raise ValueError(f"Unknown model {model_name}. Available: {sorted(specs)}")
        print(f"\n=== {model_name} ===", flush=True)
        proba, fold_scores = run_fold_fair_model(specs[model_name], x, y, classes, folds, args.sample_weight_mode)
        base_pred = classes[np.argmax(proba, axis=1)]
        base_score = float(f1_score(y, base_pred, average="macro"))
        tuned = tune_probability_class_weights(proba, y, classes, n_passes=args.calibration_passes)
        cal_pred = tuned["pred"]
        cal_score = float(tuned["macro_f1"])
        print(classification_report(y, cal_pred, digits=4, zero_division=0), flush=True)
        print(f"{model_name}: base_macro_f1={base_score:.6f} calibrated_macro_f1={cal_score:.6f}", flush=True)
        print(f"{model_name}: class_weights={','.join(f'{v:.6g}' for v in tuned['weights'])}", flush=True)
        np.savez_compressed(
            args.output_dir / f"oof_stack_{model_name}.npz",
            proba=proba,
            classes=classes,
            label=y,
            fold=folds,
            file_id=first["file_id"],
            user_id=first["user_id"],
            model_names=np.array(names, dtype=object),
        )
        pd.DataFrame(classification_report(y, cal_pred, output_dict=True, zero_division=0)).T.to_csv(
            args.output_dir / f"oof_stack_{model_name}_report.csv"
        )
        rows.append(
            {
                "name": model_name,
                "base_macro_f1": base_score,
                "calibrated_macro_f1": cal_score,
                "fold_scores": ",".join(f"{score:.6f}" for score in fold_scores),
                "class_weights": ",".join(f"{value:.6g}" for value in tuned["weights"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("calibrated_macro_f1", ascending=False)
    out.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(out.to_dict(orient="records"), indent=2), encoding="utf-8")
    print("\nSummary")
    print(out.to_string(index=False), flush=True)


def load_inputs(specs: list[str]) -> tuple[list[str], list[dict[str, np.ndarray]]]:
    names = []
    bundles = []
    base = None
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid input spec {spec!r}; expected name=path")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        data = np.load(path, allow_pickle=True)
        bundle = {
            "proba": data["proba"].astype(float),
            "classes": data["classes"].astype(int),
            "label": data["label"].astype(int),
            "file_id": data["file_id"].astype(int),
            "user_id": data["user_id"].astype(str),
        }
        if base is None:
            base = bundle
        else:
            for key in ("classes", "label", "file_id", "user_id"):
                if not np.array_equal(base[key], bundle[key]):
                    raise ValueError(f"{path} differs in {key}")
        names.append(name.strip())
        bundles.append(bundle)
    return names, bundles


def align_folds(file_ids: np.ndarray, folds_file: Path) -> np.ndarray:
    folds = pd.read_csv(folds_file, usecols=["file_id", "fold"])
    mapping = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    return np.array([mapping[int(file_id)] for file_id in file_ids], dtype=int)


def build_meta_features(
    bundles: list[dict[str, np.ndarray]],
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    *,
    include_log: bool,
    include_aggregate: bool,
    include_seq_context: bool,
    neighbor_shifts: list[int],
    neighbor_rolling: list[int],
) -> np.ndarray:
    probas = np.stack([bundle["proba"] for bundle in bundles], axis=0)
    base_meta = probas.transpose(1, 0, 2).reshape(len(file_ids), -1)
    pieces = [base_meta]
    if include_log:
        pieces.append(np.log(np.clip(probas, 1e-8, 1.0)).transpose(1, 0, 2).reshape(len(file_ids), -1))
    if include_aggregate:
        mean = probas.mean(axis=0)
        std = probas.std(axis=0)
        maxv = probas.max(axis=0)
        minv = probas.min(axis=0)
        margin = np.partition(probas, -2, axis=2)[:, :, -1] - np.partition(probas, -2, axis=2)[:, :, -2]
        entropy = -(probas * np.log(np.clip(probas, 1e-8, 1.0))).sum(axis=2).T
        pieces.extend([mean, std, maxv, minv, margin.T, entropy])
    if include_seq_context:
        frame = pd.DataFrame({"file_id": file_ids, "user_id": user_ids})
        frame = frame.sort_values(["user_id", "file_id"]).reset_index().sort_values("index")
        sizes = frame.groupby("user_id")["file_id"].transform("size").astype(float)
        pos = frame.groupby("user_id").cumcount().astype(float) / (sizes - 1).clip(lower=1.0)
        seq = np.column_stack(
            [
                pos.to_numpy(),
                (1.0 - pos).to_numpy(),
                np.minimum(pos, 1.0 - pos).to_numpy(),
                np.sin(2 * np.pi * pos.to_numpy()),
                np.cos(2 * np.pi * pos.to_numpy()),
                sizes.to_numpy(),
            ]
        )
        pieces.append(seq)
    if neighbor_shifts or neighbor_rolling:
        pieces.extend(make_neighbor_features(base_meta, file_ids, user_ids, neighbor_shifts, neighbor_rolling))
    x = np.concatenate(pieces, axis=1).astype(np.float32)
    return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def make_neighbor_features(
    values: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    shifts: list[int],
    rolling_windows: list[int],
) -> list[np.ndarray]:
    frame = pd.DataFrame({"row": np.arange(len(file_ids)), "file_id": file_ids, "user_id": user_ids})
    out = []
    for shift in shifts:
        shifted = np.zeros_like(values)
        present = np.zeros((len(values), 1), dtype=np.float32)
        for _, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False):
            rows = group["row"].to_numpy(dtype=int)
            src = np.arange(len(rows)) + shift
            ok = (src >= 0) & (src < len(rows))
            shifted[rows[ok]] = values[rows[src[ok]]]
            present[rows[ok], 0] = 1.0
        out.append(shifted)
        out.append(present)
        out.append(shifted - values)
    for window in rolling_windows:
        if window <= 1:
            continue
        rolled = np.zeros_like(values)
        for _, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False):
            rows = group["row"].to_numpy(dtype=int)
            for col in range(values.shape[1]):
                rolled[rows, col] = (
                    pd.Series(values[rows, col]).rolling(window, center=True, min_periods=1).mean().to_numpy()
                )
        out.append(rolled)
        out.append(values - rolled)
    return out


def load_feature_cache(path: Path, context: str, file_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    frame = add_context_features(frame, context)
    frame["_order"] = pd.Categorical(frame["file_id"].astype(int), categories=file_ids.astype(int), ordered=True)
    frame = frame.sort_values("_order")
    if not np.array_equal(frame["file_id"].astype(int).to_numpy(), file_ids.astype(int)):
        raise ValueError(f"{path} does not align to OOF file ids")
    features = [col for col in frame.columns if col not in META_COLS and col != "_order"]
    x = frame[features].to_numpy(dtype=np.float32)
    return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def make_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logreg_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            ),
        ),
        "logreg_l2_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=seed + 1,
            ),
        ),
        "hgb": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=0.08,
                random_state=seed + 2,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=800,
                max_features=0.65,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed + 3,
                n_jobs=-1,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=800,
                max_features=0.65,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed + 4,
                n_jobs=-1,
            ),
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lgbm_small"] = LGBMClassifier(
            objective="multiclass",
            num_class=6,
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + 5,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception as exc:
        print(f"LightGBM unavailable: {exc}", flush=True)
    return models


def run_fold_fair_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    folds: np.ndarray,
    sample_weight_mode: str,
):
    out = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold_scores = []
    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        m = clone(model)
        sample_weight = compute_sample_weight("balanced", y[tr]) if sample_weight_mode == "balanced" else None
        fit_with_optional_weight(m, x[tr], y[tr], sample_weight)
        proba = aligned_proba(m, x[va], classes)
        out[va] = proba
        pred = classes[np.argmax(proba, axis=1)]
        score = float(f1_score(y[va], pred, average="macro"))
        fold_scores.append(score)
        print(f"fold {fold}: macro-F1={score:.6f}", flush=True)
    return out, fold_scores


def fit_with_optional_weight(model, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    if hasattr(model, "steps") and model.steps:
        last_step = model.steps[-1][0]
        try:
            model.fit(x, y, **{f"{last_step}__sample_weight": sample_weight})
            return
        except TypeError:
            pass
    try:
        model.fit(x, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(x, y)


def aligned_proba(model, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    elif hasattr(model, "named_steps"):
        model_classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
    else:
        model_classes = classes
    out = np.zeros((len(x), len(classes)), dtype=float)
    for idx, cls in enumerate(model_classes.astype(int)):
        out[:, np.where(classes == cls)[0][0]] = raw[:, idx]
    return out


if __name__ == "__main__":
    main()
