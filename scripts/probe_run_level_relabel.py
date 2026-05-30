#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional experiment dependency
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional experiment dependency
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


TARGET_CLASSES = (1, 2, 3, 5)
NON_TWO_TARGETS = (1, 3, 5)
ID_COLUMNS = {"file_id", "user_id", "split", "label"}


@dataclass(frozen=True)
class RunSlice:
    run_id: int
    user_id: str
    fold: int
    row_indices: np.ndarray
    seq_indices: np.ndarray
    pred_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe fold-aware run-level relabeling for current class-2 prediction runs."
    )
    parser.add_argument("--base-npz", type=Path, default=Path("artifacts/local_best_oof_0777285/best_predictions.npz"))
    parser.add_argument("--proba-npz", type=Path, default=Path("artifacts/local_best_oof_0775951/best_predictions.npz"))
    parser.add_argument(
        "--component-npz",
        type=Path,
        default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"),
    )
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/features/train_features_landmark.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run_level_rich_relabel_probe"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k-best", type=int, default=80)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument(
        "--tree-n-jobs",
        type=int,
        default=-1,
        help="n_jobs for ExtraTrees and RandomForest models.",
    )
    parser.add_argument("--hgb-learning-rate", type=float, default=0.035)
    parser.add_argument("--hgb-max-iter", type=int, default=120)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=8)
    parser.add_argument("--hgb-l2", type=float, default=0.15)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=20)
    parser.add_argument(
        "--models",
        default="logreg,hgb",
        help="Comma-separated model names to run. Available: logreg,extra,rf,hgb,lgbm,cat.",
    )
    parser.add_argument(
        "--candidate-labels",
        default="2",
        help="Comma-separated current predicted labels whose runs are candidates for relabeling.",
    )
    parser.add_argument(
        "--binary-targets",
        default="1,3,5",
        help="Comma-separated target labels to probe with one-vs-rest binary run relabeling.",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Use only sequence/probability features and skip the tabular feature cache.",
    )
    parser.add_argument(
        "--cache-aggs",
        default="mean",
        help="Comma-separated cache aggregations. Available: mean,std,min,max,first,last.",
    )
    parser.add_argument(
        "--binary-thresholds",
        default="",
        help="Optional comma-separated binary thresholds. Defaults to the built-in fixed/quantile grid.",
    )
    parser.add_argument(
        "--multiclass-thresholds",
        default="0.34,0.38,0.42,0.46,0.50,0.55,0.60,0.66,0.72,0.80,0.90",
        help="Comma-separated multiclass probability thresholds.",
    )
    parser.add_argument(
        "--multiclass-margins",
        default="-0.10,-0.05,0.0,0.03,0.06,0.10,0.16,0.24",
        help="Comma-separated multiclass probability margins against the reference label.",
    )
    parser.add_argument(
        "--multiclass-reference-label",
        type=int,
        default=2,
        help="Reference class for multiclass margin checks.",
    )
    parser.add_argument(
        "--multiclass-target-labels",
        default="",
        help="Optional comma-separated labels multiclass relabeling may output. Defaults to all classes except reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="Features .* are constant")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection._univariate_selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(args.base_npz, allow_pickle=True)
    proba_npz = np.load(args.proba_npz, allow_pickle=True)
    component_npz = np.load(args.component_npz, allow_pickle=True)

    y = base["label"].astype(int)
    base_pred = base["pred"].astype(int)
    file_ids = base["file_id"].astype(int)
    user_ids = base["user_id"].astype(str)
    folds = base["fold"].astype(int)
    classes = base["classes"].astype(int)
    assert_aligned(base, proba_npz, ("label", "file_id", "user_id", "classes"))
    assert_aligned(base, component_npz, ("label", "file_id", "user_id", "classes"))

    proba_sources = load_probability_sources(proba_npz, component_npz)
    feature_frame = None
    if not args.no_feature_cache:
        feature_frame = load_feature_cache(args.feature_cache, file_ids)

    runs = make_predicted_runs(base_pred, file_ids, user_ids, folds, parse_int_set(args.candidate_labels))
    run_frame, run_feature_names = build_run_features(
        runs=runs,
        y=y,
        base_pred=base_pred,
        file_ids=file_ids,
        user_ids=user_ids,
        folds=folds,
        classes=classes,
        proba_sources=proba_sources,
        feature_frame=feature_frame,
        cache_aggs=parse_cache_aggs(args.cache_aggs),
    )
    run_frame.to_csv(args.output_dir / "run_features.csv", index=False)

    X = run_frame[run_feature_names].to_numpy(dtype=float)
    y_majority = run_frame["true_majority"].to_numpy(dtype=int)
    run_folds = run_frame["fold"].to_numpy(dtype=int)
    models = make_models(
        args.seed,
        args.k_best,
        X.shape[1],
        parse_model_names(args.models),
        args.logreg_c,
        args.tree_n_jobs,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_iter=args.hgb_max_iter,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
        hgb_l2=args.hgb_l2,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
    )
    binary_thresholds = parse_optional_float_tuple(args.binary_thresholds)
    multiclass_thresholds = parse_float_tuple(args.multiclass_thresholds)
    multiclass_margins = parse_float_tuple(args.multiclass_margins)
    multiclass_targets = parse_optional_int_tuple(args.multiclass_target_labels)
    if multiclass_targets is None:
        multiclass_targets = tuple(cls for cls in TARGET_CLASSES if cls != args.multiclass_reference_label)

    rows: list[dict[str, object]] = []
    best_name = ""
    best_pred = base_pred.copy()
    best_score = float(f1_score(y, base_pred, average="macro"))
    best_extra: dict[str, object] = {}

    for model_name, model in models.items():
        result = evaluate_multiclass_strategy(
            model_name,
            model,
            X,
            y_majority,
            run_folds,
            runs,
            y,
            base_pred,
            folds,
            classes=TARGET_CLASSES,
            thresholds=multiclass_thresholds,
            margins=multiclass_margins,
            reference_label=args.multiclass_reference_label,
            target_labels=multiclass_targets,
        )
        rows.append(result.summary_row)
        if result.score > best_score + 1e-12:
            best_score = result.score
            best_pred = result.pred.copy()
            best_name = result.name
            best_extra = result.extra

        for target in parse_int_tuple(args.binary_targets):
            result = evaluate_binary_strategy(
                f"{model_name}_target{target}",
                model,
                target,
                X,
                y_majority,
                run_folds,
                runs,
                y,
                base_pred,
                folds,
                thresholds=binary_thresholds,
            )
            rows.append(result.summary_row)
            if result.score > best_score + 1e-12:
                best_score = result.score
                best_pred = result.pred.copy()
                best_name = result.name
                best_extra = result.extra

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    np.savez_compressed(
        args.output_dir / "best_predictions.npz",
        pred=best_pred,
        base_pred=base_pred,
        label=y,
        file_id=file_ids,
        user_id=user_ids,
        fold=folds,
        classes=classes,
    )
    pd.DataFrame(classification_report(y, best_pred, output_dict=True, zero_division=0)).T.to_csv(
        args.output_dir / "best_report.csv"
    )
    best_payload = {
        "best_name": best_name,
        "macro_f1": float(best_score),
        "base_macro_f1": float(f1_score(y, base_pred, average="macro")),
        "changes": int(np.sum(best_pred != base_pred)),
        **best_extra,
    }
    (args.output_dir / "best_summary.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(best_payload, indent=2), flush=True)


@dataclass(frozen=True)
class StrategyResult:
    name: str
    score: float
    pred: np.ndarray
    summary_row: dict[str, object]
    extra: dict[str, object]


def assert_aligned(first: np.lib.npyio.NpzFile, other: np.lib.npyio.NpzFile, keys: Iterable[str]) -> None:
    for key in keys:
        if not np.array_equal(first[key], other[key]):
            raise ValueError(f"Input arrays are not aligned on {key}")


def load_probability_sources(
    proba_npz: np.lib.npyio.NpzFile, component_npz: np.lib.npyio.NpzFile
) -> dict[str, np.ndarray]:
    sources = {"blend": proba_npz["proba"].astype(float)}
    for name in component_npz["model_names"].astype(str).tolist():
        key = f"{name}_proba"
        if key in component_npz:
            sources[name] = component_npz[key].astype(float)
    return sources


def load_feature_cache(path: Path, file_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "file_id" not in frame.columns:
        raise ValueError(f"{path} must contain file_id")
    frame["file_id"] = frame["file_id"].astype(int)
    frame = frame.set_index("file_id").loc[file_ids].reset_index()
    numeric_cols = [
        col
        for col in frame.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return frame[numeric_cols]


def make_predicted_runs(
    pred: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    folds: np.ndarray,
    candidate_labels: set[int],
) -> list[RunSlice]:
    runs: list[RunSlice] = []
    run_id = 0
    for user_id in sorted(set(user_ids.tolist())):
        idx = np.where(user_ids == user_id)[0]
        order = np.argsort(file_ids[idx])
        seq_rows = idx[order]
        start = 0
        for end in range(1, len(seq_rows) + 1):
            if end == len(seq_rows) or pred[seq_rows[end]] != pred[seq_rows[start]]:
                label = int(pred[seq_rows[start]])
                if label in candidate_labels:
                    rows = seq_rows[start:end]
                    runs.append(
                        RunSlice(
                            run_id=run_id,
                            user_id=str(user_id),
                            fold=int(folds[rows[0]]),
                            row_indices=rows,
                            seq_indices=np.arange(start, end, dtype=int),
                            pred_label=label,
                        )
                    )
                    run_id += 1
                start = end
    return runs


def build_run_features(
    runs: list[RunSlice],
    y: np.ndarray,
    base_pred: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    folds: np.ndarray,
    classes: np.ndarray,
    proba_sources: dict[str, np.ndarray],
    feature_frame: pd.DataFrame | None,
    cache_aggs: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    sorted_by_user = {
        user_id: np.where(user_ids == user_id)[0][np.argsort(file_ids[np.where(user_ids == user_id)[0]])]
        for user_id in sorted(set(user_ids.tolist()))
    }
    rows: list[dict[str, object]] = []
    for run in runs:
        idx = run.row_indices
        seq = sorted_by_user[run.user_id]
        pos_in_seq = np.searchsorted(seq, idx)
        labels = y[idx]
        counts = {int(cls): int(np.sum(labels == int(cls))) for cls in classes}
        true_majority = max(counts, key=counts.get)
        row: dict[str, object] = {
            "run_id": run.run_id,
            "user_id": run.user_id,
            "fold": run.fold,
            "pred_label": run.pred_label,
            "start_file": int(file_ids[idx[0]]),
            "end_file": int(file_ids[idx[-1]]),
            "length": int(len(idx)),
            "start_pos": float(pos_in_seq[0] / max(len(seq) - 1, 1)),
            "end_pos": float(pos_in_seq[-1] / max(len(seq) - 1, 1)),
            "mid_pos": float((pos_in_seq[0] + pos_in_seq[-1]) / (2 * max(len(seq) - 1, 1))),
            "true_majority": int(true_majority),
            "true_purity": float(counts[int(true_majority)] / len(idx)),
        }
        add_neighbor_features(row, seq, pos_in_seq[0], pos_in_seq[-1], base_pred)
        add_window_context_features(row, seq, pos_in_seq[0], pos_in_seq[-1], base_pred, classes)
        for source_name, proba in proba_sources.items():
            add_probability_features(row, source_name, proba, idx, class_to_idx)
            add_probability_context_features(row, source_name, proba, seq, pos_in_seq[0], pos_in_seq[-1], class_to_idx)
        if feature_frame is not None:
            add_cache_features(row, feature_frame, idx, cache_aggs)
        rows.append(row)

    frame = pd.DataFrame(rows)
    feature_names = [
        col
        for col in frame.columns
        if col not in {"run_id", "user_id", "fold", "start_file", "end_file", "true_majority", "true_purity"}
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return frame, feature_names


def add_neighbor_features(row: dict[str, object], seq: np.ndarray, start: int, end: int, pred: np.ndarray) -> None:
    row["prev_label"] = int(pred[seq[start - 1]]) if start > 0 else -1
    row["next_label"] = int(pred[seq[end + 1]]) if end + 1 < len(seq) else -1
    row["prev2_label"] = int(pred[seq[start - 2]]) if start > 1 else -1
    row["next2_label"] = int(pred[seq[end + 2]]) if end + 2 < len(seq) else -1
    row["left_edge"] = int(start == 0)
    row["right_edge"] = int(end + 1 == len(seq))
    row["dist_start"] = int(start)
    row["dist_end"] = int(len(seq) - end - 1)
    row["prev_run_len"] = previous_run_length(seq, start, pred)
    row["next_run_len"] = next_run_length(seq, end, pred)


def previous_run_length(seq: np.ndarray, start: int, pred: np.ndarray) -> int:
    if start == 0:
        return 0
    label = pred[seq[start - 1]]
    cur = start - 1
    while cur >= 0 and pred[seq[cur]] == label:
        cur -= 1
    return start - cur - 1


def next_run_length(seq: np.ndarray, end: int, pred: np.ndarray) -> int:
    if end + 1 >= len(seq):
        return 0
    label = pred[seq[end + 1]]
    cur = end + 1
    while cur < len(seq) and pred[seq[cur]] == label:
        cur += 1
    return cur - end - 1


def add_window_context_features(
    row: dict[str, object],
    seq: np.ndarray,
    start: int,
    end: int,
    pred: np.ndarray,
    classes: np.ndarray,
) -> None:
    for width in (3, 6, 12):
        left = seq[max(0, start - width) : start]
        right = seq[end + 1 : min(len(seq), end + 1 + width)]
        around = np.concatenate([left, seq[start : end + 1], right])
        for prefix, arr in ((f"left{width}", left), (f"right{width}", right), (f"around{width}", around)):
            denom = max(len(arr), 1)
            for cls in classes:
                row[f"{prefix}_pred{int(cls)}_frac"] = float(np.sum(pred[arr] == int(cls)) / denom) if len(arr) else 0.0


def add_probability_features(
    row: dict[str, object],
    source_name: str,
    proba: np.ndarray,
    idx: np.ndarray,
    class_to_idx: dict[int, int],
) -> None:
    values = proba[idx]
    for cls in TARGET_CLASSES:
        col = values[:, class_to_idx[cls]]
        row[f"{source_name}_p{cls}_mean"] = float(np.mean(col))
        row[f"{source_name}_p{cls}_min"] = float(np.min(col))
        row[f"{source_name}_p{cls}_max"] = float(np.max(col))
        row[f"{source_name}_p{cls}_std"] = float(np.std(col))
        row[f"{source_name}_p{cls}_first"] = float(col[0])
        row[f"{source_name}_p{cls}_last"] = float(col[-1])
    mean_values = values.mean(axis=0)
    target_scores = np.array([mean_values[class_to_idx[cls]] for cls in TARGET_CLASSES], dtype=float)
    sorted_scores = np.sort(target_scores)
    row[f"{source_name}_target_top_margin"] = float(sorted_scores[-1] - sorted_scores[-2])
    for cls in NON_TWO_TARGETS:
        row[f"{source_name}_p{cls}_minus_p2"] = float(mean_values[class_to_idx[cls]] - mean_values[class_to_idx[2]])


def add_probability_context_features(
    row: dict[str, object],
    source_name: str,
    proba: np.ndarray,
    seq: np.ndarray,
    start: int,
    end: int,
    class_to_idx: dict[int, int],
) -> None:
    for side, arr in (
        ("prev", seq[max(0, start - 3) : start]),
        ("next", seq[end + 1 : min(len(seq), end + 4)]),
    ):
        if len(arr) == 0:
            for cls in TARGET_CLASSES:
                row[f"{source_name}_{side}_p{cls}_mean"] = 0.0
            continue
        means = proba[arr].mean(axis=0)
        for cls in TARGET_CLASSES:
            row[f"{source_name}_{side}_p{cls}_mean"] = float(means[class_to_idx[cls]])


def add_cache_features(
    row: dict[str, object],
    feature_frame: pd.DataFrame,
    idx: np.ndarray,
    cache_aggs: tuple[str, ...],
) -> None:
    values = feature_frame.iloc[idx]
    if "mean" in cache_aggs:
        for col, value in values.mean(axis=0, numeric_only=True).items():
            row[f"cache_mean_{col}"] = float(value)
    if "std" in cache_aggs:
        stds = values.std(axis=0, numeric_only=True).fillna(0.0)
        for col, value in stds.items():
            row[f"cache_std_{col}"] = float(value)
    if "min" in cache_aggs:
        for col, value in values.min(axis=0, numeric_only=True).items():
            row[f"cache_min_{col}"] = float(value)
    if "max" in cache_aggs:
        for col, value in values.max(axis=0, numeric_only=True).items():
            row[f"cache_max_{col}"] = float(value)
    if "first" in cache_aggs:
        first = values.iloc[0]
        for col, value in first.items():
            row[f"cache_first_{col}"] = float(value)
    if "last" in cache_aggs:
        last = values.iloc[-1]
        for col, value in last.items():
            row[f"cache_last_{col}"] = float(value)


def parse_model_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    available = {"logreg", "extra", "rf", "hgb", "lgbm", "cat"}
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}; available={sorted(available)}")
    if not names:
        raise ValueError("--models must include at least one model")
    return names


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_optional_int_tuple(raw: str) -> tuple[int, ...] | None:
    if not raw.strip():
        return None
    return parse_int_tuple(raw)


def parse_int_set(raw: str) -> set[int]:
    return set(parse_int_tuple(raw))


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_optional_float_tuple(raw: str) -> tuple[float, ...] | None:
    if not raw.strip():
        return None
    return parse_float_tuple(raw)


def parse_cache_aggs(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    available = {"mean", "std", "min", "max", "first", "last"}
    unknown = sorted(set(values) - available)
    if unknown:
        raise ValueError(f"Unknown cache aggregations: {unknown}; available={sorted(available)}")
    if not values:
        raise ValueError("--cache-aggs must include at least one aggregation")
    return values


def make_models(
    seed: int,
    k_best: int,
    n_features: int,
    selected: tuple[str, ...],
    logreg_c: float,
    tree_n_jobs: int,
    hgb_learning_rate: float,
    hgb_max_iter: int,
    hgb_max_leaf_nodes: int,
    hgb_l2: float,
    hgb_min_samples_leaf: int,
) -> dict[str, BaseEstimator]:
    k = min(k_best, n_features)
    models: dict[str, BaseEstimator] = {
        "logreg": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    LogisticRegression(
                        C=logreg_c,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=tree_n_jobs,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=tree_n_jobs,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=hgb_learning_rate,
                        max_iter=hgb_max_iter,
                        max_leaf_nodes=hgb_max_leaf_nodes,
                        l2_regularization=hgb_l2,
                        min_samples_leaf=hgb_min_samples_leaf,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    if "lgbm" in selected:
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for --models lgbm")
        models["lgbm"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.03,
                        num_leaves=7,
                        min_child_samples=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=tree_n_jobs,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if "cat" in selected:
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for --models cat")
        models["cat"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=250,
                        learning_rate=0.03,
                        depth=3,
                        l2_leaf_reg=5.0,
                        auto_class_weights="Balanced",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    return {name: models[name] for name in selected}


def evaluate_binary_strategy(
    name: str,
    model: BaseEstimator,
    target: int,
    X: np.ndarray,
    y_majority: np.ndarray,
    run_folds: np.ndarray,
    runs: list[RunSlice],
    y: np.ndarray,
    base_pred: np.ndarray,
    row_folds: np.ndarray,
    thresholds: tuple[float, ...] | None,
) -> StrategyResult:
    pred = base_pred.copy()
    fold_rows: list[dict[str, object]] = []
    for fold in sorted(np.unique(run_folds)):
        train_runs = run_folds != fold
        valid_runs = run_folds == fold
        inner_prob = inner_binary_oof(model, X[train_runs], y_majority[train_runs] == target, run_folds[train_runs])
        threshold, train_score = tune_binary_threshold(
            target,
            inner_prob,
            np.flatnonzero(train_runs),
            runs,
            y,
            base_pred,
            train_mask=row_folds != fold,
            thresholds=thresholds,
        )
        valid_prob = fit_predict_binary(model, X[train_runs], y_majority[train_runs] == target, X[valid_runs])
        changed = apply_binary_run_rule(pred, np.flatnonzero(valid_runs), runs, valid_prob, target, threshold)
        fold_rows.append(
            {
                "fold": int(fold),
                "threshold": None if threshold is None else float(threshold),
                "train_score": float(train_score),
                "changes": int(changed),
            }
        )
    score = float(f1_score(y, pred, average="macro"))
    return StrategyResult(
        name=name,
        score=score,
        pred=pred,
        summary_row={
            "name": name,
            "strategy": "binary",
            "target": target,
            "macro_f1": score,
            "changes": int(np.sum(pred != base_pred)),
        },
        extra={"strategy": "binary", "target": target, "fold_rules": fold_rows},
    )


def evaluate_multiclass_strategy(
    name: str,
    model: BaseEstimator,
    X: np.ndarray,
    y_majority: np.ndarray,
    run_folds: np.ndarray,
    runs: list[RunSlice],
    y: np.ndarray,
    base_pred: np.ndarray,
    row_folds: np.ndarray,
    classes: tuple[int, ...],
    thresholds: tuple[float, ...],
    margins: tuple[float, ...],
    reference_label: int,
    target_labels: tuple[int, ...],
) -> StrategyResult:
    pred = base_pred.copy()
    fold_rows: list[dict[str, object]] = []
    for fold in sorted(np.unique(run_folds)):
        train_runs = run_folds != fold
        valid_runs = run_folds == fold
        inner_proba = inner_multiclass_oof(model, X[train_runs], y_majority[train_runs], run_folds[train_runs], classes)
        threshold, margin, train_score = tune_multiclass_thresholds(
            inner_proba,
            np.flatnonzero(train_runs),
            runs,
            y,
            base_pred,
            train_mask=row_folds != fold,
            classes=classes,
            thresholds=thresholds,
            margins=margins,
            reference_label=reference_label,
            target_labels=target_labels,
        )
        valid_proba = fit_predict_multiclass(model, X[train_runs], y_majority[train_runs], X[valid_runs], classes)
        changed = apply_multiclass_run_rule(
            pred,
            np.flatnonzero(valid_runs),
            runs,
            valid_proba,
            classes=classes,
            threshold=threshold,
            margin=margin,
            reference_label=reference_label,
            target_labels=target_labels,
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "threshold": None if threshold is None else float(threshold),
                "margin": None if margin is None else float(margin),
                "train_score": float(train_score),
                "changes": int(changed),
            }
        )
    score = float(f1_score(y, pred, average="macro"))
    return StrategyResult(
        name=f"{name}_multiclass",
        score=score,
        pred=pred,
        summary_row={
            "name": f"{name}_multiclass",
            "strategy": "multiclass",
            "target": "multi",
            "macro_f1": score,
            "changes": int(np.sum(pred != base_pred)),
        },
        extra={
            "strategy": "multiclass",
            "reference_label": int(reference_label),
            "target_labels": [int(label) for label in target_labels],
            "fold_rules": fold_rows,
        },
    )


def inner_binary_oof(model: BaseEstimator, X: np.ndarray, y_binary: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prob = np.zeros(len(y_binary), dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        prob[valid] = fit_predict_binary(model, X[train], y_binary[train], X[valid])
    return prob


def fit_predict_binary(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray) -> np.ndarray:
    if len(X_valid) == 0:
        return np.array([], dtype=float)
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_valid), float(np.mean(y_train)), dtype=float)
    fitted = clone(model)
    fitted.fit(X_train, y_train.astype(int))
    proba = fitted.predict_proba(X_valid)
    classes = fitted.classes_.astype(int)
    if 1 not in classes:
        return np.zeros(len(X_valid), dtype=float)
    return proba[:, int(np.where(classes == 1)[0][0])]


def inner_multiclass_oof(
    model: BaseEstimator, X: np.ndarray, y_train: np.ndarray, folds: np.ndarray, classes: tuple[int, ...]
) -> np.ndarray:
    proba = np.zeros((len(y_train), len(classes)), dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        proba[valid] = fit_predict_multiclass(model, X[train], y_train[train], X[valid], classes)
    return proba


def fit_predict_multiclass(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    classes: tuple[int, ...],
) -> np.ndarray:
    if len(X_valid) == 0:
        return np.zeros((0, len(classes)), dtype=float)
    present = np.unique(y_train)
    if len(present) < 2:
        out = np.zeros((len(X_valid), len(classes)), dtype=float)
        if int(present[0]) in classes:
            out[:, classes.index(int(present[0]))] = 1.0
        return out
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    raw = fitted.predict_proba(X_valid)
    out = np.zeros((len(X_valid), len(classes)), dtype=float)
    for j, cls in enumerate(fitted.classes_.astype(int)):
        if int(cls) in classes:
            out[:, classes.index(int(cls))] = raw[:, j]
    return out


def tune_binary_threshold(
    target: int,
    prob: np.ndarray,
    run_indices: np.ndarray,
    runs: list[RunSlice],
    y: np.ndarray,
    base_pred: np.ndarray,
    train_mask: np.ndarray,
    thresholds: tuple[float, ...] | None,
) -> tuple[float | None, float]:
    base_score = float(f1_score(y[train_mask], base_pred[train_mask], average="macro"))
    best_score = base_score
    best_threshold: float | None = None
    if thresholds is None:
        grid = sorted(
            set(np.linspace(0.45, 0.995, 80).tolist() + np.quantile(prob, np.linspace(0.5, 0.98, 25)).tolist())
        )
    else:
        grid = thresholds
    for threshold in grid:
        trial = base_pred.copy()
        apply_binary_run_rule(trial, run_indices, runs, prob, target, float(threshold))
        score = float(f1_score(y[train_mask], trial[train_mask], average="macro"))
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def tune_multiclass_thresholds(
    proba: np.ndarray,
    run_indices: np.ndarray,
    runs: list[RunSlice],
    y: np.ndarray,
    base_pred: np.ndarray,
    train_mask: np.ndarray,
    classes: tuple[int, ...],
    thresholds: tuple[float, ...],
    margins: tuple[float, ...],
    reference_label: int,
    target_labels: tuple[int, ...],
) -> tuple[float | None, float | None, float]:
    base_score = float(f1_score(y[train_mask], base_pred[train_mask], average="macro"))
    best_score = base_score
    best_threshold: float | None = None
    best_margin: float | None = None
    for threshold in thresholds:
        for margin in margins:
            trial = base_pred.copy()
            apply_multiclass_run_rule(
                trial,
                run_indices,
                runs,
                proba,
                classes,
                float(threshold),
                float(margin),
                reference_label,
                target_labels,
            )
            score = float(f1_score(y[train_mask], trial[train_mask], average="macro"))
            if score > best_score + 1e-12:
                best_score = score
                best_threshold = float(threshold)
                best_margin = float(margin)
    return best_threshold, best_margin, best_score


def apply_binary_run_rule(
    pred: np.ndarray,
    run_indices: np.ndarray,
    runs: list[RunSlice],
    prob: np.ndarray,
    target: int,
    threshold: float | None,
) -> int:
    if threshold is None:
        return 0
    changes = 0
    for local_i, run_i in enumerate(run_indices):
        if prob[local_i] >= threshold:
            before = pred[runs[int(run_i)].row_indices].copy()
            pred[runs[int(run_i)].row_indices] = target
            changes += int(np.sum(before != target))
    return changes


def apply_multiclass_run_rule(
    pred: np.ndarray,
    run_indices: np.ndarray,
    runs: list[RunSlice],
    proba: np.ndarray,
    classes: tuple[int, ...],
    threshold: float | None,
    margin: float | None,
    reference_label: int,
    target_labels: tuple[int, ...],
) -> int:
    if threshold is None or margin is None:
        return 0
    reference_idx = classes.index(reference_label)
    target_indices = [classes.index(label) for label in target_labels]
    changes = 0
    for local_i, run_i in enumerate(run_indices):
        row = proba[local_i]
        best_target_idx = max(target_indices, key=lambda idx: row[idx])
        target = int(classes[best_target_idx])
        if row[best_target_idx] >= threshold and row[best_target_idx] - row[reference_idx] >= margin:
            before = pred[runs[int(run_i)].row_indices].copy()
            pred[runs[int(run_i)].row_indices] = target
            changes += int(np.sum(before != target))
    return changes


if __name__ == "__main__":
    main()
