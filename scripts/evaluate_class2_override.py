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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from explore_models import META_COLS, add_context_features


@dataclass
class OverrideParams:
    candidate_mode: str
    fallback_mode: str
    set_threshold: float
    keep_threshold: float
    train_macro_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-fair target-class specialist override on top of a saved Viterbi prediction.")
    parser.add_argument("--blend-npz", type=Path, default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"))
    parser.add_argument(
        "--viterbi-npz",
        type=Path,
        default=Path("artifacts/sequence_smoothing_centered_meta_round2_best/centered_meta_viterbi_predictions.npz"),
    )
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--context", default="position_shift")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/class2_override_centered_meta"))
    parser.add_argument("--model", choices=["lgbm", "lgbm_aggressive", "extra_trees"], default="lgbm")
    parser.add_argument("--target-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--positive-weight-factor", type=float, default=1.0)
    parser.add_argument(
        "--feature-set",
        choices=["proba", "proba_context"],
        default="proba_context",
        help="Use only probability-derived features, or concatenate engineered context features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blend = np.load(args.blend_npz, allow_pickle=True)
    vit = np.load(args.viterbi_npz, allow_pickle=True)
    classes = blend["classes"].astype(int)
    y = blend["label"].astype(int)
    folds = blend["fold"].astype(int)
    file_ids = blend["file_id"].astype(int)
    users = blend["user_id"].astype(str)
    base_pred = vit["pred"].astype(int)
    fair_pred = vit["fair_pred"].astype(int) if "fair_pred" in vit.files else base_pred.copy()
    if not np.array_equal(y, vit["label"].astype(int)):
        raise ValueError("blend and Viterbi labels differ")
    if not np.array_equal(file_ids, vit["file_id"].astype(int)):
        raise ValueError("blend and Viterbi file_id order differs")

    target_class = int(args.target_class)
    if target_class not in set(classes.tolist()):
        raise ValueError(f"target class {target_class} is not in classes {classes.tolist()}")

    proba_features = make_probability_features(blend, base_pred, fair_pred, target_class)
    if args.feature_set == "proba_context":
        context = load_context_features(args.feature_cache, args.context, file_ids)
        X = pd.concat([proba_features, context], axis=1)
    else:
        X = proba_features
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fallback = make_fallback_predictions(blend, fair_pred, classes, target_class)
    y_bin = (y == target_class).astype(int)
    print(
        f"model={args.model}; feature_set={args.feature_set}; X={X.shape}; "
        f"target={target_class}; positive={int(y_bin.sum())}/{len(y_bin)}; "
        f"base_macro={f1_score(y, base_pred, average='macro'):.6f}",
        flush=True,
    )

    oof_prob_target = np.zeros(len(y), dtype=float)
    override_pred = np.empty_like(y)
    fold_rows = []
    fold_params: dict[str, dict[str, object]] = {}
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = make_model(args.model, args.seed + int(fold), args.n_jobs)

        inner_prob = make_inner_oof_prob2(
            model=model,
            X=X.loc[train_mask],
            y_bin=y_bin[train_mask],
            groups=users[train_mask],
            seed=args.seed + 100 + int(fold),
            n_splits=args.inner_splits,
            positive_weight_factor=args.positive_weight_factor,
        )
        params = tune_override_params(
            y=y[train_mask],
            base_pred=base_pred[train_mask],
            fair_pred=fair_pred[train_mask],
            fallback=fallback[train_mask],
            prob2=inner_prob,
            classes=classes,
            target_class=target_class,
        )

        fitted = clone(model)
        fit_binary_model(fitted, X.loc[train_mask], y_bin[train_mask], args.positive_weight_factor)
        valid_prob = predict_positive_proba(fitted, X.loc[valid_mask])
        oof_prob_target[valid_mask] = valid_prob
        pred = apply_override(
            base_pred=base_pred[valid_mask],
            fair_pred=fair_pred[valid_mask],
            fallback=fallback[valid_mask],
            prob2=valid_prob,
            params=params,
            target_class=target_class,
        )
        override_pred[valid_mask] = pred
        fold_score = float(f1_score(y[valid_mask], pred, average="macro"))
        base_fold = float(f1_score(y[valid_mask], base_pred[valid_mask], average="macro"))
        fold_rows.append(
            {
                "fold": int(fold),
                "base_macro_f1": base_fold,
                "override_macro_f1": fold_score,
                "candidate_mode": params.candidate_mode,
                "fallback_mode": params.fallback_mode,
                "set_threshold": params.set_threshold,
                "keep_threshold": params.keep_threshold,
                "inner_train_macro_f1": params.train_macro_f1,
            }
        )
        fold_params[str(int(fold))] = fold_rows[-1]
        print(
            f"fold {fold}: base={base_fold:.6f} override={fold_score:.6f} "
            f"mode={params.candidate_mode}/{params.fallback_mode} "
            f"set={params.set_threshold:.4f} keep={params.keep_threshold:.4f}",
            flush=True,
        )

    base_score = float(f1_score(y, base_pred, average="macro"))
    override_score = float(f1_score(y, override_pred, average="macro"))
    report = classification_report(y, override_pred, digits=4, zero_division=0, output_dict=True)
    print("\nbase_macro_f1", base_score, flush=True)
    print("override_macro_f1", override_score, flush=True)
    print(classification_report(y, override_pred, digits=4, zero_division=0), flush=True)

    stem = f"class{target_class}_override_{args.model}_{args.feature_set}_{args.context}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        prob2=oof_prob_target,
        pred=override_pred,
        base_pred=base_pred,
        fair_pred=fair_pred,
        fallback=fallback,
        label=y,
        file_id=file_ids,
        user_id=users,
        fold=folds,
        classes=classes,
        target_class=target_class,
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / f"{stem}_folds.csv", index=False)
    pd.DataFrame(report).T.to_csv(args.output_dir / f"{stem}_report.csv")
    payload = {
        "model": args.model,
        "target_class": target_class,
        "feature_set": args.feature_set,
        "context": args.context,
        "positive_weight_factor": args.positive_weight_factor,
        "base_macro_f1": base_score,
        "override_macro_f1": override_score,
        "fold_params": fold_params,
    }
    (args.output_dir / f"{stem}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(args.output_dir / f"{stem}_metrics.csv", index=False)


def make_probability_features(
    blend: np.lib.npyio.NpzFile,
    base_pred: np.ndarray,
    fair_pred: np.ndarray,
    target_class: int,
) -> pd.DataFrame:
    proba = blend["proba"].astype(float)
    classes = blend["classes"].astype(int)
    users = blend["user_id"].astype(str)
    frame: dict[str, np.ndarray] = {}
    add_proba_block(frame, "blend", proba, classes)
    for key in blend.files:
        if key.endswith("_proba") and blend[key].ndim == 2:
            add_proba_block(frame, key[:-6], blend[key].astype(float), classes)
    class_weights = blend["class_weights"].astype(float) if "class_weights" in blend.files else np.ones(len(classes), dtype=float)
    weighted = proba * class_weights.reshape(1, -1)
    target_idx = int(np.where(classes == target_class)[0][0])
    non_target = [i for i in range(len(classes)) if i != target_idx]
    frame[f"blend_p{target_class}_minus_max_non_target"] = proba[:, target_idx] - proba[:, non_target].max(axis=1)
    frame[f"blend_weighted_p{target_class}_minus_max_non_target"] = (
        weighted[:, target_idx] - weighted[:, non_target].max(axis=1)
    )
    frame[f"blend_p{target_class}_rank"] = 1 + np.sum(proba > proba[:, [target_idx]], axis=1)
    for label_name, pred in (("viterbi", base_pred), ("fair", fair_pred)):
        for cls in classes:
            frame[f"{label_name}_pred_is_{int(cls)}"] = (pred == cls).astype(float)
    df = pd.DataFrame(frame)
    return add_user_probability_context(df, users)


def add_proba_block(out: dict[str, np.ndarray], prefix: str, proba: np.ndarray, classes: np.ndarray) -> None:
    clipped = np.clip(proba, 1e-8, 1.0)
    for idx, cls in enumerate(classes):
        out[f"{prefix}_p{int(cls)}"] = proba[:, idx]
        out[f"{prefix}_logp{int(cls)}"] = np.log(clipped[:, idx])


def add_user_probability_context(df: pd.DataFrame, users: np.ndarray) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}
    work = df.copy()
    work["_user"] = users
    base_cols = [c for c in df.columns if c.startswith("blend_p") or c.startswith("blend_weighted")]
    grouped = work.groupby("_user", sort=False)
    for col in base_cols:
        series = work[col]
        prev1 = grouped[col].shift(1)
        next1 = grouped[col].shift(-1)
        additions[f"{col}_prev1"] = prev1.fillna(series)
        additions[f"{col}_next1"] = next1.fillna(series)
        additions[f"{col}_prev1_delta"] = (series - prev1).fillna(0.0)
        additions[f"{col}_next1_delta"] = (next1 - series).fillna(0.0)
        for window in (3, 5):
            roll = grouped[col].transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
            additions[f"{col}_roll{window}"] = roll
            additions[f"{col}_roll{window}_delta"] = series - roll
    if not additions:
        return df
    return pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)


def load_context_features(path: Path, context: str, file_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = add_context_features(frame, context)
    frame = frame.set_index(frame["file_id"].astype(int), drop=False)
    aligned = frame.loc[file_ids].reset_index(drop=True)
    cols = [c for c in aligned.columns if c not in META_COLS]
    out = aligned[cols].copy()
    out.columns = [f"ctx_{c}" for c in out.columns]
    return out


def make_fallback_predictions(
    blend: np.lib.npyio.NpzFile,
    fair_pred: np.ndarray,
    classes: np.ndarray,
    target_class: int,
) -> np.ndarray:
    classes = classes.astype(int)
    class_weights = blend["class_weights"].astype(float) if "class_weights" in blend.files else np.ones(len(classes), dtype=float)
    weighted = blend["proba"].astype(float) * class_weights.reshape(1, -1)
    target_idx = int(np.where(classes == target_class)[0][0])
    weighted[:, target_idx] = -np.inf
    non_target_argmax = classes[np.argmax(weighted, axis=1)]
    return np.where(fair_pred != target_class, fair_pred, non_target_argmax)


def make_model(name: str, seed: int, n_jobs: int):
    if name in {"lgbm", "lgbm_aggressive"}:
        from lightgbm import LGBMClassifier

        if name == "lgbm_aggressive":
            return LGBMClassifier(
                objective="binary",
                n_estimators=850,
                learning_rate=0.018,
                num_leaves=47,
                min_child_samples=5,
                subsample=0.9,
                colsample_bytree=0.78,
                reg_alpha=0.02,
                reg_lambda=0.18,
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=-1,
            )
        return LGBMClassifier(
            objective="binary",
            n_estimators=650,
            learning_rate=0.024,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.88,
            colsample_bytree=0.75,
            reg_alpha=0.04,
            reg_lambda=0.25,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    return ExtraTreesClassifier(
        n_estimators=700,
        max_features=0.35,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=n_jobs,
    )


def make_inner_oof_prob2(
    model,
    X: pd.DataFrame,
    y_bin: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_splits: int,
    positive_weight_factor: float,
) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_bin), dtype=float)
    for inner_fold, (tr, va) in enumerate(cv.split(X, y_bin, groups), start=1):
        fitted = clone(model)
        fit_binary_model(fitted, X.iloc[tr], y_bin[tr], positive_weight_factor)
        oof[va] = predict_positive_proba(fitted, X.iloc[va])
        score = f1_score(y_bin[va], (oof[va] >= 0.5).astype(int), zero_division=0)
        print(f"  inner fold {inner_fold}: binary_f1@0.5={score:.4f}", flush=True)
    return oof


def fit_binary_model(model, X: pd.DataFrame, y_bin: np.ndarray, positive_weight_factor: float) -> None:
    sample_weight = compute_sample_weight("balanced", y_bin)
    if positive_weight_factor != 1.0:
        sample_weight = sample_weight * np.where(y_bin == 1, positive_weight_factor, 1.0)
    model.fit(X, y_bin, sample_weight=sample_weight)


def predict_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return raw[:, int(np.where(classes == 1)[0][0])]


def tune_override_params(
    y: np.ndarray,
    base_pred: np.ndarray,
    fair_pred: np.ndarray,
    fallback: np.ndarray,
    prob2: np.ndarray,
    classes: np.ndarray,
    target_class: int,
) -> OverrideParams:
    base_score = float(f1_score(y, base_pred, average="macro"))
    best = OverrideParams("none", "fallback", 1.01, 0.0, base_score)
    set_thresholds = make_threshold_grid(prob2, lower=0.01, upper=0.99)
    keep_thresholds = np.unique(np.r_[0.0, np.linspace(0.02, 0.70, 35), np.quantile(prob2, np.linspace(0.05, 0.80, 28))])
    for candidate_mode in ("active_confusions", "active_confusions_or_fair", "nonzero", "all_non_target"):
        for fallback_mode in ("fallback", "fair", "weighted_non_target"):
            for set_threshold in set_thresholds:
                for keep_threshold in keep_thresholds:
                    pred = apply_override(
                        base_pred=base_pred,
                        fair_pred=fair_pred,
                        fallback=fallback,
                        prob2=prob2,
                        params=OverrideParams(candidate_mode, fallback_mode, float(set_threshold), float(keep_threshold), 0.0),
                        target_class=target_class,
                    )
                    score = float(f1_score(y, pred, average="macro"))
                    if score > best.train_macro_f1 + 1e-12:
                        best = OverrideParams(
                            candidate_mode=candidate_mode,
                            fallback_mode=fallback_mode,
                            set_threshold=float(set_threshold),
                            keep_threshold=float(keep_threshold),
                            train_macro_f1=score,
                        )
    return best


def make_threshold_grid(prob: np.ndarray, lower: float, upper: float) -> np.ndarray:
    linear = np.linspace(lower, upper, 80)
    quantiles = np.quantile(prob, np.linspace(0.50, 0.995, 80))
    return np.unique(np.clip(np.r_[linear, quantiles], lower, upper))


def apply_override(
    base_pred: np.ndarray,
    fair_pred: np.ndarray,
    fallback: np.ndarray,
    prob2: np.ndarray,
    params: OverrideParams,
    target_class: int,
) -> np.ndarray:
    pred = base_pred.copy()
    if params.candidate_mode == "none":
        return pred
    active_confusions = [cls for cls in (1, 2, 3, 5) if cls != target_class]
    if params.candidate_mode == "active_confusions":
        candidate = np.isin(base_pred, active_confusions)
    elif params.candidate_mode == "active_confusions_or_fair":
        candidate = np.isin(base_pred, active_confusions) | (fair_pred == target_class)
    elif params.candidate_mode == "nonzero":
        candidate = np.isin(base_pred, [1, 2, 3, 4, 5])
    else:
        candidate = base_pred != target_class
    pred[candidate & (prob2 >= params.set_threshold)] = target_class

    demote = (base_pred == target_class) & (prob2 < params.keep_threshold)
    if params.fallback_mode == "fair":
        replacement = np.where(fair_pred != target_class, fair_pred, fallback)
    else:
        replacement = fallback
    pred[demote] = replacement[demote]
    return pred


if __name__ == "__main__":
    main()
