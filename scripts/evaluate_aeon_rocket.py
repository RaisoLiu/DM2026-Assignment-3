#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket, Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.modeling import tune_probability_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate aeon ROCKET-family raw sequence classifiers with grouped OOF.")
    parser.add_argument("--sequence-cache", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rocket_oof"))
    parser.add_argument("--method", choices=["minirocket", "multirocket", "rocket"], default="minirocket")
    parser.add_argument("--representation", choices=["raw", "augmented"], default="augmented")
    parser.add_argument("--n-kernels", type=int, default=10000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-folds", type=int, default=0, help="Debug: limit number of folds; 0 means all.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.sequence_cache, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(int)
    users = data["users"].astype(str)
    file_ids = data["file_ids"].astype(int)
    x = make_representation(x, args.representation)
    classes = np.array(sorted(np.unique(y)), dtype=int)
    folds = make_folds(x, y, users, args.n_splits, args.seed)
    fold_values = sorted(np.unique(folds))
    if args.max_folds:
        fold_values = fold_values[: args.max_folds]

    print(
        f"method={args.method}; representation={args.representation}; x={x.shape}; "
        f"classes={classes.tolist()}; users={len(np.unique(users))}",
        flush=True,
    )
    scores = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold_scores: list[float] = []
    for fold in fold_values:
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        transformer = make_transformer(args.method, args.n_kernels, args.n_jobs, args.seed + int(fold))
        classifier = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10),
                class_weight="balanced",
            ),
        )
        print(f"fold {fold}: transform train={len(tr)} valid={len(va)}", flush=True)
        x_tr = transformer.fit_transform(x[tr])
        x_va = transformer.transform(x[va])
        print(f"fold {fold}: transformed train={x_tr.shape} valid={x_va.shape}", flush=True)
        classifier.fit(x_tr, y[tr])
        raw_scores = classifier.decision_function(x_va)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])
        scores[va] = align_scores(raw_scores, classifier.named_steps["ridgeclassifiercv"].classes_, classes)
        pred = classes[np.argmax(scores[va], axis=1)]
        score = float(f1_score(y[va], pred, average="macro"))
        fold_scores.append(score)
        print(f"fold {fold}: macro-F1={score:.5f}", flush=True)

    covered = np.isin(folds, fold_values)
    proba = softmax(scores)
    base_pred = classes[np.argmax(scores[covered], axis=1)]
    base_score = float(f1_score(y[covered], base_pred, average="macro"))
    tuned = tune_probability_class_weights(proba[covered], y[covered], classes, n_passes=6)
    calibrated_score = float(tuned["macro_f1"])
    calibrated_pred = tuned["pred"]
    print("\nbase_macro_f1", base_score, flush=True)
    print("calibrated_macro_f1", calibrated_score, flush=True)
    print(classification_report(y[covered], calibrated_pred, digits=4, zero_division=0), flush=True)

    stem = f"oof_{args.method}_{args.representation}_k{args.n_kernels}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        proba=proba,
        scores=scores,
        classes=classes,
        label=y,
        fold=folds,
        file_id=file_ids,
        user_id=users,
        covered=covered,
    )
    payload = {
        "method": args.method,
        "representation": args.representation,
        "n_kernels": args.n_kernels,
        "base_macro_f1": base_score,
        "calibrated_macro_f1": calibrated_score,
        "fold_scores": fold_scores,
        "class_weights": [float(v) for v in tuned["weights"]],
    }
    (args.output_dir / f"{stem}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(args.output_dir / f"{stem}_metrics.csv", index=False)


def make_representation(x: np.ndarray, representation: str) -> np.ndarray:
    if representation == "raw":
        return x
    mean = x[:, :3, :]
    std = x[:, 3:6, :]
    mean_vm = np.sqrt(np.sum(mean**2, axis=1, keepdims=True))
    std_vm = np.sqrt(np.sum(std**2, axis=1, keepdims=True))
    gravity_delta = np.abs(mean_vm - 1.0)
    orient = np.divide(mean, mean_vm + 1e-6)
    mean_diff = prepend_zero(np.diff(mean, axis=2))
    std_diff = prepend_zero(np.diff(std, axis=2))
    orient_step = prepend_zero(
        np.arccos(np.clip(np.sum(orient[:, :, 1:] * orient[:, :, :-1], axis=1, keepdims=True), -1.0, 1.0))
    )
    return np.concatenate(
        [
            x,
            mean_vm,
            std_vm,
            gravity_delta,
            orient,
            mean_diff,
            std_diff,
            orient_step,
        ],
        axis=1,
    ).astype(np.float32)


def prepend_zero(values: np.ndarray) -> np.ndarray:
    pad = np.zeros((*values.shape[:-1], 1), dtype=values.dtype)
    return np.concatenate([pad, values], axis=-1)


def make_transformer(method: str, n_kernels: int, n_jobs: int, seed: int):
    if method == "minirocket":
        return MiniRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=seed)
    if method == "multirocket":
        return MultiRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=seed)
    return Rocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=seed)


def make_folds(x: np.ndarray, y: np.ndarray, users: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(x)), y, users), start=1):
        folds[valid_idx] = fold
    if np.any(folds < 0):
        raise RuntimeError("Some rows were not assigned to a fold")
    return folds


def align_scores(raw_scores: np.ndarray, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.full((len(raw_scores), len(classes)), raw_scores.min() - 1.0, dtype=np.float32)
    for idx, cls in enumerate(model_classes.astype(int)):
        aligned[:, np.where(classes == cls)[0][0]] = raw_scores[:, idx]
    return aligned


def softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(centered, -50, 50))
    return exp / exp.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
