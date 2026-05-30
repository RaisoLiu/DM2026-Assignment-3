#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.modeling import tune_probability_class_weights
from evaluate_aeon_rocket import make_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate aeon convolution classifier OOF probabilities.")
    parser.add_argument("--sequence-cache", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aeon_classifier_oof"))
    parser.add_argument(
        "--method",
        choices=[
            "arsenal",
            "drcif",
            "hydra",
            "multirocket_hydra",
            "mrseql",
            "mrsqm",
            "rist",
            "rdst",
            "rise",
            "rocket_classifier",
            "minirocket_classifier",
            "multirocket_classifier",
            "quant",
        ],
        required=True,
    )
    parser.add_argument("--representation", choices=["raw", "augmented"], default="augmented")
    parser.add_argument("--n-kernels", type=int, default=2000)
    parser.add_argument("--n-estimators", type=int, default=15)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--jitter-eps", type=float, default=0.0, help="Add deterministic tiny noise for estimators that reject constant channels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.sequence_cache, allow_pickle=True)
    x = make_representation(data["x"].astype(np.float32), args.representation)
    if args.method == "rdst":
        x = np.ascontiguousarray(x.astype(np.float64))
    if args.jitter_eps > 0:
        rng = np.random.default_rng(args.seed)
        x = x + rng.normal(0.0, args.jitter_eps, size=x.shape).astype(np.float32)
    y = data["y"].astype(int)
    users = data["users"].astype(str)
    file_ids = data["file_ids"].astype(int)
    classes = np.array(sorted(np.unique(y)), dtype=int)
    folds = make_folds(x, y, users, args.n_splits, args.seed)
    fold_values = sorted(np.unique(folds))
    if args.max_folds:
        fold_values = fold_values[: args.max_folds]
    print(
        f"method={args.method}; representation={args.representation}; x={x.shape}; classes={classes.tolist()}",
        flush=True,
    )
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold_scores = []
    for fold in fold_values:
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        model = make_classifier(args, seed=args.seed + int(fold))
        print(f"fold {fold}: fit train={len(tr)} valid={len(va)}", flush=True)
        model.fit(x[tr], y[tr])
        raw = model.predict_proba(x[va])
        proba[va] = align_proba(raw, model.classes_, classes)
        pred = classes[np.argmax(proba[va], axis=1)]
        score = float(f1_score(y[va], pred, average="macro"))
        fold_scores.append(score)
        print(f"fold {fold}: macro-F1={score:.6f}", flush=True)

    covered = np.isin(folds, fold_values)
    base_pred = classes[np.argmax(proba[covered], axis=1)]
    base_score = float(f1_score(y[covered], base_pred, average="macro"))
    tuned = tune_probability_class_weights(proba[covered], y[covered], classes, n_passes=8)
    cal_score = float(tuned["macro_f1"])
    cal_pred = tuned["pred"]
    print("\nbase_macro_f1", base_score, flush=True)
    print("calibrated_macro_f1", cal_score, flush=True)
    print(classification_report(y[covered], cal_pred, digits=4, zero_division=0), flush=True)
    stem = f"oof_{args.method}_{args.representation}_k{args.n_kernels}"
    if args.method == "arsenal":
        stem += f"_e{args.n_estimators}"
    if args.max_folds:
        stem += f"_folds{args.max_folds}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        proba=proba,
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
        "n_estimators": args.n_estimators,
        "base_macro_f1": base_score,
        "calibrated_macro_f1": cal_score,
        "fold_scores": fold_scores,
        "class_weights": [float(v) for v in tuned["weights"]],
        "covered": int(covered.sum()),
    }
    (args.output_dir / f"{stem}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(args.output_dir / f"{stem}_metrics.csv", index=False)


def make_classifier(args: argparse.Namespace, seed: int):
    if args.method == "arsenal":
        return Arsenal(
            n_kernels=args.n_kernels,
            n_estimators=args.n_estimators,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.method == "drcif":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(
            n_estimators=args.n_estimators,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if args.method == "hydra":
        return HydraClassifier(
            n_kernels=args.n_kernels,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.method == "multirocket_hydra":
        return MultiRocketHydraClassifier(
            n_kernels=args.n_kernels,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.method == "mrsqm":
        from aeon.classification.dictionary_based import MrSQMClassifier

        return MrSQMClassifier(
            features_per_rep=max(50, args.n_kernels),
            selection_per_rep=max(200, args.n_kernels * 4),
            random_state=seed,
        )
    if args.method == "mrseql":
        from aeon.classification.dictionary_based import MrSEQLClassifier

        return MrSEQLClassifier()
    if args.method == "rist":
        from aeon.classification.hybrid import RISTClassifier

        return RISTClassifier(n_jobs=args.n_jobs, random_state=seed)
    if args.method == "rdst":
        from aeon.classification.shapelet_based import RDSTClassifier

        return RDSTClassifier(
            max_shapelets=args.n_kernels,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.method == "rise":
        from aeon.classification.interval_based import RandomIntervalSpectralEnsembleClassifier

        return RandomIntervalSpectralEnsembleClassifier(
            n_estimators=args.n_estimators,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if args.method == "rocket_classifier":
        return RocketClassifier(
            n_kernels=args.n_kernels,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.method == "minirocket_classifier":
        return MiniRocketClassifier(
            n_kernels=args.n_kernels,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.method == "quant":
        from aeon.classification.interval_based import QUANTClassifier

        return QUANTClassifier(
            class_weight="balanced",
            random_state=seed,
        )
    return MultiRocketClassifier(
        n_kernels=args.n_kernels,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=seed,
    )


def make_folds(x: np.ndarray, y: np.ndarray, users: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(x)), y, users), start=1):
        folds[valid_idx] = fold
    if np.any(folds < 0):
        raise RuntimeError("Some rows were not assigned to a fold")
    return folds


def align_proba(raw: np.ndarray, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(raw), len(classes)), dtype=np.float32)
    for idx, cls in enumerate(model_classes.astype(int)):
        out[:, np.where(classes == cls)[0][0]] = raw[:, idx]
    return out


if __name__ == "__main__":
    main()
