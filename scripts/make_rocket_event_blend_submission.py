#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from evaluate_aeon_rocket import align_scores, make_representation, softmax
from explore_models import META_COLS, add_context_features, aligned_proba, fit_model, make_models
from make_minirocket_blend_submission import load_test_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OOF 0.761016 rocket/event blend submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sequence-cache", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    parser.add_argument("--base-proba", type=Path, default=Path("artifacts/blend_search/test_blend_round2_oofbest.npz"))
    parser.add_argument("--mini10-proba", type=Path, default=Path("artifacts/blend_search/test_blend_minirocket_oof07597.npz"))
    parser.add_argument("--train-event-cache", type=Path, default=Path("artifacts/features_event/train_features_event.csv"))
    parser.add_argument("--test-event-cache", type=Path, default=Path("artifacts/features_event/test_features_event.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-proba", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--model-weights",
        default=(
            "xgb=0.168907,cat=0.168173,xgb_d6=0.0480888,"
            "mini10=0.453043,mini20=0.0359402,miniraw=0.0292555,"
            "multi=0.0646225,event_lgbm=0.0144224,event_xgb=0.0175467"
        ),
    )
    parser.add_argument("--class-weights", default="0.963819,0.877599,1.23896,0.877599,1.23896,0.877599")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_weights = parse_model_weights(args.model_weights)
    base = np.load(args.base_proba, allow_pickle=True)
    mini10_bundle = np.load(args.mini10_proba, allow_pickle=True)
    classes = base["classes"].astype(int)
    file_id = base["file_id"].astype(int)
    class_weights = parse_class_weights(args.class_weights, len(classes))
    if not np.array_equal(mini10_bundle["file_id"].astype(int), file_id):
        raise ValueError("MiniRocket 10k test probabilities do not match base test file order")

    train_seq = np.load(args.sequence_cache, allow_pickle=True)
    x_train_raw = train_seq["x"].astype(np.float32)
    y = train_seq["y"].astype(int)
    if not np.array_equal(np.array(sorted(np.unique(y)), dtype=int), classes):
        raise ValueError("Class order differs between sequence cache and base probabilities")
    x_test_raw, test_file_id = load_test_sequences(args.data_dir)
    if not np.array_equal(test_file_id, file_id):
        raise ValueError("Test sequence file order differs from base probabilities")

    proba_parts: dict[str, np.ndarray] = {
        "xgb": base["xgb_base_proba"],
        "cat": base["catboost_base_proba"],
        "xgb_d6": base["xgb_depth6_proba"],
        "mini10": mini10_bundle["minirocket_proba"],
    }
    proba_parts["mini20"] = rocket_proba(
        x_train_raw,
        x_test_raw,
        y,
        classes,
        method="minirocket",
        representation="augmented",
        n_kernels=20000,
        n_jobs=args.n_jobs,
        seed=args.seed + 20000,
    )
    proba_parts["miniraw"] = rocket_proba(
        x_train_raw,
        x_test_raw,
        y,
        classes,
        method="minirocket",
        representation="raw",
        n_kernels=10000,
        n_jobs=args.n_jobs,
        seed=args.seed + 100,
    )
    proba_parts["multi"] = rocket_proba(
        x_train_raw,
        x_test_raw,
        y,
        classes,
        method="multirocket",
        representation="augmented",
        n_kernels=2500,
        n_jobs=args.n_jobs,
        seed=args.seed + 2500,
    )
    event_probas = event_model_probas(args.train_event_cache, args.test_event_cache, y, classes, args.seed)
    proba_parts.update(event_probas)

    blended = np.zeros((len(file_id), len(classes)), dtype=float)
    for name, weight in model_weights.items():
        if name not in proba_parts:
            raise ValueError(f"Missing probability part {name!r}; available={sorted(proba_parts)}")
        blended += weight * proba_parts[name]
    pred = classes[np.argmax(blended * class_weights.reshape(1, -1), axis=1)]
    submission = merge_submission(args.data_dir, file_id, pred)
    validate_submission(args.data_dir, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    digest = sha256_file(args.output)
    counts = submission["Label"].value_counts().sort_index().to_dict()
    print(f"Wrote {args.output} rows={len(submission)} sha256={digest}", flush=True)
    print(f"label_counts={counts}", flush=True)

    if args.save_proba is not None:
        args.save_proba.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_proba,
            proba=blended,
            classes=classes,
            file_id=file_id,
            model_names=np.array(list(model_weights.keys()), dtype=object),
            model_weights=np.array(list(model_weights.values()), dtype=float),
            class_weights=class_weights,
            **{f"{name}_proba": proba for name, proba in proba_parts.items()},
        )
        print(f"Saved probabilities to {args.save_proba}", flush=True)


def rocket_proba(
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    method: str,
    representation: str,
    n_kernels: int,
    n_jobs: int,
    seed: int,
) -> np.ndarray:
    x_train = make_representation(x_train_raw, representation)
    x_test = make_representation(x_test_raw, representation)
    if method == "minirocket":
        transformer = MiniRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=seed)
    elif method == "multirocket":
        transformer = MultiRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=seed)
    else:
        raise ValueError(f"Unknown rocket method {method}")
    print(
        f"Training {method} {representation} k={n_kernels}: train={x_train.shape}; test={x_test.shape}",
        flush=True,
    )
    x_train_t = transformer.fit_transform(x_train)
    x_test_t = transformer.transform(x_test)
    print(f"Transformed {method}: train={x_train_t.shape}; test={x_test_t.shape}", flush=True)
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"),
    )
    classifier.fit(x_train_t, y)
    raw_scores = classifier.decision_function(x_test_t)
    if raw_scores.ndim == 1:
        raw_scores = np.column_stack([-raw_scores, raw_scores])
    return softmax(align_scores(raw_scores, classifier.named_steps["ridgeclassifiercv"].classes_, classes))


def event_model_probas(train_path: Path, test_path: Path, y: np.ndarray, classes: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing event feature cache(s): {train_path}, {test_path}")
    train = add_context_features(pd.read_csv(train_path), "position")
    test = add_context_features(pd.read_csv(test_path), "position")
    features = [col for col in train.columns if col not in META_COLS]
    for col in [col for col in features if col not in test.columns]:
        test[col] = 0.0
    x_train = train[features]
    x_test = test[features]
    model_specs = make_models(seed)
    sample_weight = compute_sample_weight("balanced", y)
    out = {}
    for model_name, out_name in (("lgbm_base", "event_lgbm"), ("xgb_base", "event_xgb")):
        print(f"Training {model_name} on event features: train={x_train.shape}; test={x_test.shape}", flush=True)
        model = clone(model_specs[model_name])
        fit_model(model, x_train, y, sample_weight)
        out[out_name] = aligned_proba(model, x_test, classes)
    return out


def parse_model_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Model weights must sum to a positive value")
    return {name: value / total for name, value in weights.items()}


def parse_class_weights(text: str, expected: int) -> np.ndarray:
    weights = np.array([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)
    if len(weights) != expected or np.any(weights <= 0):
        raise ValueError(f"Expected {expected} positive class weights")
    return weights


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
