#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from evaluate_aeon_rocket import align_scores, make_representation, softmax


SIGNAL_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full MiniRocket and blend with saved tabular test probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sequence-cache", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    parser.add_argument("--base-proba", type=Path, default=Path("artifacts/blend_search/test_blend_round2_oofbest.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-proba", type=Path, default=None)
    parser.add_argument("--n-kernels", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--representation", choices=["raw", "augmented"], default="augmented")
    parser.add_argument(
        "--model-weights",
        default="lgbm_leaves63=0.05,xgb_base=0.25,catboost_base=0.15,minirocket=0.55",
    )
    parser.add_argument("--class-weights", default="1.30744,0.937342,1.1937,0.84554,0.975248,0.828961")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cache = np.load(args.sequence_cache, allow_pickle=True)
    x_train = make_representation(train_cache["x"].astype(np.float32), args.representation)
    y = train_cache["y"].astype(int)
    classes = np.array(sorted(np.unique(y)), dtype=int)
    x_test, test_file_ids = load_test_sequences(args.data_dir)
    x_test = make_representation(x_test, args.representation)
    print(f"MiniRocket full train: train={x_train.shape}; test={x_test.shape}", flush=True)

    transformer = MiniRocket(n_kernels=args.n_kernels, n_jobs=args.n_jobs, random_state=args.seed)
    x_train_transform = transformer.fit_transform(x_train)
    x_test_transform = transformer.transform(x_test)
    print(f"Transformed: train={x_train_transform.shape}; test={x_test_transform.shape}", flush=True)
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"),
    )
    classifier.fit(x_train_transform, y)
    raw_scores = classifier.decision_function(x_test_transform)
    if raw_scores.ndim == 1:
        raw_scores = np.column_stack([-raw_scores, raw_scores])
    minirocket_proba = softmax(align_scores(raw_scores, classifier.named_steps["ridgeclassifiercv"].classes_, classes))

    base = np.load(args.base_proba, allow_pickle=True)
    if not np.array_equal(base["classes"].astype(int), classes):
        raise ValueError("Class order differs between MiniRocket and base probabilities")
    if not np.array_equal(base["file_id"].astype(int), test_file_ids):
        raise ValueError("Test file order differs between MiniRocket and base probabilities")
    model_weights = parse_model_weights(args.model_weights)
    class_weights = parse_class_weights(args.class_weights, len(classes))
    blended = np.zeros_like(minirocket_proba, dtype=float)
    for name, weight in model_weights.items():
        if name == "minirocket":
            proba = minirocket_proba
        else:
            key = f"{name}_proba"
            if key not in base.files:
                raise ValueError(f"{key} missing from {args.base_proba}")
            proba = base[key]
        blended += weight * proba

    pred = classes[np.argmax(blended * class_weights.reshape(1, -1), axis=1)]
    submission = merge_submission(args.data_dir, test_file_ids, pred)
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
            minirocket_proba=minirocket_proba,
            classes=classes,
            file_id=test_file_ids,
            model_names=np.array(list(model_weights.keys()), dtype=object),
            model_weights=np.array(list(model_weights.values()), dtype=float),
            class_weights=class_weights,
        )
        print(f"Saved probabilities to {args.save_proba}", flush=True)


def load_test_sequences(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    file_ids = []
    for csv_path in sorted((data_dir / "test").rglob("*.csv")):
        df = pd.read_csv(csv_path).sort_values("index")
        arr = df[SIGNAL_COLS].interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        if arr.shape[0] != 300:
            arr = resample_to_300(arr)
        rows.append(arr.T)
        file_ids.append(int(df["file_id"].iloc[0]) if "file_id" in df.columns else int(csv_path.stem))
    return np.stack(rows).astype(np.float32), np.array(file_ids, dtype=int)


def resample_to_300(arr: np.ndarray) -> np.ndarray:
    old = np.linspace(0, 1, len(arr))
    new = np.linspace(0, 1, 300)
    return np.vstack([np.interp(new, old, arr[:, i]) for i in range(arr.shape[1])]).T.astype(np.float32)


def parse_model_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        name, value = part.split("=", 1)
        out[name.strip()] = float(value)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Model weights must sum to a positive value")
    return {name: value / total for name, value in out.items()}


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
