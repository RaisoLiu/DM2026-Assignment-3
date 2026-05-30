#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import load_sample_submission, normalize_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply class decision weights to saved test probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--proba-npz", type=Path, required=True)
    parser.add_argument("--class-weights", required=True, help="Comma-separated class decision weights.")
    parser.add_argument(
        "--class-weight-scales",
        default="1,1,1,1,1,1",
        help="Comma-separated multiplicative scale per class, applied after --class-weights.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = np.load(args.proba_npz, allow_pickle=True)
    proba = bundle["proba"]
    classes = bundle["classes"].astype(int)
    file_id = bundle["file_id"]
    class_weights = parse_weights(args.class_weights)
    class_scales = parse_weights(args.class_weight_scales)
    if len(class_weights) != len(classes) or len(class_scales) != len(classes):
        raise ValueError(f"Expected {len(classes)} class weights/scales")

    final_weights = class_weights * class_scales
    pred = classes[np.argmax(proba * final_weights.reshape(1, -1), axis=1)]
    submission = merge_submission(args.data_dir, file_id, pred)
    validate_submission(args.data_dir, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} rows={len(submission)} sha256={sha256_file(args.output)}")
    print(f"weights={','.join(f'{v:.6g}' for v in final_weights)}")
    print(f"label_counts={submission['Label'].value_counts().sort_index().to_dict()}")


def parse_weights(text: str) -> np.ndarray:
    weights = np.array([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)
    if weights.size == 0 or np.any(weights <= 0):
        raise ValueError("Weights must be positive")
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
    if submission["Id"].duplicated().any():
        raise ValueError("Duplicate Id values in submission")
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
