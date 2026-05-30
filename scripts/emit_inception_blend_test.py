#!/usr/bin/env python3
"""Emit the test-time CSV for the InceptionTime + centered-meta blend.

Reads:
  - The new OOF blend npz (provides lambda, alpha, beta, class_weights)
  - The existing centered-meta test proba (artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz)
  - The InceptionTime test proba (averaged across seeds if available)

Writes:
  - submissions/submission_inception_blend_viterbi.csv
  - artifacts/blend_search/test_blend_centered_meta_with_inception.{npz,json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission
from evaluate_sequence_smoothing import (
    estimate_transition_model,
    viterbi_predict_by_user,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit InceptionTime + centered-meta blend test CSV.")
    parser.add_argument(
        "--new-blend-oof",
        type=Path,
        default=Path("artifacts/inception_blend/avg_v2/oof_blend_centered_meta_with_inception.npz"),
    )
    parser.add_argument(
        "--centered-meta-test",
        type=Path,
        default=Path("artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz"),
    )
    parser.add_argument(
        "--inception-test",
        type=Path,
        nargs="+",
        default=[Path("artifacts/inception_full/m60_test_seed2026/test_proba_seed2026.npz")],
        help="One or more InceptionTime test proba npz files; averaged if multiple.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--submission-csv",
        type=Path,
        default=Path("submissions/submission_inception_blend_viterbi.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/blend_search/test_blend_centered_meta_with_inception.json"),
    )
    parser.add_argument(
        "--save-test-proba",
        type=Path,
        default=Path("artifacts/blend_search/test_blend_centered_meta_with_inception.npz"),
    )
    parser.add_argument("--description", default="inception_blend_viterbi_redo")
    return parser.parse_args()


def merge_submission(data_dir: Path, file_ids: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    sample = load_sample_submission(data_dir)
    df = pd.DataFrame({"Id": file_ids.astype(int), "Label": predictions.astype(int)})
    df = df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    if df["Label"].isna().any():
        raise ValueError("Some test rows did not receive predictions")
    df["Label"] = df["Label"].astype(int)
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    args.submission_csv.parent.mkdir(parents=True, exist_ok=True)

    new = np.load(args.new_blend_oof, allow_pickle=True)
    classes = new["classes"].astype(int)
    class_weights = new["class_weights"].astype(float)
    lam = float(new["lambda_inception"]) if "lambda_inception" in new.files else 0.12
    alpha = float(new["alpha"]) if "alpha" in new.files else 1.0
    beta = float(new["beta"]) if "beta" in new.files else 0.08

    cm_test = np.load(args.centered_meta_test, allow_pickle=True)
    cm_proba = cm_test["proba"].astype(float)
    cm_file_ids = cm_test["file_id"].astype(int)
    cm_users = cm_test["user_id"].astype(str)
    print(f"Centered-meta test: {len(cm_file_ids)} rows", flush=True)

    inc_arrays = []
    for path in args.inception_test:
        d = np.load(path, allow_pickle=True)
        inc_index = {int(f): i for i, f in enumerate(d["file_id"])}
        order = np.array([inc_index[int(f)] for f in cm_file_ids], dtype=int)
        inc_arrays.append(d["proba"][order].astype(float))
        print(f"  Loaded inception test from {path}: shape {d['proba'].shape}", flush=True)
    inc_proba = np.mean(np.stack(inc_arrays, axis=0), axis=0) if inc_arrays else np.zeros_like(cm_proba)

    blended = (1.0 - lam) * cm_proba + lam * inc_proba
    print(f"Blended test proba: lambda={lam}, alpha={alpha}, beta={beta}", flush=True)

    # Use the OOF labels to estimate transition matrix on training data
    transition, start = estimate_transition_model(
        y=new["label"].astype(int),
        classes=classes,
        file_ids=new["file_id"].astype(int),
        user_ids=new["user_id"].astype(str),
        alpha=alpha,
    )
    pred = viterbi_predict_by_user(
        proba=blended,
        classes=classes,
        file_ids=cm_file_ids,
        user_ids=cm_users,
        class_weights=class_weights,
        transition=transition,
        start=start,
        beta=beta,
        stay_bonus=0.0,
    )

    submission = merge_submission(args.data_dir, cm_file_ids, pred)
    submission.to_csv(args.submission_csv, index=False)
    digest = sha256_file(args.submission_csv)
    label_counts = {int(k): int(v) for k, v in submission["Label"].value_counts().sort_index().items()}

    metadata = {
        "source_oof_blend": str(args.new_blend_oof),
        "centered_meta_test": str(args.centered_meta_test),
        "inception_test": [str(p) for p in args.inception_test],
        "lambda_inception": lam,
        "alpha": alpha,
        "beta": beta,
        "class_weights": class_weights.tolist(),
        "label_counts": label_counts,
        "sha256": digest,
        "submission": str(args.submission_csv),
        "description": args.description,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2))

    args.save_test_proba.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.save_test_proba,
        proba=blended,
        classes=classes,
        file_id=cm_file_ids,
        user_id=cm_users,
        pred=pred,
        cm_proba=cm_proba,
        inc_proba=inc_proba,
        class_weights=class_weights,
        lambda_inception=lam,
        alpha=alpha,
        beta=beta,
    )

    print(f"Wrote {args.submission_csv}: rows={len(submission)} sha256={digest}")
    print(f"label_counts={label_counts}")


if __name__ == "__main__":
    main()
