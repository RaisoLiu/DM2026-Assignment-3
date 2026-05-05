#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.modeling import tune_probability_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved OOF probabilities rigorously.")
    parser.add_argument("--folds-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/oof_eval"))
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Named OOF npz in the form name:path/to/oof.npz. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.folds_file)
    rows = []
    for spec in args.input:
        name, path = parse_input_spec(spec)
        result = evaluate_one(name, Path(path), folds, args.output_dir)
        rows.append(result)
    results = pd.DataFrame(rows).sort_values("fold_fair_calibrated_macro_f1", ascending=False)
    results.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(results.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(results.to_string(index=False))


def parse_input_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        path = Path(spec)
        return path.stem, spec
    name, path = spec.split(":", 1)
    return name, path


def evaluate_one(name: str, path: Path, folds: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    proba = data["proba"]
    y = data["label"].astype(int)
    classes = data["classes"].astype(int)
    file_ids = data["file_id"]
    fold_ids = align_folds(file_ids, folds)

    base_pred = classes[np.argmax(proba, axis=1)]
    base_score = float(f1_score(y, base_pred, average="macro"))

    global_cal = tune_probability_class_weights(proba, y, classes, n_passes=6)
    global_score = float(global_cal["macro_f1"])

    fair_pred = np.empty_like(y)
    fair_weights = {}
    for fold in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold
        tuned = tune_probability_class_weights(proba[train_mask], y[train_mask], classes, n_passes=6)
        weights = tuned["weights"]
        weighted = proba[valid_mask] * weights.reshape(1, -1)
        fair_pred[valid_mask] = classes[np.argmax(weighted, axis=1)]
        fair_weights[str(int(fold))] = [float(v) for v in weights]
    fair_score = float(f1_score(y, fair_pred, average="macro"))

    report = classification_report(y, fair_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"{name}_fold_fair_report.csv")
    np.savez_compressed(output_dir / f"{name}_fold_fair_predictions.npz", pred=fair_pred, label=y, file_id=file_ids)
    payload = {
        "name": name,
        "path": str(path),
        "base_macro_f1": base_score,
        "global_calibrated_macro_f1": global_score,
        "fold_fair_calibrated_macro_f1": fair_score,
        "global_weights": [float(v) for v in global_cal["weights"]],
        "fold_weights": fair_weights,
    }
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def align_folds(file_ids: np.ndarray, folds: pd.DataFrame) -> np.ndarray:
    mapping = dict(zip(folds["file_id"], folds["fold"]))
    fold_ids = np.array([mapping[int(fid)] for fid in file_ids], dtype=int)
    return fold_ids


if __name__ == "__main__":
    main()

