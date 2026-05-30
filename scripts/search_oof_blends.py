#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.modeling import tune_probability_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search soft blends from saved OOF probability files.")
    parser.add_argument("--input", action="append", required=True, help="Named input as name=path/to/oof.npz. Repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--top-base", type=int, default=500, help="Calibrate only the top N blends by uncalibrated macro-F1.")
    parser.add_argument("--calibration-passes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named = [parse_named_input(item) for item in args.input]
    names, paths = zip(*named, strict=True)
    probas, y, classes = load_aligned(paths)

    units = round(1.0 / args.step)
    if not np.isclose(units * args.step, 1.0):
        raise ValueError(f"--step must divide 1.0 exactly, got {args.step}")
    raw_rows = []
    for parts in compositions(int(units), len(names)):
        weights = np.array(parts, dtype=float) / units
        proba = np.tensordot(weights, probas, axes=(0, 0))
        pred = classes[np.argmax(proba, axis=1)]
        raw_rows.append((float(f1_score(y, pred, average="macro")), weights))
    raw_rows.sort(key=lambda item: item[0], reverse=True)

    calibrated_rows = []
    for base, weights in raw_rows[: args.top_base]:
        proba = np.tensordot(weights, probas, axes=(0, 0))
        tuned = tune_probability_class_weights(proba, y, classes, n_passes=args.calibration_passes)
        pred = tuned["pred"]
        counts = np.bincount(pred.astype(int), minlength=len(classes))
        calibrated_rows.append(
            {
                "models": ",".join(names),
                "weights": ",".join(f"{value:.6g}" for value in weights),
                "base_macro_f1": base,
                "calibrated_macro_f1": float(tuned["macro_f1"]),
                "class_weights": ",".join(f"{value:.6g}" for value in tuned["weights"]),
                "pred_counts": ",".join(str(int(value)) for value in counts),
            }
        )

    out = pd.DataFrame(calibrated_rows).sort_values("calibrated_macro_f1", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.head(20).to_string(index=False))
    print(f"Wrote {args.output} rows={len(out)}")


def parse_named_input(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Invalid input {text!r}; expected name=path")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid input {text!r}; empty name")
    return name, Path(path)


def load_aligned(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundles = [np.load(path, allow_pickle=True) for path in paths]
    base_file_id = bundles[0]["file_id"]
    base_label = bundles[0]["label"].astype(int)
    base_classes = bundles[0]["classes"].astype(int)
    probas = []
    for path, bundle in zip(paths, bundles, strict=True):
        if not np.array_equal(base_file_id, bundle["file_id"]):
            raise ValueError(f"file_id order differs in {path}")
        if not np.array_equal(base_label, bundle["label"].astype(int)):
            raise ValueError(f"label differs in {path}")
        if not np.array_equal(base_classes, bundle["classes"].astype(int)):
            raise ValueError(f"classes differ in {path}")
        probas.append(bundle["proba"])
    return np.stack(probas, axis=0), base_label, base_classes


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for rest in compositions(total - value, parts - 1):
            yield (value, *rest)


if __name__ == "__main__":
    main()
