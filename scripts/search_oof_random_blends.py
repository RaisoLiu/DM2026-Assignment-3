#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.modeling import tune_probability_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random/local search soft blends from saved OOF probability files.")
    parser.add_argument("--input", action="append", required=True, help="name=path/to/oof.npz; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--top-base", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-passes", type=int, default=8)
    parser.add_argument(
        "--center",
        default=None,
        help="Optional comma-separated name=value center weights. Local perturbations are sampled around this center.",
    )
    parser.add_argument("--local-frac", type=float, default=0.65)
    parser.add_argument("--local-scale", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names, probas, y, classes = load_inputs(args.input)
    rng = np.random.default_rng(args.seed)
    center = parse_center(args.center, names) if args.center else None
    rows = []
    raw_best: list[tuple[float, np.ndarray]] = []

    def add_candidate(weights: np.ndarray) -> None:
        weights = normalize(weights)
        proba = np.tensordot(weights, probas, axes=(0, 0))
        pred = classes[np.argmax(proba, axis=1)]
        score = float(f1_score(y, pred, average="macro"))
        raw_best.append((score, weights.copy()))

    if center is not None:
        add_candidate(center)
        for keep in (0.70, 0.80, 0.90, 0.95):
            for idx in range(len(names)):
                w = center * keep
                w[idx] += 1.0 - keep
                add_candidate(w)

    n_local = int(args.trials * args.local_frac) if center is not None else 0
    n_global = args.trials - n_local
    alpha_bank = [
        np.full(len(names), 0.12),
        np.full(len(names), 0.35),
        np.full(len(names), 1.0),
        np.full(len(names), 3.0),
    ]
    for trial in range(n_global):
        alpha = alpha_bank[trial % len(alpha_bank)]
        add_candidate(rng.dirichlet(alpha))
    if center is not None:
        for _ in range(n_local):
            noise = rng.normal(0.0, args.local_scale, size=len(names))
            add_candidate(np.clip(center + noise, 0.0, None))

    raw_best.sort(key=lambda item: item[0], reverse=True)
    deduped = []
    seen = set()
    for base_score, weights in raw_best:
        key = tuple(np.round(weights, 5))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((base_score, weights))
        if len(deduped) >= args.top_base:
            break

    for rank, (base_score, weights) in enumerate(deduped, start=1):
        proba = np.tensordot(weights, probas, axes=(0, 0))
        tuned = tune_probability_class_weights(proba, y, classes, n_passes=args.calibration_passes)
        pred = tuned["pred"].astype(int)
        rows.append(
            {
                "rank_base": rank,
                "base_macro_f1": base_score,
                "calibrated_macro_f1": float(tuned["macro_f1"]),
                "weights": ",".join(f"{value:.8g}" for value in weights),
                "named_weights": ",".join(f"{name}={value:.8g}" for name, value in zip(names, weights, strict=True)),
                "class_weights": ",".join(f"{value:.8g}" for value in tuned["weights"]),
                "pred_counts": ",".join(str(int(v)) for v in np.bincount(pred, minlength=len(classes))),
            }
        )
        if rank % 100 == 0:
            best = max(rows, key=lambda row: row["calibrated_macro_f1"])
            print(
                f"calibrated {rank}/{len(deduped)}; current_best={best['calibrated_macro_f1']:.6f}",
                flush=True,
            )

    out = pd.DataFrame(rows).sort_values("calibrated_macro_f1", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    metadata = {"names": names, "trials": args.trials, "top_base": args.top_base, "seed": args.seed}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out.head(20).to_string(index=False), flush=True)
    print(f"Wrote {args.output} rows={len(out)}", flush=True)


def load_inputs(specs: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = []
    bundles = []
    base = None
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid input {spec!r}; expected name=path")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        data = np.load(path, allow_pickle=True)
        bundle = {
            "proba": data["proba"].astype(float),
            "classes": data["classes"].astype(int),
            "label": data["label"].astype(int),
            "file_id": data["file_id"].astype(int),
            "user_id": data["user_id"].astype(str),
        }
        if base is None:
            base = bundle
        else:
            for key in ("classes", "label", "file_id", "user_id"):
                if not np.array_equal(base[key], bundle[key]):
                    raise ValueError(f"{path} differs in {key}")
        names.append(name.strip())
        bundles.append(bundle)
    return names, np.stack([b["proba"] for b in bundles], axis=0), base["label"], base["classes"]


def parse_center(text: str, names: list[str]) -> np.ndarray:
    values = {name: 0.0 for name in names}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in values:
            raise ValueError(f"Center references unknown input {name!r}; available={names}")
        values[name] = float(value)
    return normalize(np.array([values[name] for name in names], dtype=float))


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


if __name__ == "__main__":
    main()
