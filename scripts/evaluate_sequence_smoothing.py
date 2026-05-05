#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.modeling import predict_from_weighted_proba, tune_probability_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-fold-only sequence smoothing on saved OOF probabilities.")
    parser.add_argument("--folds-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequence_smoothing"))
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Named OOF npz in the form name:path/to/oof.npz. Can be repeated.",
    )
    parser.add_argument(
        "--ensemble-weights",
        default=None,
        help="Optional comma-separated weights for the repeated inputs. If omitted, uses equal weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.folds_file)

    loaded = [load_named_npz(spec) for spec in args.input]
    rows = [evaluate_one(name, payload, folds, args.output_dir) for name, payload in loaded]
    if len(loaded) > 1:
        ensemble = make_ensemble_payload(loaded, args.ensemble_weights)
        rows.append(evaluate_one("ensemble", ensemble, folds, args.output_dir))

    summary = pd.DataFrame(rows).sort_values("viterbi_macro_f1", ascending=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def load_named_npz(spec: str) -> tuple[str, dict[str, np.ndarray]]:
    if ":" not in spec:
        path = Path(spec)
        name = path.stem
    else:
        name, raw_path = spec.split(":", 1)
        path = Path(raw_path)
    data = np.load(path, allow_pickle=True)
    payload = {
        "proba": data["proba"].astype(float),
        "classes": data["classes"].astype(int),
        "label": data["label"].astype(int),
        "file_id": data["file_id"].astype(int),
        "user_id": data["user_id"].astype(str),
    }
    return name, payload


def make_ensemble_payload(loaded: list[tuple[str, dict[str, np.ndarray]]], raw_weights: str | None) -> dict[str, np.ndarray]:
    first = loaded[0][1]
    weights = parse_weights(raw_weights, len(loaded))
    proba = np.zeros_like(first["proba"], dtype=float)
    for weight, (_, payload) in zip(weights, loaded, strict=True):
        assert_compatible(first, payload)
        proba += weight * payload["proba"]
    return {
        "proba": proba,
        "classes": first["classes"],
        "label": first["label"],
        "file_id": first["file_id"],
        "user_id": first["user_id"],
    }


def parse_weights(raw_weights: str | None, n_items: int) -> np.ndarray:
    if raw_weights is None:
        return np.full(n_items, 1.0 / n_items)
    weights = np.array([float(v.strip()) for v in raw_weights.split(",") if v.strip()], dtype=float)
    if len(weights) != n_items:
        raise ValueError(f"Expected {n_items} ensemble weights, got {len(weights)}")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Ensemble weights must be non-negative and sum to a positive value")
    return weights / weights.sum()


def assert_compatible(first: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> None:
    for key in ("classes", "label", "file_id", "user_id"):
        if not np.array_equal(first[key], other[key]):
            raise ValueError(f"Cannot ensemble inputs with different {key}")


def evaluate_one(name: str, payload: dict[str, np.ndarray], folds: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    proba = payload["proba"]
    y = payload["label"]
    classes = payload["classes"]
    file_ids = payload["file_id"]
    user_ids = payload["user_id"]
    fold_ids = align_folds(file_ids, folds)

    base_pred = classes[np.argmax(proba, axis=1)]
    base_score = float(f1_score(y, base_pred, average="macro"))

    fair_pred = np.empty_like(y)
    viterbi_pred = np.empty_like(y)
    fold_params = {}
    for fold in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold

        tuned = tune_probability_class_weights(proba[train_mask], y[train_mask], classes, n_passes=6)
        weights = tuned["weights"]
        fair_pred[valid_mask] = predict_from_weighted_proba(proba[valid_mask], classes, weights)

        best_params = tune_viterbi_params(
            proba=proba[train_mask],
            y=y[train_mask],
            classes=classes,
            file_ids=file_ids[train_mask],
            user_ids=user_ids[train_mask],
            class_weights=weights,
        )
        transition, start = estimate_transition_model(
            y=y[train_mask],
            classes=classes,
            file_ids=file_ids[train_mask],
            user_ids=user_ids[train_mask],
            alpha=best_params["alpha"],
        )
        viterbi_pred[valid_mask] = viterbi_predict_by_user(
            proba=proba[valid_mask],
            classes=classes,
            file_ids=file_ids[valid_mask],
            user_ids=user_ids[valid_mask],
            class_weights=weights,
            transition=transition,
            start=start,
            beta=best_params["beta"],
        )
        fold_params[str(int(fold))] = {
            "class_weights": [float(v) for v in weights],
            "alpha": float(best_params["alpha"]),
            "beta": float(best_params["beta"]),
            "train_macro_f1": float(best_params["train_macro_f1"]),
        }

    fair_score = float(f1_score(y, fair_pred, average="macro"))
    viterbi_score = float(f1_score(y, viterbi_pred, average="macro"))

    pd.DataFrame(classification_report(y, viterbi_pred, output_dict=True, zero_division=0)).T.to_csv(
        output_dir / f"{name}_viterbi_report.csv"
    )
    np.savez_compressed(
        output_dir / f"{name}_viterbi_predictions.npz",
        pred=viterbi_pred,
        fair_pred=fair_pred,
        label=y,
        file_id=file_ids,
        user_id=user_ids,
    )
    result = {
        "name": name,
        "base_macro_f1": base_score,
        "fold_fair_class_bias_macro_f1": fair_score,
        "viterbi_macro_f1": viterbi_score,
        "fold_params": fold_params,
    }
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def align_folds(file_ids: np.ndarray, folds: pd.DataFrame) -> np.ndarray:
    mapping = dict(zip(folds["file_id"], folds["fold"]))
    return np.array([mapping[int(fid)] for fid in file_ids], dtype=int)


def tune_viterbi_params(
    proba: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    class_weights: np.ndarray,
) -> dict[str, float]:
    best = {"macro_f1": -np.inf, "alpha": 1.0, "beta": 0.0}
    alpha_grid = (0.1, 0.3, 1.0, 3.0)
    beta_grid = (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30)
    for alpha in alpha_grid:
        transition, start = estimate_transition_model(y, classes, file_ids, user_ids, alpha=alpha)
        for beta in beta_grid:
            pred = viterbi_predict_by_user(proba, classes, file_ids, user_ids, class_weights, transition, start, beta=beta)
            score = float(f1_score(y, pred, average="macro"))
            if score > best["macro_f1"] + 1e-10:
                best = {"macro_f1": score, "alpha": float(alpha), "beta": float(beta)}
    return {"alpha": best["alpha"], "beta": best["beta"], "train_macro_f1": best["macro_f1"]}


def estimate_transition_model(
    y: np.ndarray,
    classes: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    n_classes = len(classes)
    transition = np.full((n_classes, n_classes), alpha, dtype=float)
    start = np.full(n_classes, alpha, dtype=float)
    for _, idx in sequence_indices(file_ids, user_ids):
        if len(idx) == 0:
            continue
        labels = y[idx]
        start[class_to_idx[int(labels[0])]] += 1.0
        for prev, cur in zip(labels[:-1], labels[1:], strict=False):
            transition[class_to_idx[int(prev)], class_to_idx[int(cur)]] += 1.0
    transition /= transition.sum(axis=1, keepdims=True)
    start /= start.sum()
    return transition, start


def viterbi_predict_by_user(
    proba: np.ndarray,
    classes: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    class_weights: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    beta: float,
) -> np.ndarray:
    pred = np.empty(len(file_ids), dtype=int)
    log_transition = np.log(np.clip(transition, 1e-12, 1.0))
    log_start = np.log(np.clip(start, 1e-12, 1.0))
    weighted = np.clip(proba * class_weights.reshape(1, -1), 1e-12, None)
    log_emission = np.log(weighted)
    for _, idx in sequence_indices(file_ids, user_ids):
        path = viterbi_path(log_emission[idx], log_start, log_transition, beta=beta)
        pred[idx] = classes[path]
    return pred


def sequence_indices(file_ids: np.ndarray, user_ids: np.ndarray):
    frame = pd.DataFrame({"row": np.arange(len(file_ids)), "file_id": file_ids, "user_id": user_ids})
    for user_id, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False):
        yield user_id, group["row"].to_numpy(dtype=int)


def viterbi_path(log_emission: np.ndarray, log_start: np.ndarray, log_transition: np.ndarray, beta: float) -> np.ndarray:
    n_steps, n_classes = log_emission.shape
    scores = np.empty((n_steps, n_classes), dtype=float)
    back = np.zeros((n_steps, n_classes), dtype=np.int16)
    scores[0] = log_start + log_emission[0]
    transition_score = beta * log_transition
    for t in range(1, n_steps):
        candidates = scores[t - 1][:, None] + transition_score
        back[t] = np.argmax(candidates, axis=0)
        scores[t] = candidates[back[t], np.arange(n_classes)] + log_emission[t]
    path = np.empty(n_steps, dtype=np.int16)
    path[-1] = int(np.argmax(scores[-1]))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


if __name__ == "__main__":
    main()
