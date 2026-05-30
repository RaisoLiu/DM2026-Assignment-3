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
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.modeling import predict_from_weighted_proba, tune_probability_class_weights
from evaluate_sequence_smoothing import estimate_transition_model, sequence_indices, viterbi_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-fair Viterbi smoothing with normalized-position class priors.")
    parser.add_argument("--input", required=True, help="name:path/to/oof.npz")
    parser.add_argument("--folds-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-grid", default="0.1,0.3,1.0,3.0")
    parser.add_argument("--beta-grid", default="0,0.08,0.12,0.18,0.27,0.40")
    parser.add_argument("--gamma-grid", default="0,0.05,0.10,0.15,0.22,0.30,0.42,0.60,0.85,1.20")
    parser.add_argument("--bins-grid", default="6,8,10,12,16,20")
    parser.add_argument("--prior-alpha-grid", default="0.5,1.0,2.0,5.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name, payload = load_named_npz(args.input)
    folds = align_folds(payload["file_id"], args.folds_file)
    alpha_grid = parse_float_grid(args.alpha_grid)
    beta_grid = parse_float_grid(args.beta_grid)
    gamma_grid = parse_float_grid(args.gamma_grid)
    bins_grid = tuple(int(v) for v in parse_float_grid(args.bins_grid))
    prior_alpha_grid = parse_float_grid(args.prior_alpha_grid)
    result = evaluate_one(
        name,
        payload,
        folds,
        args.output_dir,
        alpha_grid,
        beta_grid,
        gamma_grid,
        bins_grid,
        prior_alpha_grid,
    )
    out = pd.DataFrame([result])
    out.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(out.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(out.to_string(index=False), flush=True)


def load_named_npz(spec: str) -> tuple[str, dict[str, np.ndarray]]:
    if ":" not in spec:
        path = Path(spec)
        name = path.stem
    else:
        name, raw_path = spec.split(":", 1)
        path = Path(raw_path)
    data = np.load(path, allow_pickle=True)
    return name, {
        "proba": data["proba"].astype(float),
        "classes": data["classes"].astype(int),
        "label": data["label"].astype(int),
        "file_id": data["file_id"].astype(int),
        "user_id": data["user_id"].astype(str),
    }


def parse_float_grid(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def align_folds(file_ids: np.ndarray, folds_file: Path) -> np.ndarray:
    folds = pd.read_csv(folds_file)
    mapping = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    return np.array([mapping[int(fid)] for fid in file_ids], dtype=int)


def evaluate_one(
    name: str,
    payload: dict[str, np.ndarray],
    folds: np.ndarray,
    output_dir: Path,
    alpha_grid: tuple[float, ...],
    beta_grid: tuple[float, ...],
    gamma_grid: tuple[float, ...],
    bins_grid: tuple[int, ...],
    prior_alpha_grid: tuple[float, ...],
) -> dict[str, object]:
    proba = payload["proba"]
    y = payload["label"]
    classes = payload["classes"]
    file_ids = payload["file_id"]
    user_ids = payload["user_id"]
    pos = normalized_positions(file_ids, user_ids)

    base_pred = classes[np.argmax(proba, axis=1)]
    base_score = float(f1_score(y, base_pred, average="macro"))
    fair_pred = np.empty_like(y)
    smooth_pred = np.empty_like(y)
    fold_params = {}
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        tuned = tune_probability_class_weights(proba[train_mask], y[train_mask], classes, n_passes=8)
        weights = tuned["weights"]
        fair_pred[valid_mask] = predict_from_weighted_proba(proba[valid_mask], classes, weights)
        best_params = tune_params(
            proba=proba[train_mask],
            y=y[train_mask],
            classes=classes,
            file_ids=file_ids[train_mask],
            user_ids=user_ids[train_mask],
            pos=pos[train_mask],
            class_weights=weights,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            gamma_grid=gamma_grid,
            bins_grid=bins_grid,
            prior_alpha_grid=prior_alpha_grid,
        )
        transition, start = estimate_transition_model(
            y[train_mask],
            classes,
            file_ids[train_mask],
            user_ids[train_mask],
            alpha=best_params["alpha"],
        )
        prior = estimate_position_prior(
            y[train_mask],
            pos[train_mask],
            classes,
            n_bins=best_params["bins"],
            prior_alpha=best_params["prior_alpha"],
        )
        smooth_pred[valid_mask] = viterbi_predict_with_position(
            proba[valid_mask],
            classes,
            file_ids[valid_mask],
            user_ids[valid_mask],
            pos[valid_mask],
            weights,
            transition,
            start,
            prior,
            beta=best_params["beta"],
            gamma=best_params["gamma"],
        )
        fold_params[str(int(fold))] = {key: float(value) for key, value in best_params.items()}

    fair_score = float(f1_score(y, fair_pred, average="macro"))
    smooth_score = float(f1_score(y, smooth_pred, average="macro"))
    pd.DataFrame(classification_report(y, smooth_pred, output_dict=True, zero_division=0)).T.to_csv(
        output_dir / f"{name}_position_viterbi_report.csv"
    )
    np.savez_compressed(
        output_dir / f"{name}_position_viterbi_predictions.npz",
        pred=smooth_pred,
        fair_pred=fair_pred,
        label=y,
        file_id=file_ids,
        user_id=user_ids,
        position=pos,
    )
    return {
        "name": name,
        "base_macro_f1": base_score,
        "fold_fair_class_bias_macro_f1": fair_score,
        "position_viterbi_macro_f1": smooth_score,
        "fold_params": fold_params,
    }


def tune_params(
    proba: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    pos: np.ndarray,
    class_weights: np.ndarray,
    alpha_grid: tuple[float, ...],
    beta_grid: tuple[float, ...],
    gamma_grid: tuple[float, ...],
    bins_grid: tuple[int, ...],
    prior_alpha_grid: tuple[float, ...],
) -> dict[str, float]:
    best = {
        "macro_f1": -np.inf,
        "alpha": alpha_grid[0],
        "beta": 0.0,
        "gamma": 0.0,
        "bins": bins_grid[0],
        "prior_alpha": prior_alpha_grid[0],
    }
    for alpha in alpha_grid:
        transition, start = estimate_transition_model(y, classes, file_ids, user_ids, alpha=alpha)
        for n_bins in bins_grid:
            for prior_alpha in prior_alpha_grid:
                prior = estimate_position_prior(y, pos, classes, n_bins=n_bins, prior_alpha=prior_alpha)
                for beta in beta_grid:
                    for gamma in gamma_grid:
                        pred = viterbi_predict_with_position(
                            proba, classes, file_ids, user_ids, pos, class_weights, transition, start, prior, beta, gamma
                        )
                        score = float(f1_score(y, pred, average="macro"))
                        if score > best["macro_f1"] + 1e-12:
                            best = {
                                "macro_f1": score,
                                "alpha": float(alpha),
                                "beta": float(beta),
                                "gamma": float(gamma),
                                "bins": int(n_bins),
                                "prior_alpha": float(prior_alpha),
                            }
    return best


def normalized_positions(file_ids: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    out = np.empty(len(file_ids), dtype=float)
    for _, idx in sequence_indices(file_ids, user_ids):
        order = np.argsort(file_ids[idx])
        sorted_idx = idx[order]
        denom = max(len(sorted_idx) - 1, 1)
        out[sorted_idx] = np.arange(len(sorted_idx), dtype=float) / denom
    return out


def estimate_position_prior(
    y: np.ndarray,
    pos: np.ndarray,
    classes: np.ndarray,
    n_bins: int,
    prior_alpha: float,
) -> np.ndarray:
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    counts = np.full((n_bins, len(classes)), prior_alpha, dtype=float)
    bins = np.minimum((pos * n_bins).astype(int), n_bins - 1)
    for bin_idx, label in zip(bins, y, strict=True):
        counts[bin_idx, class_to_idx[int(label)]] += 1.0
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def viterbi_predict_with_position(
    proba: np.ndarray,
    classes: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    pos: np.ndarray,
    class_weights: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    position_prior: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    pred = np.empty(len(file_ids), dtype=int)
    log_transition = np.log(np.clip(transition, 1e-12, 1.0))
    log_start = np.log(np.clip(start, 1e-12, 1.0))
    weighted = np.clip(proba * class_weights.reshape(1, -1), 1e-12, None)
    log_emission = np.log(weighted)
    bins = np.minimum((pos * len(position_prior)).astype(int), len(position_prior) - 1)
    log_position = np.log(np.clip(position_prior[bins], 1e-12, 1.0))
    log_emission = log_emission + gamma * log_position
    for _, idx in sequence_indices(file_ids, user_ids):
        path = viterbi_path(log_emission[idx], log_start, log_transition, beta=beta)
        pred[idx] = classes[path]
    return pred


if __name__ == "__main__":
    main()
