#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sequence_smoothing import estimate_transition_model, viterbi_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search OOF blend weights under fixed fold-fair Viterbi smoother parameters."
    )
    parser.add_argument("--source-npz", type=Path, default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"))
    parser.add_argument("--folds-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    parser.add_argument(
        "--smoother-summary",
        type=Path,
        default=Path("artifacts/sequence_smoothing_centered_meta_round2_best/summary.csv"),
        help="Summary CSV containing fold_params from a fold-fair Viterbi run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--center", default=None, help="Optional comma-separated name=value center weights.")
    parser.add_argument("--scales", default="0.008,0.014,0.022,0.035")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--first-lock",
        type=int,
        default=0,
        help="Force the first N windows of each user sequence to class 0 after Viterbi decoding.",
    )
    parser.add_argument(
        "--fold-fair-demote-class",
        type=int,
        default=None,
        help="Optionally tune a train-fold-only low-margin demotion rule for this predicted class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = np.load(args.source_npz, allow_pickle=True)
    names = [str(name) for name in source["model_names"]]
    components = np.stack([source[f"{name}_proba"].astype(np.float32) for name in names]).astype(np.float32)
    y = source["label"].astype(int)
    classes = source["classes"].astype(int)
    file_ids = source["file_id"].astype(int)
    user_ids = source["user_id"].astype(str)
    fold_ids = align_folds(file_ids, args.folds_file)
    center = parse_center(args.center, names) if args.center else normalize(source["model_weights"].astype(float))
    scales = tuple(float(part.strip()) for part in args.scales.split(",") if part.strip())
    if not scales:
        raise ValueError("--scales must contain at least one value")

    fold_infos = make_fold_infos(
        y=y,
        classes=classes,
        file_ids=file_ids,
        user_ids=user_ids,
        fold_ids=fold_ids,
        smoother_summary=args.smoother_summary,
    )
    first_lock_groups = make_first_lock_groups(file_ids, user_ids, args.first_lock)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    best: list[tuple[float, np.ndarray, np.ndarray]] = []
    for trial in range(args.trials):
        if trial == 0:
            weights = center
        else:
            scale = scales[(trial - 1) % len(scales)]
            weights = normalize(center + rng.normal(0.0, scale, size=len(center)))
        pred = predict_fixed_viterbi(components, weights, classes, fold_infos, first_lock_groups)
        score = float(f1_score(y, pred, average="macro"))
        rows.append(
            {
                "trial": trial,
                "macro_f1": score,
                "weights": ",".join(f"{value:.8g}" for value in weights),
                "named_weights": ",".join(f"{name}={value:.8g}" for name, value in zip(names, weights, strict=True)),
                "pred_counts": ",".join(str(int(value)) for value in np.bincount(pred, minlength=len(classes))),
            }
        )
        if len(best) < args.top_k or score > best[-1][0]:
            best.append((score, weights.copy(), pred.copy()))
            best.sort(key=lambda item: item[0], reverse=True)
            best = best[: args.top_k]
        if (trial + 1) % 500 == 0:
            print(f"trial {trial + 1}/{args.trials}; best={best[0][0]:.6f}", flush=True)

    result_frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result_frame.to_csv(args.output_dir / "search_results.csv", index=False)
    best_score, best_weights, best_pred = best[0]
    best_proba = np.tensordot(best_weights, components, axes=(0, 0))
    demote_rows: list[dict[str, object]] = []
    if args.fold_fair_demote_class is not None:
        best_pred, demote_rows = apply_fold_fair_demotion(
            y=y,
            pred=best_pred,
            proba=best_proba,
            classes=classes,
            fold_ids=fold_ids,
            target_class=args.fold_fair_demote_class,
        )
        best_score = float(f1_score(y, best_pred, average="macro"))

    np.savez_compressed(
        args.output_dir / "best_predictions.npz",
        proba=best_proba,
        pred=best_pred,
        label=y,
        classes=classes,
        file_id=file_ids,
        user_id=user_ids,
        fold=fold_ids,
        model_names=np.array(names, dtype=object),
        model_weights=best_weights,
    )
    pd.DataFrame(classification_report(y, best_pred, output_dict=True, zero_division=0)).T.to_csv(
        args.output_dir / "best_report.csv"
    )
    if demote_rows:
        pd.DataFrame(demote_rows).to_csv(args.output_dir / "fold_demote_rules.csv", index=False)
    summary = {
        "macro_f1": float(best_score),
        "first_lock": int(args.first_lock),
        "fold_fair_demote_class": args.fold_fair_demote_class,
        "source_npz": str(args.source_npz),
        "smoother_summary": str(args.smoother_summary),
        "model_weights": {name: float(value) for name, value in zip(names, best_weights, strict=True)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


def align_folds(file_ids: np.ndarray, folds_file: Path) -> np.ndarray:
    folds = pd.read_csv(folds_file, usecols=["file_id", "fold"])
    mapping = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    return np.array([mapping[int(file_id)] for file_id in file_ids], dtype=int)


def parse_center(text: str, names: list[str]) -> np.ndarray:
    values = {name: 0.0 for name in names}
    for part in [item.strip() for item in text.split(",") if item.strip()]:
        name, raw_value = part.split("=", 1)
        name = name.strip()
        if name not in values:
            raise ValueError(f"Unknown center component {name!r}; available={names}")
        values[name] = float(raw_value)
    return normalize(np.array([values[name] for name in names], dtype=float))


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def make_fold_infos(
    y: np.ndarray,
    classes: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    fold_ids: np.ndarray,
    smoother_summary: Path,
) -> list[dict[str, object]]:
    fold_params = ast.literal_eval(pd.read_csv(smoother_summary).loc[0, "fold_params"])
    infos = []
    for fold in sorted(np.unique(fold_ids)):
        params = fold_params[str(int(fold))]
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold
        transition, start = estimate_transition_model(
            y=y[train_mask],
            classes=classes,
            file_ids=file_ids[train_mask],
            user_ids=user_ids[train_mask],
            alpha=float(params["alpha"]),
        )
        frame = pd.DataFrame(
            {"row": np.flatnonzero(valid_mask), "file_id": file_ids[valid_mask], "user_id": user_ids[valid_mask]}
        )
        groups = [
            group["row"].to_numpy(dtype=int)
            for _, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False)
        ]
        infos.append(
            {
                "groups": groups,
                "class_weights": np.array(params["class_weights"], dtype=float),
                "log_start": np.log(np.clip(start, 1e-12, 1.0)),
                "log_transition": np.log(np.clip(transition, 1e-12, 1.0)),
                "beta": float(params["beta"]),
            }
        )
    return infos


def make_first_lock_groups(file_ids: np.ndarray, user_ids: np.ndarray, first_lock: int) -> list[np.ndarray]:
    if first_lock <= 0:
        return []
    frame = pd.DataFrame({"row": np.arange(len(file_ids)), "file_id": file_ids, "user_id": user_ids})
    return [
        group["row"].to_numpy(dtype=int)[:first_lock]
        for _, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False)
    ]


def predict_fixed_viterbi(
    components: np.ndarray,
    weights: np.ndarray,
    classes: np.ndarray,
    fold_infos: list[dict[str, object]],
    first_lock_groups: list[np.ndarray],
) -> np.ndarray:
    proba = np.tensordot(weights, components, axes=(0, 0))
    pred = np.empty(len(proba), dtype=int)
    for info in fold_infos:
        log_emission = np.log(np.clip(proba * info["class_weights"].reshape(1, -1), 1e-12, None))
        for idx in info["groups"]:
            path = viterbi_path(
                log_emission[idx],
                info["log_start"],
                info["log_transition"],
                beta=info["beta"],
            )
            pred[idx] = classes[path]
    for idx in first_lock_groups:
        pred[idx] = 0
    return pred


def apply_fold_fair_demotion(
    y: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    fold_ids: np.ndarray,
    target_class: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if target_class not in set(classes.tolist()):
        raise ValueError(f"Cannot demote missing class {target_class}; classes={classes.tolist()}")
    target_idx = int(np.where(classes == target_class)[0][0])
    non_target_idx = [idx for idx, cls in enumerate(classes) if int(cls) != target_class]
    score = np.log(np.clip(proba[:, target_idx], 1e-12, 1.0)) - np.log(
        np.clip(proba[:, non_target_idx].max(axis=1), 1e-12, 1.0)
    )
    fallback = classes[np.array(non_target_idx)[np.argmax(proba[:, non_target_idx], axis=1)]]
    out = pred.copy()
    rows: list[dict[str, object]] = []
    for fold in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold
        base_train_score = float(f1_score(y[train_mask], pred[train_mask], average="macro"))
        best_score = base_train_score
        best_threshold: float | None = None
        thresholds = np.unique(np.quantile(score[train_mask], np.linspace(0.70, 0.999, 180)))
        for threshold in thresholds:
            trial = pred[train_mask].copy()
            demote = (trial == target_class) & (score[train_mask] < threshold)
            trial[demote] = fallback[train_mask][demote]
            trial_score = float(f1_score(y[train_mask], trial, average="macro"))
            if trial_score > best_score + 1e-12:
                best_score = trial_score
                best_threshold = float(threshold)

        valid_pred = pred[valid_mask].copy()
        if best_threshold is not None:
            demote_valid = (valid_pred == target_class) & (score[valid_mask] < best_threshold)
            valid_pred[demote_valid] = fallback[valid_mask][demote_valid]
        out[valid_mask] = valid_pred
        rows.append(
            {
                "fold": int(fold),
                "target_class": int(target_class),
                "train_base_macro_f1": base_train_score,
                "train_rule_macro_f1": best_score,
                "threshold": best_threshold,
                "base_valid_macro_f1": float(f1_score(y[valid_mask], pred[valid_mask], average="macro")),
                "valid_macro_f1": float(f1_score(y[valid_mask], valid_pred, average="macro")),
                "changes": int(np.sum(valid_pred != pred[valid_mask])),
            }
        )
    return out, rows


if __name__ == "__main__":
    main()
