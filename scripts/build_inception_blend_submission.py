#!/usr/bin/env python3
"""Add InceptionTime as a sidecar to the centered-meta blend.

Strategy:
  - Restrict the existing 11-component OOF blend (60 users) to the 52
    train-CV users.
  - Combine the centered-meta mixed proba with the InceptionTime softmax via a
    learned ``lambda`` and class-bias multipliers; search to maximize
    fold-fair Viterbi macro-F1 on the seed-2026 + seed-2027 averaged folds.
  - Fit per-fold Viterbi (alpha, beta) for the new blend.
  - Translate to test using the existing centered-meta test proba plus
    InceptionTime test proba; emit a Kaggle CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission

from evaluate_sequence_smoothing import (
    estimate_transition_model,
    tune_viterbi_params,
    viterbi_predict_by_user,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build InceptionTime + centered-meta blend submission.")
    parser.add_argument(
        "--centered-meta-oof",
        type=Path,
        default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"),
    )
    parser.add_argument(
        "--inception-oof",
        type=Path,
        default=Path("artifacts/inception_oof/oof_inception_avg.npz"),
        help="Averaged InceptionTime OOF (over seeds 2026 & 2027).",
    )
    parser.add_argument(
        "--inception-test",
        type=Path,
        default=Path("artifacts/inception_full/test_proba_avg.npz"),
        help="Averaged InceptionTime test softmax (over seeds 2026 & 2027).",
    )
    parser.add_argument(
        "--centered-meta-test-csv",
        type=Path,
        default=Path("submissions/submission_centered_meta_viterbi_oof07693.csv"),
        help="Existing centered-meta submission CSV; we use it to recover the centered-meta test proba via re-execution if not cached.",
    )
    parser.add_argument(
        "--centered-meta-test-proba",
        type=Path,
        default=Path("artifacts/inception_blend/test_centered_meta_proba.npz"),
        help="Cache for the centered-meta blended test proba (mixed across 11 components, before Viterbi).",
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
    )
    parser.add_argument(
        "--secondary-fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2027_train52.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/inception_blend"),
    )
    parser.add_argument(
        "--submission-csv",
        type=Path,
        default=Path("submissions/submission_inception_blend_viterbi.csv"),
    )
    parser.add_argument(
        "--lambda-grid",
        default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5",
    )
    parser.add_argument(
        "--class-bias-trials",
        type=int,
        default=200,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--alpha-grid",
        default="0.1,0.3,1.0,3.0",
    )
    parser.add_argument(
        "--beta-grid",
        default="0.0,0.02,0.05,0.08,0.12,0.18,0.27,0.40,0.60,0.90,1.30",
    )
    return parser.parse_args()


def parse_grid(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def load_centered_meta(npz_path: Path) -> dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "proba": d["proba"].astype(float),
        "classes": d["classes"].astype(int),
        "label": d["label"].astype(int),
        "file_id": d["file_id"].astype(int),
        "user_id": d["user_id"].astype(str),
        "fold": d["fold"].astype(int) if "fold" in d.files else None,
        "class_weights": d["class_weights"].astype(float),
    }


def load_npz_proba(npz_path: Path) -> dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "proba": d["proba"].astype(float),
        "classes": d["classes"].astype(int),
        "label": d["label"].astype(int) if "label" in d.files else None,
        "file_id": d["file_id"].astype(int),
        "user_id": d["user_id"].astype(str) if "user_id" in d.files else None,
    }


def align_to_keys(
    src: dict[str, np.ndarray], target_file_ids: np.ndarray
) -> dict[str, np.ndarray]:
    src_index = {int(fid): i for i, fid in enumerate(src["file_id"])}
    order = np.array([src_index[int(f)] for f in target_file_ids], dtype=np.int64)
    out = {
        "proba": src["proba"][order],
        "classes": src["classes"],
        "label": src["label"][order] if src.get("label") is not None else None,
        "file_id": src["file_id"][order],
    }
    if src.get("user_id") is not None:
        out["user_id"] = src["user_id"][order]
    return out


def macro_f1_from(
    proba: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
) -> float:
    pred = classes[np.argmax(proba * class_weights.reshape(1, -1), axis=1)]
    return float(f1_score(labels, pred, average="macro"))


def search_blend(
    cm_proba: np.ndarray,
    inc_proba: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    folds: pd.DataFrame,
    lambda_grid: tuple[float, ...],
    class_bias_trials: int,
    alpha_grid: tuple[float, ...],
    beta_grid: tuple[float, ...],
    base_class_weights: np.ndarray,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    fold_lookup = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    fold_ids = np.array([fold_lookup[int(f)] for f in file_ids], dtype=int)
    n_folds = int(fold_ids.max())
    assert set(fold_ids.tolist()) == set(range(1, n_folds + 1))

    rows: list[dict[str, object]] = []
    for lam in lambda_grid:
        blended = (1.0 - lam) * cm_proba + lam * inc_proba
        # Class-weight search (random multipliers around 1.0)
        for trial in range(class_bias_trials):
            if trial == 0:
                bias = base_class_weights.copy()
            else:
                jitter = rng.uniform(0.7, 1.4, size=len(classes))
                bias = (base_class_weights * jitter).astype(float)
                bias = bias / bias.mean() * base_class_weights.mean()
            base_macro = macro_f1_from(blended, classes, labels, bias)
            rows.append({
                "lambda": float(lam),
                "trial": trial,
                "class_weights": bias.tolist(),
                "base_macro_f1": base_macro,
            })

    rows.sort(key=lambda r: -r["base_macro_f1"])
    # Keep top-3 per lambda + overall top-15 to ensure each lambda is evaluated
    by_lambda: dict[float, list[dict]] = {}
    for r in rows:
        by_lambda.setdefault(float(r["lambda"]), []).append(r)
    top = []
    seen = set()
    for lam, group in by_lambda.items():
        for r in group[:3]:
            key = (r["lambda"], r["trial"])
            if key not in seen:
                top.append(r)
                seen.add(key)
    # Add overall best 15 trials regardless of lambda
    for r in rows[:15]:
        key = (r["lambda"], r["trial"])
        if key not in seen:
            top.append(r)
            seen.add(key)
    top.sort(key=lambda r: -r["base_macro_f1"])
    print(f"Top base macro-F1 candidates (out of {len(rows)} trials, {len(top)} kept for fold-fair eval):")
    for r in top[:8]:
        print(f"  lambda={r['lambda']:.2f} base={r['base_macro_f1']:.6f}")

    # For top candidates, evaluate fold-fair Viterbi
    fold_fair_results = []
    for r in top[:60]:
        lam = r["lambda"]
        blended = (1.0 - lam) * cm_proba + lam * inc_proba
        cw = np.array(r["class_weights"], dtype=float)
        params = tune_viterbi_params(
            proba=blended,
            y=labels,
            classes=classes,
            file_ids=file_ids,
            user_ids=user_ids,
            class_weights=cw,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            stay_grid=(0.0,),
        )
        # Approximate fold-fair score: use this fold's tuned params on this fold,
        # but evaluate via global-Viterbi macro-F1 (a tight upper bound) and a
        # leave-this-fold-out macro-F1 estimate.
        transition, start = estimate_transition_model(
            y=labels,
            classes=classes,
            file_ids=file_ids,
            user_ids=user_ids,
            alpha=params["alpha"],
        )
        global_pred = viterbi_predict_by_user(
            proba=blended,
            classes=classes,
            file_ids=file_ids,
            user_ids=user_ids,
            class_weights=cw,
            transition=transition,
            start=start,
            beta=params["beta"],
            stay_bonus=params["stay_bonus"],
        )
        global_macro = float(f1_score(labels, global_pred, average="macro"))
        # Cheap leave-fold-out fairness: tune Viterbi on each held-in fold, score on left-out
        fold_macros = []
        for k in range(1, n_folds + 1):
            train_mask = fold_ids != k
            if not train_mask.any():
                continue
            params_k = tune_viterbi_params(
                proba=blended[train_mask],
                y=labels[train_mask],
                classes=classes,
                file_ids=file_ids[train_mask],
                user_ids=user_ids[train_mask],
                class_weights=cw,
                alpha_grid=alpha_grid,
                beta_grid=beta_grid,
                stay_grid=(0.0,),
            )
            tr_k, st_k = estimate_transition_model(
                y=labels[train_mask],
                classes=classes,
                file_ids=file_ids[train_mask],
                user_ids=user_ids[train_mask],
                alpha=params_k["alpha"],
            )
            valid_mask = fold_ids == k
            if not valid_mask.any():
                continue
            pred_k = viterbi_predict_by_user(
                proba=blended[valid_mask],
                classes=classes,
                file_ids=file_ids[valid_mask],
                user_ids=user_ids[valid_mask],
                class_weights=cw,
                transition=tr_k,
                start=st_k,
                beta=params_k["beta"],
                stay_bonus=params_k["stay_bonus"],
            )
            fold_macros.append(float(f1_score(labels[valid_mask], pred_k, average="macro")))
        fold_fair_macro = float(np.mean(fold_macros)) if fold_macros else float("nan")
        fold_fair_results.append({
            **r,
            "alpha": float(params["alpha"]),
            "beta": float(params["beta"]),
            "global_viterbi_macro_f1": global_macro,
            "fold_fair_viterbi_macro_f1": fold_fair_macro,
        })
        print(
            f"  lambda={lam:.2f} alpha={params['alpha']} beta={params['beta']:.2f}"
            f" global={global_macro:.6f} fair={fold_fair_macro:.6f}",
            flush=True,
        )

    fold_fair_results.sort(key=lambda r: -r["fold_fair_viterbi_macro_f1"])
    return {
        "best": fold_fair_results[0] if fold_fair_results else None,
        "candidates": fold_fair_results,
    }


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cm = load_centered_meta(args.centered_meta_oof)
    inc = load_npz_proba(args.inception_oof)
    print(
        f"Loaded centered-meta OOF: {len(cm['file_id'])} rows, classes={cm['classes'].tolist()}",
        flush=True,
    )
    print(
        f"Loaded inception OOF: {len(inc['file_id'])} rows",
        flush=True,
    )

    folds = pd.read_csv(args.fold_file)
    target_file_ids = folds["file_id"].astype(int).to_numpy()
    target_users = folds["user_id"].astype(str).to_numpy()
    target_labels = folds["label"].astype(int).to_numpy()

    cm_aligned = align_to_keys(cm, target_file_ids)
    inc_aligned = align_to_keys(inc, target_file_ids)

    # Sanity: labels match
    assert np.array_equal(cm_aligned["label"], target_labels), "centered_meta label mismatch"
    if inc_aligned.get("label") is not None:
        assert np.array_equal(inc_aligned["label"], target_labels), "inception label mismatch"

    classes = cm_aligned["classes"]
    base_cw = cm["class_weights"]
    print(f"Existing class_weights: {base_cw}", flush=True)

    # Sanity: baseline (lambda=0) should reproduce the existing 52-user OOF
    base_blend = cm_aligned["proba"]
    base_macro = macro_f1_from(base_blend, classes, target_labels, base_cw)
    print(f"Centered-meta-only base macro-F1 (52-user, with class weights): {base_macro:.6f}", flush=True)

    lambda_grid = parse_grid(args.lambda_grid)
    alpha_grid = parse_grid(args.alpha_grid)
    beta_grid = parse_grid(args.beta_grid)

    result = search_blend(
        cm_proba=cm_aligned["proba"],
        inc_proba=inc_aligned["proba"],
        classes=classes,
        labels=target_labels,
        file_ids=target_file_ids,
        user_ids=target_users,
        folds=folds,
        lambda_grid=lambda_grid,
        class_bias_trials=args.class_bias_trials,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        base_class_weights=base_cw,
        seed=args.seed,
    )

    best = result["best"]
    if best is None:
        raise SystemExit("No valid blend found")

    print("\n=== Best blend ===")
    print(json.dumps({k: v for k, v in best.items() if k != "trial"}, indent=2))

    # Save the OOF blend with all 52-user info
    blended_oof = (1.0 - best["lambda"]) * cm_aligned["proba"] + best["lambda"] * inc_aligned["proba"]
    cw_best = np.array(best["class_weights"], dtype=float)
    out_oof = args.output_dir / "oof_blend_centered_meta_with_inception.npz"
    np.savez_compressed(
        out_oof,
        proba=blended_oof,
        classes=classes,
        label=target_labels,
        file_id=target_file_ids,
        user_id=target_users,
        class_weights=cw_best,
        lambda_inception=float(best["lambda"]),
        alpha=float(best["alpha"]),
        beta=float(best["beta"]),
    )
    print(f"Wrote {out_oof}", flush=True)

    out_summary = args.output_dir / "search_summary.json"
    out_summary.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_summary}", flush=True)


if __name__ == "__main__":
    main()
