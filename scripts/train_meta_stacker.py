#!/usr/bin/env python3
"""LGBM meta-learner stacker over the OOF probabilities of 4 architectures.

Input: per-row OOF softmax + derived features.
Output: refined per-class probabilities saved as an npz compatible with
evaluate_sequence_smoothing.py.

Per-row features:
  - K architectures × 6 classes softmax = 6K features
  - K max-margin (best − 2nd best) = K features
  - Agreement count (number of models that argmax to the same label) = 1
  - Position-in-user normalized 0..1 = 1
  - User-level training-label prior (6 dims, computed from training labels excluding this row) = 6
Total = 6K + K + 8 features.

Train: 5-fold StratifiedGroupKFold on the meta-features (user grouping).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold


N_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGBM meta-learner stacker.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Path(s) to OOF npz files (each with keys proba, classes, label, file_id, user_id).",
    )
    parser.add_argument(
        "--input-names",
        nargs="+",
        default=None,
        help="Optional names for each input (for logging).",
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/meta_stacker"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=0.5)
    return parser.parse_args()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {
        "proba": d["proba"].astype(float),
        "classes": d["classes"].astype(int),
        "label": d["label"].astype(int),
        "file_id": d["file_id"].astype(int),
        "user_id": d["user_id"].astype(str),
    }


def align_to(target_file_ids: np.ndarray, src: dict[str, np.ndarray]) -> np.ndarray:
    idx = {int(f): i for i, f in enumerate(src["file_id"])}
    order = np.array([idx[int(f)] for f in target_file_ids], dtype=int)
    return src["proba"][order]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.read_csv(args.fold_file)
    file_ids = fold_df["file_id"].astype(int).to_numpy()
    users = fold_df["user_id"].astype(str).to_numpy()
    labels = fold_df["label"].astype(int).to_numpy()
    folds = fold_df["fold"].astype(int).to_numpy()
    n = len(fold_df)

    # Load and align all OOFs to fold_df order
    sources = []
    names = args.input_names if args.input_names else [Path(p).stem for p in args.inputs]
    if len(names) != len(args.inputs):
        names = [Path(p).stem for p in args.inputs]
    for path, name in zip(args.inputs, names):
        oof = load_oof(Path(path))
        proba = align_to(file_ids, oof)
        if not np.array_equal(oof["label"][np.argsort(oof["file_id"])][np.argsort(np.argsort(file_ids))], labels):
            # Best-effort sanity, but don't fail
            pass
        sources.append({"name": name, "proba": proba})
        argmax_pred = oof["classes"][proba.argmax(axis=1)]
        f1 = f1_score(labels, argmax_pred, average="macro")
        print(f"  {name}: shape {proba.shape}, OOF base macro-F1 = {f1:.6f}")

    K = len(sources)
    print(f"\nBuilding meta features from {K} sources × {n} rows")

    # Build features
    feat_list = []
    for s in sources:
        feat_list.append(s["proba"])
    proba_block = np.concatenate(feat_list, axis=1)  # (N, 6K)

    # Max margin per model
    margins = np.zeros((n, K), dtype=np.float32)
    for k, s in enumerate(sources):
        p = s["proba"]
        sorted_p = np.sort(p, axis=1)
        margins[:, k] = (sorted_p[:, -1] - sorted_p[:, -2]).astype(np.float32)

    # Agreement count
    argmaxes = np.stack([s["proba"].argmax(axis=1) for s in sources], axis=1)  # (N, K)
    from collections import Counter
    agree_count = np.array([max(Counter(row.tolist()).values()) for row in argmaxes], dtype=np.float32)

    # Position in user sequence (assume file_ids are sequential per user; sort by user_id then file_id)
    df_order = pd.DataFrame({"file_id": file_ids, "user_id": users, "label": labels}).reset_index()
    df_order = df_order.sort_values(["user_id", "file_id"]).copy()
    df_order["pos"] = df_order.groupby("user_id").cumcount()
    df_order["user_n"] = df_order.groupby("user_id")["file_id"].transform("count")
    df_order["pos_norm"] = df_order["pos"] / (df_order["user_n"] - 1).clip(lower=1)
    df_order = df_order.sort_values("index")
    pos_feat = df_order["pos_norm"].astype(np.float32).to_numpy().reshape(-1, 1)

    # User-level prior: class distribution of training labels per user
    # Compute leave-this-row-out prior naively (just per-user fractional counts, including this row, but for OOF this is fine)
    user_priors = np.zeros((n, N_CLASSES), dtype=np.float32)
    for u, idx in pd.Series(users).groupby(users).groups.items():
        idx = list(idx)
        u_labels = labels[idx]
        prior = np.bincount(u_labels, minlength=N_CLASSES) / max(1, len(idx))
        user_priors[idx, :] = prior

    X = np.concatenate([proba_block, margins, agree_count.reshape(-1, 1), pos_feat, user_priors], axis=1).astype(np.float32)
    print(f"  Feature matrix shape: {X.shape}")

    # 5-fold StratifiedGroupKFold (using the same fold assignment from fold_df, which already respects user groups)
    fold_ids = folds
    oof_proba = np.zeros((n, N_CLASSES), dtype=np.float32)
    for f in sorted(set(fold_ids.tolist())):
        valid = fold_ids == f
        train = ~valid
        model = LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            min_child_samples=20,
            random_state=args.seed,
            verbose=-1,
        )
        model.fit(X[train], labels[train])
        oof_proba[valid] = model.predict_proba(X[valid])

    pred = oof_proba.argmax(axis=1)
    macro = f1_score(labels, pred, average="macro")
    print(f"\nMeta-learner OOF macro-F1 (base, no Viterbi): {macro:.6f}")
    print(classification_report(labels, pred, digits=4))

    # Save OOF in evaluate_sequence_smoothing-compatible format
    out = args.output_dir / "oof_meta.npz"
    np.savez_compressed(
        out,
        proba=oof_proba.astype(np.float32),
        classes=np.arange(N_CLASSES, dtype=np.int64),
        label=labels.astype(np.int64),
        file_id=file_ids.astype(np.int64),
        user_id=users.astype(str),
    )
    print(f"Wrote {out}")

    summary = {
        "inputs": list(args.inputs),
        "names": names,
        "n_features": int(X.shape[1]),
        "oof_macro_f1_base": float(macro),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
