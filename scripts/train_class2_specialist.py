#!/usr/bin/env python3
"""H4: Class-2 (and class-4) binary specialist on transition-shape features.

Per-class binary classifiers using hand-crafted features. Targets the rare-class
F1 bottleneck without overfit risk (engineered features, small LGBM/XGB).

Outputs:
  - OOF probas: artifacts/class_specialist_oof/oof_c{2,4}.npz
  - Test probas: artifacts/class_specialist_full/test_c{2,4}.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class-K binary specialist.")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("artifacts/features_transition/train_transition.csv"),
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=Path("artifacts/features_transition/test_transition.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/class_specialist"),
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=2,
        help="Class to specialize on (binary one-vs-rest).",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                        help="If set, pass to LGBM for class-imbalance handling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_features)
    feat_cols = [c for c in train_df.columns if c not in ("file_id", "user_id", "label")]
    print(f"Features: {len(feat_cols)} columns; train rows: {len(train_df)}")

    X = train_df[feat_cols].to_numpy().astype(np.float32)
    y_full = train_df["label"].astype(int).to_numpy()
    y_bin = (y_full == args.target_class).astype(int)
    users = train_df["user_id"].astype(str).to_numpy()
    file_ids = train_df["file_id"].astype(int).to_numpy()

    n_pos = int(y_bin.sum())
    n_neg = int((1 - y_bin).sum())
    print(f"Class-{args.target_class} positives: {n_pos} / {len(y_bin)} ({100*n_pos/len(y_bin):.2f}%)")

    if args.scale_pos_weight is None:
        # Default: ratio of negatives to positives
        spw = float(n_neg / max(n_pos, 1))
    else:
        spw = args.scale_pos_weight
    print(f"scale_pos_weight: {spw:.2f}")

    # CV: StratifiedGroupKFold by user, stratified on binary label
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_proba = np.zeros(len(y_bin), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y_bin, users), start=1):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            scale_pos_weight=spw,
            reg_alpha=0.05,
            reg_lambda=0.5,
            min_child_samples=10,
            random_state=args.seed,
            verbose=-1,
        )
        model.fit(X[tr_idx], y_bin[tr_idx])
        oof_proba[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        ap = average_precision_score(y_bin[va_idx], oof_proba[va_idx])
        f1_at_05 = f1_score(y_bin[va_idx], (oof_proba[va_idx] >= 0.5).astype(int))
        fold_scores.append({"fold": fold, "ap": float(ap), "f1_at_0.5": float(f1_at_05)})
        print(f"  fold {fold}: AP={ap:.4f}, F1@0.5={f1_at_05:.4f}")

    oof_ap = average_precision_score(y_bin, oof_proba)
    print(f"\nOOF AP (avg precision): {oof_ap:.4f}")

    # Find best decision threshold on OOF
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_bin, (oof_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    print(f"Best OOF F1 (binary): {best_f1:.4f} at threshold {best_thresh:.3f}")

    # Train on full data, predict test
    print("Training full-data model for test predictions...")
    full_model = LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        scale_pos_weight=spw,
        reg_alpha=0.05,
        reg_lambda=0.5,
        min_child_samples=10,
        random_state=args.seed,
        verbose=-1,
    )
    full_model.fit(X, y_bin)
    test_df = pd.read_csv(args.test_features)
    X_test = test_df[feat_cols].to_numpy().astype(np.float32)
    test_proba = full_model.predict_proba(X_test)[:, 1]
    print(f"Test proba shape: {test_proba.shape}, min={test_proba.min():.4f}, max={test_proba.max():.4f}")
    print(f"Test rows above best_thresh ({best_thresh:.3f}): {int((test_proba >= best_thresh).sum())}")

    # Save OOF and test
    np.savez_compressed(
        args.output_dir / f"oof_c{args.target_class}.npz",
        proba=oof_proba.astype(np.float32),
        binary_label=y_bin.astype(np.int64),
        full_label=y_full.astype(np.int64),
        file_id=file_ids.astype(np.int64),
        user_id=users.astype(str),
        target_class=args.target_class,
        best_threshold=best_thresh,
    )
    np.savez_compressed(
        args.output_dir / f"test_c{args.target_class}.npz",
        proba=test_proba.astype(np.float32),
        file_id=test_df["file_id"].astype(int).to_numpy(),
        user_id=test_df["user_id"].astype(str).to_numpy(),
        target_class=args.target_class,
        best_threshold=best_thresh,
    )
    summary = {
        "target_class": args.target_class,
        "oof_ap": float(oof_ap),
        "best_oof_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "fold_scores": fold_scores,
        "n_features": int(X.shape[1]),
        "n_positives": int(n_pos),
        "n_negatives": int(n_neg),
        "scale_pos_weight": float(spw),
    }
    (args.output_dir / f"summary_c{args.target_class}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote OOF + test probas + summary for class-{args.target_class}")


if __name__ == "__main__":
    main()
