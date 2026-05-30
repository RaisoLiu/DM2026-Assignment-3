#!/usr/bin/env python3
"""Train LightGBM on Chronos embeddings (5-fold StratifiedGroupKFold OOF).

Inputs:
  artifacts/external_pretrained/chronos_train_embeddings.npz
  artifacts/external_pretrained/chronos_test_embeddings.npz
  artifacts/folds/sgkf_seed2026.csv

Outputs:
  artifacts/chronos_lgbm/oof_proba.npz     (N_train, 6) + file_id, y
  artifacts/chronos_lgbm/test_proba.npz    (N_test, 6) + file_id
  artifacts/chronos_lgbm/summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-emb", type=Path, default=Path("artifacts/external_pretrained/chronos_train_embeddings.npz"))
    p.add_argument("--test-emb", type=Path, default=Path("artifacts/external_pretrained/chronos_test_embeddings.npz"))
    p.add_argument("--fold-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/chronos_lgbm"))
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--include-loc-scale", action="store_true", default=True,
                   help="Include per-channel loc/scale as extra features (6 * 2 = 12).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    train = np.load(args.train_emb, allow_pickle=True)
    test = np.load(args.test_emb, allow_pickle=True)

    X_train = train["emb"].astype(np.float32)
    y = train["y"].astype(int)
    train_fids = train["file_ids"].astype(int)

    X_test = test["emb"].astype(np.float32)
    test_fids = test["file_ids"].astype(int)

    if args.include_loc_scale and "loc" in train.files and "scale" in train.files:
        # Combine loc + scale as small handcrafted features alongside the embedding
        X_train = np.concatenate([X_train, train["loc"], train["scale"]], axis=1).astype(np.float32)
        X_test = np.concatenate([X_test, test["loc"], test["scale"]], axis=1).astype(np.float32)
    elif args.include_loc_scale:
        print("Skipping loc/scale (not in embeddings npz)")

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    folds = pd.read_csv(args.fold_file)
    fold_lookup = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    fold_arr = np.array([fold_lookup.get(int(fid), -1) for fid in train_fids])
    print(f"Folds present in fold-file for train: {(fold_arr != -1).sum()}/{len(train_fids)}")

    n_classes = int(y.max()) + 1
    oof_proba = np.zeros((len(y), n_classes), dtype=np.float32)
    test_proba_folds = np.zeros((len(X_test), n_classes), dtype=np.float32)
    fold_metrics = []
    unique_folds = sorted(set(fold_arr.tolist()) - {-1})

    for fold in unique_folds:
        tr_mask = fold_arr != fold
        va_mask = fold_arr == fold
        if va_mask.sum() == 0:
            continue
        print(f"Fold {fold}: train {tr_mask.sum()}, val {va_mask.sum()}")
        t0 = time.time()
        model = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=20,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=args.seed,
            verbose=-1,
        )
        model.fit(X_train[tr_mask], y[tr_mask])
        oof_proba[va_mask] = model.predict_proba(X_train[va_mask])
        oof_pred = oof_proba[va_mask].argmax(axis=1)
        f1 = f1_score(y[va_mask], oof_pred, average="macro")
        # Also predict on test for this fold
        test_proba_folds += model.predict_proba(X_test) / len(unique_folds)
        fold_metrics.append({"fold": int(fold), "macro_f1": float(f1), "time_s": time.time() - t0})
        print(f"  macro-F1 = {f1:.4f} ({time.time() - t0:.0f}s)")

    overall_oof = f1_score(y, oof_proba.argmax(axis=1), average="macro")
    per_class_oof = f1_score(y, oof_proba.argmax(axis=1), average=None)
    print(f"\nOverall OOF macro-F1: {overall_oof:.4f}")
    print(f"Per-class F1: {[f'{f:.3f}' for f in per_class_oof]}")
    print("\nClassification report:")
    print(classification_report(y, oof_proba.argmax(axis=1), digits=4))

    out_oof = args.output_dir / "oof_proba.npz"
    np.savez_compressed(
        out_oof,
        proba=oof_proba,
        y=y,
        file_id=train_fids,
        classes=np.arange(n_classes, dtype=np.int64),
        user_id=train["users"],
    )
    print(f"Wrote {out_oof}")

    out_test = args.output_dir / "test_proba.npz"
    np.savez_compressed(
        out_test,
        proba=test_proba_folds,
        file_id=test_fids,
        classes=np.arange(n_classes, dtype=np.int64),
        user_id=test["users"],
    )
    print(f"Wrote {out_test}")

    summary = {
        "overall_oof_macro_f1": float(overall_oof),
        "per_class_oof_f1": [float(x) for x in per_class_oof],
        "fold_metrics": fold_metrics,
        "n_features": int(X_train.shape[1]),
        "n_classes": int(n_classes),
        "model": str(train.get("model", "unknown")),
        "pool": str(train.get("pool", "unknown")),
        "include_loc_scale": bool(args.include_loc_scale),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
