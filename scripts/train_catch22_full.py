#!/usr/bin/env python3
"""Train Catch24-LGBM and Catch24-XGB on all 60 users; emit test predictions.

Reuses the existing train-feature CSV at artifacts/catch22_oof/features_catch22_raw_catch24.csv.
Computes test Catch24 features from artifacts/sequence/test_sequences.npz using
aeon's Catch22 transformer (catch24=True).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Catch24-LGBM/XGB on full 60 users.")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("artifacts/catch22_oof/features_catch22_raw_catch24.csv"),
    )
    parser.add_argument(
        "--test-sequences",
        type=Path,
        default=Path("artifacts/sequence/test_sequences.npz"),
    )
    parser.add_argument(
        "--test-features-cache",
        type=Path,
        default=Path("artifacts/catch22_full/test_features_catch24.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/catch22_full"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_representation_augmented(x: np.ndarray) -> np.ndarray:
    """Augmented representation: original 6 channels + delta + magnitude (≈18 channels typical)."""
    # x: (N, 6, T). For catch24 we use the raw representation to match the existing train CSV
    # which has 144 = 24 * 6 features.
    return x


def compute_test_features(test_seq_path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Loading cached test features from {cache_path}")
        return pd.read_csv(cache_path)

    # Lazy import aeon Catch22
    from aeon.transformations.collection.feature_based import Catch22

    print(f"Loading test sequences from {test_seq_path}")
    d = np.load(test_seq_path, allow_pickle=True)
    x = d["x"].astype(np.float32)  # (N, 6, T)
    file_ids = d["file_ids"].astype(int)
    users = d["users"].astype(str)
    print(f"Test sequences: {x.shape}")

    print("Computing Catch24 features (this takes a few minutes)...")
    transformer = Catch22(catch24=True, replace_nans=True, n_jobs=1)
    values = transformer.fit_transform(x)
    print(f"Catch24 features shape: {values.shape}")

    df = pd.DataFrame(values, columns=[f"c22_{i:03d}" for i in range(values.shape[1])])
    df.insert(0, "file_id", file_ids)
    df.insert(1, "user_id", users)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {cache_path}")
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load train features
    train_df = pd.read_csv(args.train_features)
    train_y = train_df["label"].astype(int).to_numpy() if "label" in train_df.columns else None
    train_feat_cols = [c for c in train_df.columns if c.startswith("c22_")]
    X_train = train_df[train_feat_cols].to_numpy().astype(np.float32)
    print(f"Train features: {X_train.shape}, labels: {train_y.shape}")

    # Compute or load test features
    test_df = compute_test_features(args.test_sequences, args.test_features_cache)
    test_feat_cols = [c for c in test_df.columns if c.startswith("c22_")]
    if set(test_feat_cols) != set(train_feat_cols):
        # Align columns
        common = sorted(set(test_feat_cols) & set(train_feat_cols))
        print(f"WARN: column mismatch; using {len(common)} common features")
        X_train = train_df[common].to_numpy().astype(np.float32)
        X_test = test_df[common].to_numpy().astype(np.float32)
    else:
        X_test = test_df[train_feat_cols].to_numpy().astype(np.float32)
    print(f"Test features: {X_test.shape}")

    test_file_ids = test_df["file_id"].astype(int).to_numpy()
    test_users = test_df["user_id"].astype(str).to_numpy()

    classes = np.arange(6, dtype=int)

    # --- LGBM ---
    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=6,
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        reg_alpha=0.05,
        reg_lambda=0.5,
        min_child_samples=20,
        random_state=args.seed,
        verbose=-1,
    )
    print("\nTraining LGBM on 60 users...")
    lgbm.fit(X_train, train_y)
    test_proba_lgbm = lgbm.predict_proba(X_test)
    print(f"LGBM test proba shape: {test_proba_lgbm.shape}")
    np.savez_compressed(
        args.output_dir / "test_proba_lgbm.npz",
        proba=test_proba_lgbm.astype(np.float32),
        classes=classes,
        file_id=test_file_ids,
        user_id=test_users,
    )

    # --- XGB ---
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        reg_alpha=0.05,
        reg_lambda=0.5,
        tree_method="hist",
        random_state=args.seed,
        verbosity=0,
    )
    print("\nTraining XGB on 60 users...")
    xgb.fit(X_train, train_y)
    test_proba_xgb = xgb.predict_proba(X_test)
    print(f"XGB test proba shape: {test_proba_xgb.shape}")
    np.savez_compressed(
        args.output_dir / "test_proba_xgb.npz",
        proba=test_proba_xgb.astype(np.float32),
        classes=classes,
        file_id=test_file_ids,
        user_id=test_users,
    )

    # Sanity: training macro-F1
    train_pred_lgbm = lgbm.predict(X_train)
    train_pred_xgb = xgb.predict(X_train)
    print(f"\nTrain macro-F1 LGBM: {f1_score(train_y, train_pred_lgbm, average='macro'):.4f}")
    print(f"Train macro-F1 XGB:  {f1_score(train_y, train_pred_xgb, average='macro'):.4f}")

    summary = {
        "lgbm_test_proba": str(args.output_dir / "test_proba_lgbm.npz"),
        "xgb_test_proba": str(args.output_dir / "test_proba_xgb.npz"),
        "lgbm_train_macro_f1": float(f1_score(train_y, train_pred_lgbm, average="macro")),
        "xgb_train_macro_f1": float(f1_score(train_y, train_pred_xgb, average="macro")),
        "n_features": len(train_feat_cols),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
