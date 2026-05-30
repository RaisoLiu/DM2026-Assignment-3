#!/usr/bin/env python3
"""Constrained weighted-average ensemble across N sources, validated on LNUO×6.

NO meta-learner. NO heavy pseudo-labels. Simple convex combination of softmax
probabilities, with weights searched to maximize the *averaged* macro-F1
across 6 leave-15-users-out folds. The convex constraint (weights ≥ 0, sum=1)
prevents the ensemble from collapsing to a degenerate solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted-average ensemble with LNUO validation.")
    parser.add_argument(
        "--source-oofs",
        nargs="+",
        required=True,
        help="OOF npz files (each with proba, label, file_id).",
    )
    parser.add_argument(
        "--source-tests",
        nargs="+",
        required=True,
        help="Test npz files (each with proba, file_id, optional user_id).",
    )
    parser.add_argument(
        "--source-names",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--lnuo-folds",
        nargs="+",
        required=True,
        help="LNUO fold CSVs (each has file_id, user_id, label, fold).",
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
        help="Master fold CSV for the 52-user training set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/weighted_ensemble"),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5000,
        help="Random Dirichlet-sampled weight trials.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {
        "proba": d["proba"].astype(float),
        "label": d["label"].astype(int),
        "file_id": d["file_id"].astype(int),
    }


def load_test(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    out = {
        "proba": d["proba"].astype(float),
        "file_id": d["file_id"].astype(int),
    }
    if "user_id" in d.files:
        out["user_id"] = d["user_id"].astype(str)
    if "classes" in d.files:
        out["classes"] = d["classes"].astype(int)
    return out


def align_to(target_file_ids: np.ndarray, src_proba: np.ndarray, src_file_ids: np.ndarray) -> np.ndarray:
    idx = {int(f): i for i, f in enumerate(src_file_ids)}
    order = np.array([idx[int(f)] for f in target_file_ids], dtype=int)
    return src_proba[order]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === Load reference fold file for target IDs ===
    fold_df = pd.read_csv(args.fold_file)
    target_file_ids = fold_df["file_id"].astype(int).to_numpy()
    target_labels = fold_df["label"].astype(int).to_numpy()
    target_users = fold_df["user_id"].astype(str).to_numpy()
    n = len(target_file_ids)
    print(f"Target 52-user training set: {n} rows")

    # === Load all OOFs, align to target_file_ids ===
    source_oofs = []
    for path, name in zip(args.source_oofs, args.source_names):
        oof = load_oof(Path(path))
        aligned = align_to(target_file_ids, oof["proba"], oof["file_id"])
        source_oofs.append({"name": name, "proba": aligned})
        f1 = f1_score(target_labels, aligned.argmax(axis=1), average="macro")
        print(f"  TRAIN-OOF {name}: shape {aligned.shape}, base macro-F1 = {f1:.6f}")

    # === Load all tests, align to first test's file_ids ===
    source_tests = []
    test_file_ids = None
    test_users = None
    for path, name in zip(args.source_tests, args.source_names):
        test = load_test(Path(path))
        if test_file_ids is None:
            test_file_ids = test["file_id"]
            test_users = test.get("user_id")
        aligned = align_to(test_file_ids, test["proba"], test["file_id"])
        source_tests.append({"name": name, "proba": aligned})
        print(f"  TEST {name}: shape {aligned.shape}")

    K = len(source_oofs)
    classes = np.arange(6, dtype=int)
    if test_users is None:
        # Try to infer from file_id ranges (test IDs > 11020)
        test_users = np.array([f"User_unknown_{int(f)}" for f in test_file_ids])

    # === Load LNUO fold assignments ===
    lnuo_fold_dfs = []
    for path in args.lnuo_folds:
        df = pd.read_csv(path)
        lnuo_fold_dfs.append(df)
        n_held = int((df["fold"] == 1).sum())
        print(f"  LNUO {Path(path).stem}: {n_held} held-out rows / {len(df)}")

    # === Random weight search ===
    rng = np.random.default_rng(args.seed)
    best_score = -1.0
    best_weights = None
    print(f"\nSearching {args.n_trials} Dirichlet-sampled weight trials...")
    for trial in range(args.n_trials):
        weights = rng.dirichlet(np.ones(K) * 0.5)  # tend toward sparsity
        blended = np.zeros_like(source_oofs[0]["proba"])
        for k, s in enumerate(source_oofs):
            blended += weights[k] * s["proba"]

        # Evaluate on LNUO×6: for each fold file, compute macro-F1 on held-out rows
        macros = []
        for df in lnuo_fold_dfs:
            held_ids = df.loc[df["fold"] == 1, "file_id"].astype(int).to_numpy()
            held_idx = {int(f): i for i, f in enumerate(target_file_ids)}
            mask = np.array([int(f) in held_idx for f in held_ids])
            held_ids_filtered = held_ids[mask]
            held_idx_arr = np.array([held_idx[int(f)] for f in held_ids_filtered], dtype=int)
            if len(held_idx_arr) == 0:
                continue
            held_labels = target_labels[held_idx_arr]
            held_pred = blended[held_idx_arr].argmax(axis=1)
            macros.append(f1_score(held_labels, held_pred, average="macro"))
        if not macros:
            continue
        mean_macro = float(np.mean(macros))
        if mean_macro > best_score:
            best_score = mean_macro
            best_weights = weights.copy()

    print(f"\nBest LNUO×6 averaged macro-F1: {best_score:.6f}")
    print(f"Best weights: {{")
    for k, name in enumerate(args.source_names):
        print(f"  {name}: {best_weights[k]:.4f}")
    print(f"}}")

    # === Compute final OOF and test ensembles with best weights ===
    oof_proba = np.zeros_like(source_oofs[0]["proba"])
    test_proba = np.zeros_like(source_tests[0]["proba"])
    for k in range(K):
        oof_proba += best_weights[k] * source_oofs[k]["proba"]
        test_proba += best_weights[k] * source_tests[k]["proba"]
    # Re-normalize (should be ~1.0 already)
    test_proba = test_proba / test_proba.sum(axis=1, keepdims=True).clip(min=1e-8)
    oof_proba = oof_proba / oof_proba.sum(axis=1, keepdims=True).clip(min=1e-8)

    oof_macro = f1_score(target_labels, oof_proba.argmax(axis=1), average="macro")
    print(f"\nFinal ensemble OOF macro-F1 (on 52-user fold): {oof_macro:.6f}")

    # Save outputs
    np.savez_compressed(
        args.output_dir / "oof.npz",
        proba=oof_proba.astype(np.float32),
        classes=classes,
        label=target_labels.astype(np.int64),
        file_id=target_file_ids.astype(np.int64),
        user_id=target_users.astype(str),
        weights=best_weights.astype(np.float32),
        source_names=np.array(args.source_names, dtype=object),
    )
    np.savez_compressed(
        args.output_dir / "test.npz",
        proba=test_proba.astype(np.float32),
        classes=classes,
        file_id=test_file_ids.astype(np.int64),
        user_id=test_users.astype(str),
        weights=best_weights.astype(np.float32),
        source_names=np.array(args.source_names, dtype=object),
    )
    summary = {
        "best_lnuo_macro_f1": float(best_score),
        "final_oof_macro_f1": float(oof_macro),
        "weights": {name: float(w) for name, w in zip(args.source_names, best_weights)},
        "source_oofs": list(args.source_oofs),
        "source_tests": list(args.source_tests),
        "n_trials": args.n_trials,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir}/{{oof,test,summary}}.{{npz,json}}")


if __name__ == "__main__":
    main()
