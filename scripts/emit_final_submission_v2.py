#!/usr/bin/env python3
"""Combine 4-architecture ensemble + centered-meta into the final submission CSV.

Pipeline:
  1. Load OOF probas from 4 architectures + centered-meta (train, 9589 rows)
  2. Load test probas from 4 architectures (full-60-user M60 models, avg across seeds)
     + centered-meta test proba (from existing artifact)
  3. Train meta-learner LGBM on ALL OOF data (not fold-CV); apply to test features
  4. Apply trained hybrid-rules classifier to test predictions for class-recovery
  5. Write final CSV
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dm2026_asg3.data import load_sample_submission
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_sequence_smoothing import (
    estimate_transition_model,
    viterbi_predict_by_user,
)


N_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit final ensemble submission CSV.")
    parser.add_argument(
        "--source-oofs",
        nargs="+",
        default=[
            "artifacts/inception_oof_ssl_v2/oof_inception_avg5.npz",
            "artifacts/resnet1d_oof_pseudo/oof_resnet1d_avg3.npz",
            "artifacts/tcn_oof_pseudo/oof_tcn_avg3.npz",
            "artifacts/patchtst_oof_pseudo/oof_patchtst_avg3.npz",
            "artifacts/blend_search/oof_blend_centered_meta_round2_best.npz",
        ],
    )
    parser.add_argument(
        "--source-test-probas",
        nargs="+",
        default=[
            "artifacts/inception_full_v2/test_avg.npz",
            "artifacts/resnet1d_full_v2/test_avg.npz",
            "artifacts/tcn_full_v2/test_avg.npz",
            "artifacts/patchtst_full_v2/test_avg.npz",
            "artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz",
        ],
    )
    parser.add_argument(
        "--source-names",
        nargs="+",
        default=["inception_v2", "resnet1d", "tcn", "patchtst", "centered_meta"],
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
    )
    parser.add_argument(
        "--hybrid-rules",
        type=Path,
        default=Path("artifacts/hybrid_rules/rules.json"),
    )
    parser.add_argument(
        "--hybrid-classifier",
        type=Path,
        default=Path("artifacts/hybrid_rules/switch_classifier.joblib"),
    )
    parser.add_argument(
        "--anchor-csv",
        type=Path,
        default=Path("submissions/submission_centered_meta_viterbi_oof07693.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/submission_ensemble_v2.csv"),
    )
    parser.add_argument(
        "--meta-metrics",
        type=Path,
        default=Path("artifacts/meta_stacker_viterbi/meta_metrics.json"),
        help="meta_metrics.json from fold-fair Viterbi tuning on meta-learner OOF.",
    )
    parser.add_argument(
        "--apply-viterbi",
        action="store_true",
        default=True,
        help="Apply Viterbi smoothing to meta_test_proba before hybrid recovery.",
    )
    parser.add_argument(
        "--no-viterbi",
        dest="apply_viterbi",
        action="store_false",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {
        "proba": d["proba"].astype(float),
        "classes": d["classes"].astype(int),
        "label": d["label"].astype(int),
        "file_id": d["file_id"].astype(int),
        "user_id": d["user_id"].astype(str) if "user_id" in d.files else None,
    }


def load_test(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {
        "proba": d["proba"].astype(float),
        "classes": d["classes"].astype(int),
        "file_id": d["file_id"].astype(int),
        "user_id": d["user_id"].astype(str) if "user_id" in d.files else None,
    }


def align_to(target_file_ids: np.ndarray, src: dict[str, np.ndarray]) -> np.ndarray:
    idx = {int(f): i for i, f in enumerate(src["file_id"])}
    order = np.array([idx[int(f)] for f in target_file_ids], dtype=int)
    return src["proba"][order]


def build_meta_features(
    proba_per_source: list[np.ndarray],
    labels_for_user_prior: np.ndarray | None,
    file_ids: np.ndarray,
    users: np.ndarray,
    user_prior_table: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Mirror of train_meta_stacker.py feature build."""
    n = proba_per_source[0].shape[0]
    K = len(proba_per_source)
    feats = []
    # Per-source soft probas
    feats.append(np.concatenate(proba_per_source, axis=1))   # 6K
    # Max margin per source
    margins = np.zeros((n, K), dtype=np.float32)
    for k, p in enumerate(proba_per_source):
        sp = np.sort(p, axis=1)
        margins[:, k] = (sp[:, -1] - sp[:, -2]).astype(np.float32)
    feats.append(margins)
    # Agreement count
    from collections import Counter
    argmaxes = np.stack([p.argmax(axis=1) for p in proba_per_source], axis=1)
    agree_count = np.array([max(Counter(row.tolist()).values()) for row in argmaxes], dtype=np.float32)
    feats.append(agree_count.reshape(-1, 1))
    # Position in user sequence
    df_order = pd.DataFrame({"file_id": file_ids, "user_id": users}).reset_index()
    df_order = df_order.sort_values(["user_id", "file_id"]).copy()
    df_order["pos"] = df_order.groupby("user_id").cumcount()
    df_order["user_n"] = df_order.groupby("user_id")["file_id"].transform("count")
    df_order["pos_norm"] = df_order["pos"] / (df_order["user_n"] - 1).clip(lower=1)
    df_order = df_order.sort_values("index")
    feats.append(df_order["pos_norm"].astype(np.float32).to_numpy().reshape(-1, 1))
    # User-level prior
    if user_prior_table is None:
        # Compute from labels_for_user_prior (only valid for train)
        if labels_for_user_prior is None:
            user_priors = np.zeros((n, N_CLASSES), dtype=np.float32)
        else:
            user_priors = np.zeros((n, N_CLASSES), dtype=np.float32)
            for u, idx in pd.Series(users).groupby(users).groups.items():
                idx = list(idx)
                u_labels = labels_for_user_prior[idx]
                prior = np.bincount(u_labels, minlength=N_CLASSES) / max(1, len(idx))
                user_priors[idx, :] = prior
    else:
        user_priors = np.zeros((n, N_CLASSES), dtype=np.float32)
        for i, u in enumerate(users):
            user_priors[i] = user_prior_table.get(str(u), np.zeros(N_CLASSES, dtype=np.float32))
    feats.append(user_priors)
    return np.concatenate(feats, axis=1).astype(np.float32)


def build_hybrid_features(
    meta_proba: np.ndarray,
    meta_pred: np.ndarray,
    anchor_pred: np.ndarray,
    proba_per_source: list[np.ndarray],
) -> np.ndarray:
    """Mirror of train_hybrid_rules.py feature build."""
    n = meta_proba.shape[0]
    K = len(proba_per_source)
    feats = []
    feats.append(meta_proba)                                                          # 6
    feats.append(np.eye(N_CLASSES)[meta_pred].astype(np.float32))                    # 6
    feats.append(np.eye(N_CLASSES)[anchor_pred].astype(np.float32))                  # 6
    src_at_anchor = np.zeros((n, K), dtype=np.float32)
    src_max_margin = np.zeros((n, K), dtype=np.float32)
    for k, p in enumerate(proba_per_source):
        src_at_anchor[:, k] = p[np.arange(n), anchor_pred]
        sp = np.sort(p, axis=1)
        src_max_margin[:, k] = (sp[:, -1] - sp[:, -2]).astype(np.float32)
    feats.append(src_at_anchor)
    feats.append(src_max_margin)
    agree = np.zeros((n, K), dtype=np.float32)
    for k, p in enumerate(proba_per_source):
        agree[:, k] = (p.argmax(axis=1) == anchor_pred).astype(np.float32)
    feats.append(agree)
    feats.append((meta_pred == anchor_pred).astype(np.float32).reshape(-1, 1))
    feats.append(anchor_pred.astype(np.float32).reshape(-1, 1))
    feats.append(meta_pred.astype(np.float32).reshape(-1, 1))
    return np.concatenate(feats, axis=1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_submission(data_dir: Path, file_ids: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    sample = load_sample_submission(data_dir)
    df = pd.DataFrame({"Id": file_ids.astype(int), "Label": predictions.astype(int)})
    df = df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    if df["Label"].isna().any():
        raise ValueError("Some test rows did not receive predictions")
    df["Label"] = df["Label"].astype(int)
    return df


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # === Load train OOF ===
    fold_df = pd.read_csv(args.fold_file)
    train_file_ids = fold_df["file_id"].astype(int).to_numpy()
    train_users = fold_df["user_id"].astype(str).to_numpy()
    train_labels = fold_df["label"].astype(int).to_numpy()

    train_sources = []
    test_sources = []
    test_file_ids = None
    test_users = None

    for name, oof_path, test_path in zip(args.source_names, args.source_oofs, args.source_test_probas):
        oof = load_oof(Path(oof_path))
        test = load_test(Path(test_path))
        # Align OOF to train_file_ids
        train_proba = align_to(train_file_ids, oof)
        train_sources.append({"name": name, "proba": train_proba})
        argmax_pred = oof["classes"][train_proba.argmax(axis=1)]
        f1 = f1_score(train_labels, argmax_pred, average="macro")
        print(f"  TRAIN {name}: shape {train_proba.shape}, OOF base macro-F1 = {f1:.6f}")

        if test_file_ids is None:
            test_file_ids = test["file_id"]
            test_users = test["user_id"] if test.get("user_id") is not None else np.array([f"User_{i % 40 + 61:03d}" for i in range(len(test_file_ids))])
        test_proba = align_to(test_file_ids, test)
        test_sources.append({"name": name, "proba": test_proba})
        print(f"  TEST  {name}: shape {test_proba.shape}")

    # === Build meta features for TRAIN ===
    train_proba_list = [s["proba"] for s in train_sources]
    test_proba_list = [s["proba"] for s in test_sources]

    # User priors: compute from train, look up by user_id for test
    user_prior_table: dict[str, np.ndarray] = {}
    for u in np.unique(train_users):
        mask = train_users == u
        prior = np.bincount(train_labels[mask], minlength=N_CLASSES) / max(1, mask.sum())
        user_prior_table[str(u)] = prior.astype(np.float32)
    # For unseen test users, use overall train distribution
    overall_prior = np.bincount(train_labels, minlength=N_CLASSES) / len(train_labels)
    for u in np.unique(test_users):
        if str(u) not in user_prior_table:
            user_prior_table[str(u)] = overall_prior.astype(np.float32)

    X_train = build_meta_features(train_proba_list, train_labels, train_file_ids, train_users, user_prior_table)
    X_test = build_meta_features(test_proba_list, None, test_file_ids, test_users, user_prior_table)
    print(f"\nMeta features: train {X_train.shape}, test {X_test.shape}")

    # === Train ONE meta-learner on all OOF data ===
    meta_model = LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=63,
        reg_alpha=0.05,
        reg_lambda=0.5,
        min_child_samples=20,
        random_state=args.seed,
        verbose=-1,
    )
    meta_model.fit(X_train, train_labels)
    meta_test_proba = meta_model.predict_proba(X_test)
    meta_train_proba = meta_model.predict_proba(X_train)
    train_macro = f1_score(train_labels, meta_train_proba.argmax(axis=1), average="macro")
    print(f"Meta-learner TRAIN macro-F1 (on all data, no holdout): {train_macro:.6f}")
    print(f"Meta-learner test predictions shape: {meta_test_proba.shape}")

    # === Optionally apply Viterbi smoothing to meta_test_proba ===
    if args.apply_viterbi and args.meta_metrics.exists():
        metrics = json.loads(args.meta_metrics.read_text())
        fold_params = metrics["fold_params"]
        # Median fold params for global Viterbi
        alphas = sorted(float(p["alpha"]) for p in fold_params.values())
        betas = sorted(float(p["beta"]) for p in fold_params.values())
        median_alpha = alphas[len(alphas) // 2]
        median_beta = betas[len(betas) // 2]
        # Median class weights per class
        cw_arr = np.array([p["class_weights"] for p in fold_params.values()])
        median_cw = np.median(cw_arr, axis=0).astype(float)
        print(f"\nViterbi smoothing on test meta_proba: alpha={median_alpha}, beta={median_beta}")
        print(f"  median class_weights: {np.round(median_cw, 3).tolist()}")
        classes = np.arange(N_CLASSES, dtype=np.int64)
        # Train transition matrix from train labels using train fold structure
        transition, start = estimate_transition_model(
            y=train_labels,
            classes=classes,
            file_ids=train_file_ids,
            user_ids=train_users,
            alpha=median_alpha,
        )
        test_pred_viterbi = viterbi_predict_by_user(
            proba=meta_test_proba,
            classes=classes,
            file_ids=test_file_ids,
            user_ids=test_users,
            class_weights=median_cw,
            transition=transition,
            start=start,
            beta=median_beta,
            stay_bonus=0.0,
        )
        meta_test_pred = test_pred_viterbi
        # Also apply Viterbi on TRAIN proba (for diagnostic)
        train_pred_viterbi = viterbi_predict_by_user(
            proba=meta_train_proba,
            classes=classes,
            file_ids=train_file_ids,
            user_ids=train_users,
            class_weights=median_cw,
            transition=transition,
            start=start,
            beta=median_beta,
            stay_bonus=0.0,
        )
        train_viterbi_f1 = f1_score(train_labels, train_pred_viterbi, average="macro")
        print(f"  Meta-learner TRAIN Viterbi macro-F1: {train_viterbi_f1:.6f}")
    else:
        meta_test_pred = meta_test_proba.argmax(axis=1)

    # === Load anchor (centered_meta_viterbi predictions on test) ===
    anchor_df = pd.read_csv(args.anchor_csv)
    anchor_idx = {int(i): k for k, i in enumerate(anchor_df["Id"])}
    anchor_pred = np.array([anchor_df["Label"].iloc[anchor_idx[int(f)]] for f in test_file_ids], dtype=int)
    print(f"Anchor ({args.anchor_csv}) test label counts: {pd.Series(anchor_pred).value_counts().sort_index().to_dict()}")

    # === Apply trained hybrid recovery 2.0 ===
    import joblib
    classifier = joblib.load(args.hybrid_classifier)
    rules = json.loads(args.hybrid_rules.read_text())["rules"]

    X_hybrid_test = build_hybrid_features(meta_test_proba, meta_test_pred, anchor_pred, test_proba_list)
    switch_proba = classifier.predict_proba(X_hybrid_test)[:, 1]

    final_pred = meta_test_pred.copy()
    n_switches = 0
    for trans, info in rules.items():
        meta_c, anchor_c = trans.split("->")
        meta_c, anchor_c = int(meta_c), int(anchor_c)
        thresh = info["threshold"]
        mask = (meta_test_pred == meta_c) & (anchor_pred == anchor_c) & (switch_proba >= thresh)
        n_switches += int(mask.sum())
        final_pred[mask] = anchor_c
    print(f"\nHybrid recovery 2.0 switched {n_switches} rows (out of {len(final_pred)})")
    print(f"Meta-only label counts: {pd.Series(meta_test_pred).value_counts().sort_index().to_dict()}")
    print(f"Final label counts: {pd.Series(final_pred).value_counts().sort_index().to_dict()}")

    # === Write CSV ===
    submission = merge_submission(args.data_dir, test_file_ids, final_pred)
    submission.to_csv(args.output, index=False)
    digest = sha256_file(args.output)
    print(f"\nWrote {args.output}")
    print(f"SHA256: {digest}")
    label_counts = submission["Label"].value_counts().sort_index().to_dict()
    print(f"Label counts: {label_counts}")

    # === Save metadata ===
    meta_path = Path(str(args.output).replace(".csv", "_metadata.json"))
    metadata = {
        "output_csv": str(args.output),
        "sha256": digest,
        "sources": list(zip(args.source_names, args.source_oofs, args.source_test_probas)),
        "anchor_csv": str(args.anchor_csv),
        "rules_applied": list(rules.keys()),
        "n_switches": n_switches,
        "label_counts": {int(k): int(v) for k, v in label_counts.items()},
        "meta_oof_macro_f1_internal": float(train_macro),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
