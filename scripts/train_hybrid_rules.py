#!/usr/bin/env python3
"""Learn hybrid-recovery rules: for each row, decide whether to switch from the
meta-learner's argmax to the anchor's prediction.

Anchor at test time = today's 0.8240 hybrid CSV.

Approach: train per-(new_class, anchor_class) binary classifiers (or a single
classifier with both class IDs as features) on cross-validated OOF data. The
classifier predicts P(switching is better than keeping new), based on per-row
soft-proba features.

Output: rule table saved as JSON (and a small LGBM that can be invoked at test time).
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


N_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned hybrid recovery rules.")
    parser.add_argument(
        "--meta-oof",
        type=Path,
        default=Path("artifacts/meta_stacker/oof_meta.npz"),
    )
    parser.add_argument(
        "--anchor-csv",
        type=Path,
        default=Path("submissions/submission_ssl_hybrid_recover.csv"),
    )
    parser.add_argument(
        "--source-oofs",
        nargs="+",
        required=True,
        help="OOF npz files of the architecture-level models (used for per-row features).",
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hybrid_rules"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--anchor-train-labels",
        type=Path,
        default=None,
        help="If anchor CSV maps only to test, we cannot directly evaluate. We instead simulate by using "
        "the 0.8240 baseline predictions on the TRAIN OOF (computed via separate analysis).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.read_csv(args.fold_file)
    file_ids = fold_df["file_id"].astype(int).to_numpy()
    labels = fold_df["label"].astype(int).to_numpy()
    fold_ids = fold_df["fold"].astype(int).to_numpy()
    n = len(fold_df)

    meta = np.load(args.meta_oof, allow_pickle=True)
    # Align meta to fold_df
    idx = {int(f): i for i, f in enumerate(meta["file_id"])}
    order = np.array([idx[int(f)] for f in file_ids], dtype=int)
    meta_proba = meta["proba"][order].astype(float)
    meta_pred = meta_proba.argmax(axis=1)

    # The anchor here is today's 0.8240 hybrid. The anchor is defined on TEST rows, not TRAIN.
    # For the TRAIN OOF (which we need to train the rule on), we use a proxy: yesterday's
    # centered-meta Viterbi predictions on TRAIN ('pred' field of the OOF file).
    # This is the closest analog of "anchor" for supervised rule training.
    cm_pred_path = Path("artifacts/sequence_smoothing_centered_meta_round2_best/centered_meta_viterbi_predictions.npz")
    cm_pred_npz = np.load(cm_pred_path, allow_pickle=True)
    cm_pred_idx = {int(f): i for i, f in enumerate(cm_pred_npz["file_id"])}
    anchor_order = np.array([cm_pred_idx[int(f)] for f in file_ids], dtype=int)
    anchor_pred = cm_pred_npz["pred"][anchor_order].astype(int)
    print(f"Anchor (proxy = centered-meta Viterbi pred) loaded for {len(anchor_pred)} rows.")

    # Load source OOFs for per-row soft-proba features
    sources = []
    for path in args.source_oofs:
        d = np.load(Path(path), allow_pickle=True)
        idx2 = {int(f): i for i, f in enumerate(d["file_id"])}
        ord2 = np.array([idx2[int(f)] for f in file_ids], dtype=int)
        sources.append(d["proba"][ord2].astype(float))
        argmax_pred = d["classes"][d["proba"][ord2].argmax(axis=1)]
        f1 = f1_score(labels, argmax_pred, average="macro")
        print(f"  {path}: shape {d['proba'][ord2].shape}, OOF base macro-F1 = {f1:.6f}")

    # Build features per row
    # Features:
    #  - meta-learner soft probas (6)
    #  - meta-learner argmax class one-hot (6)
    #  - anchor argmax class one-hot (6)
    #  - per-source soft proba for the anchor's class (one per source)
    #  - per-source max-margin (one per source)
    #  - agreement count among sources for anchor's class
    K = len(sources)
    rows = []
    rows.append(meta_proba)                                            # 6
    rows.append(np.eye(N_CLASSES)[meta_pred].astype(np.float32))      # 6
    rows.append(np.eye(N_CLASSES)[anchor_pred].astype(np.float32))    # 6
    # Per-source proba for anchor class
    src_at_anchor = np.zeros((n, K), dtype=np.float32)
    src_max_margin = np.zeros((n, K), dtype=np.float32)
    for k, s in enumerate(sources):
        src_at_anchor[:, k] = s[np.arange(n), anchor_pred]
        sorted_s = np.sort(s, axis=1)
        src_max_margin[:, k] = (sorted_s[:, -1] - sorted_s[:, -2]).astype(np.float32)
    rows.append(src_at_anchor)                                         # K
    rows.append(src_max_margin)                                        # K
    # Source argmax agreement with anchor
    agree_with_anchor = np.zeros((n, K), dtype=np.float32)
    for k, s in enumerate(sources):
        agree_with_anchor[:, k] = (s.argmax(axis=1) == anchor_pred).astype(np.float32)
    rows.append(agree_with_anchor)                                     # K
    rows.append((meta_pred == anchor_pred).astype(np.float32).reshape(-1, 1))  # 1
    # Also: anchor class id and meta class id as scalar features
    rows.append(anchor_pred.astype(np.float32).reshape(-1, 1))         # 1
    rows.append(meta_pred.astype(np.float32).reshape(-1, 1))           # 1
    X = np.concatenate(rows, axis=1)
    print(f"Feature matrix: {X.shape}")

    # Target: switch_is_better = (anchor_pred == true_label) AND (meta_pred != true_label)
    # i.e., whenever the anchor is right and meta is wrong, we should switch.
    target = ((anchor_pred == labels) & (meta_pred != labels)).astype(int)

    # Also consider: switch_is_worse = (meta_pred == true_label) AND (anchor_pred != true_label)
    # We won't directly train against this, but we'll evaluate.
    n_switch_better = int(target.sum())
    n_keep_better = int(((meta_pred == labels) & (anchor_pred != labels)).sum())
    n_both_correct = int(((meta_pred == labels) & (anchor_pred == labels)).sum())
    n_both_wrong = int(((meta_pred != labels) & (anchor_pred != labels)).sum())
    print(f"\nOOF rows: {n}")
    print(f"  switch_better (anchor right, meta wrong): {n_switch_better}")
    print(f"  keep_better (meta right, anchor wrong): {n_keep_better}")
    print(f"  both correct: {n_both_correct}")
    print(f"  both wrong: {n_both_wrong}")
    print(f"  net potential if we always switched on the disagreement: {n_switch_better - n_keep_better}")

    # Train binary classifier with 5-fold CV
    oof_switch_proba = np.zeros(n, dtype=np.float32)
    for f in sorted(set(fold_ids.tolist())):
        valid = fold_ids == f
        train = ~valid
        if target[train].sum() == 0 or (1 - target[train]).sum() == 0:
            # No positive or negative class — skip
            oof_switch_proba[valid] = target[train].mean()
            continue
        model = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=args.seed,
            verbose=-1,
        )
        model.fit(X[train], target[train])
        oof_switch_proba[valid] = model.predict_proba(X[valid])[:, 1]

    # Find optimal threshold per (meta_pred, anchor_pred) transition
    rules = {}
    final_pred = meta_pred.copy()
    for meta_c in range(N_CLASSES):
        for anchor_c in range(N_CLASSES):
            if meta_c == anchor_c:
                continue
            mask = (meta_pred == meta_c) & (anchor_pred == anchor_c)
            if mask.sum() == 0:
                continue
            best_thresh = None
            best_gain = -1.0
            best_macro = None
            # Search over thresholds
            for thresh in np.linspace(0.10, 0.90, 17):
                hypothetical_pred = meta_pred.copy()
                switch_mask = mask & (oof_switch_proba >= thresh)
                hypothetical_pred[switch_mask] = anchor_c
                macro = f1_score(labels, hypothetical_pred, average="macro")
                # Compare to no switching (meta only)
                base_macro = f1_score(labels, meta_pred, average="macro")
                gain = macro - base_macro
                if gain > best_gain:
                    best_gain = gain
                    best_thresh = float(thresh)
                    best_macro = float(macro)
            if best_gain > 0:
                rules[f"{meta_c}->{anchor_c}"] = {
                    "threshold": best_thresh,
                    "gain": float(best_gain),
                    "macro_at_threshold": best_macro,
                    "n_candidate": int(mask.sum()),
                    "n_switched": int((mask & (oof_switch_proba >= best_thresh)).sum()),
                }
                # Apply
                switch_mask = mask & (oof_switch_proba >= best_thresh)
                final_pred[switch_mask] = anchor_c

    print(f"\nLearned {len(rules)} positive-gain transitions:")
    for trans, info in sorted(rules.items(), key=lambda kv: -kv[1]["gain"]):
        print(f"  {trans}: threshold={info['threshold']:.3f}, gain={info['gain']:+.6f}, switched={info['n_switched']}/{info['n_candidate']}")

    # Final OOF metrics
    base_macro = f1_score(labels, meta_pred, average="macro")
    new_macro = f1_score(labels, final_pred, average="macro")
    print(f"\nBaseline meta-learner OOF: {base_macro:.6f}")
    print(f"With hybrid rules OOF: {new_macro:.6f}")
    print(f"Δ from hybrid rules: {new_macro - base_macro:+.6f}")

    # Save
    out_rules = args.output_dir / "rules.json"
    out_rules.write_text(json.dumps({
        "rules": rules,
        "n_features": int(X.shape[1]),
        "anchor_proxy_path": str(cm_pred_path),
        "baseline_oof_macro_f1": float(base_macro),
        "with_rules_oof_macro_f1": float(new_macro),
    }, indent=2))
    print(f"Wrote {out_rules}")

    # Save the switch proba and a small "applier" predictor (use the proba directly + threshold table)
    np.savez_compressed(
        args.output_dir / "switch_proba_oof.npz",
        switch_proba=oof_switch_proba.astype(np.float32),
        meta_pred=meta_pred.astype(np.int64),
        anchor_pred=anchor_pred.astype(np.int64),
        final_pred=final_pred.astype(np.int64),
        file_id=file_ids.astype(np.int64),
        label=labels.astype(np.int64),
    )

    # Also re-train one final binary classifier on ALL OOF data, for test-time inference
    model_all = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=args.seed,
        verbose=-1,
    )
    if target.sum() > 0 and (1 - target).sum() > 0:
        model_all.fit(X, target)
        import joblib
        joblib.dump(model_all, args.output_dir / "switch_classifier.joblib")
        print(f"Wrote switch_classifier.joblib")
    else:
        print("Skipping final classifier (no positive samples)")


if __name__ == "__main__":
    main()
