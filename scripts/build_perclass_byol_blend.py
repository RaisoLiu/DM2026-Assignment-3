#!/usr/bin/env python3
"""Per-class adaptive BYOL blend.

For each class K, blend BYOL probas with the existing weighted ensemble probas
ONLY if BYOL's OOF F1 on class K is competitive (≥ existing_F1 - 0.05). Otherwise
use the existing ensemble probas unchanged for that class.

Mechanism:
  blended[:, k] = (1 - w_k) * base[:, k] + w_k * byol[:, k]   if competitive
                = base[:, k]                                    otherwise

Then re-normalize across the row.

Rationale: Chronos LGBM had c2 F1 = 0.031 — catastrophic. Adding it at any
non-zero weight would collapse c2 prediction. Per-class gating bounds the damage
while still exploiting classes where BYOL is competitive (likely c0, c1, c4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-test", type=Path, default=Path("artifacts/weighted_ensemble/test.npz"))
    p.add_argument("--byol-test", type=Path, default=Path("artifacts/byol_lgbm/test_proba.npz"))
    p.add_argument("--base-oof", type=Path, default=Path("artifacts/weighted_ensemble/oof.npz"))
    p.add_argument("--byol-oof", type=Path, default=Path("artifacts/byol_lgbm/oof_proba.npz"))
    p.add_argument("--max-weight", type=float, default=0.20,
                   help="Max BYOL weight per class when competitive.")
    p.add_argument("--competitive-gap", type=float, default=0.05,
                   help="BYOL is 'competitive' on class K if F1_byol[k] >= F1_base[k] - gap.")
    p.add_argument("--output-csv", type=Path, default=Path("submissions/submission_byol_perclass_blend.csv"))
    p.add_argument("--apply-viterbi", action="store_true", default=True)
    return p.parse_args()


def align(target_fids, src_proba, src_fids):
    idx = {int(f): i for i, f in enumerate(src_fids)}
    order = np.array([idx[int(f)] for f in target_fids], dtype=int)
    return src_proba[order]


def main() -> None:
    args = parse_args()

    base_test = np.load(args.base_test, allow_pickle=True)
    byol_test = np.load(args.byol_test, allow_pickle=True)
    base_oof = np.load(args.base_oof, allow_pickle=True)
    byol_oof = np.load(args.byol_oof, allow_pickle=True)

    # OOF aligned for per-class F1 analysis
    base_o_fid = base_oof["file_id"].astype(int)
    base_o_y = base_oof["label"].astype(int)
    base_o_proba = base_oof["proba"].astype(np.float64)
    byol_o_proba_aligned = align(base_o_fid, byol_oof["proba"].astype(np.float64), byol_oof["file_id"].astype(int))

    base_o_pred = base_o_proba.argmax(axis=1)
    byol_o_pred = byol_o_proba_aligned.argmax(axis=1)
    base_f1 = f1_score(base_o_y, base_o_pred, average=None)
    byol_f1 = f1_score(base_o_y, byol_o_pred, average=None)
    n_classes = len(base_f1)

    print("Per-class OOF F1 comparison:")
    print(f"{'class':>6} {'base_F1':>10} {'byol_F1':>10} {'competitive?':>14} {'weight':>8}")
    weights = np.zeros(n_classes, dtype=np.float64)
    for k in range(n_classes):
        compete = byol_f1[k] >= base_f1[k] - args.competitive_gap
        if compete:
            # Linear scale: full max_weight if BYOL beats base, less if just within gap
            scale = float(np.clip((byol_f1[k] - (base_f1[k] - args.competitive_gap)) / max(args.competitive_gap, 1e-6), 0.0, 1.0))
            weights[k] = args.max_weight * scale
        print(f"{k:>6} {base_f1[k]:>10.4f} {byol_f1[k]:>10.4f} {str(compete):>14} {weights[k]:>8.3f}")

    # Test-side blend
    base_t_fid = base_test["file_id"].astype(int)
    base_t_uid = base_test["user_id"].astype(str)
    base_t_proba = base_test["proba"].astype(np.float64)
    byol_t_aligned = align(base_t_fid, byol_test["proba"].astype(np.float64), byol_test["file_id"].astype(int))

    # Per-class blend
    blended = np.zeros_like(base_t_proba)
    for k in range(n_classes):
        blended[:, k] = (1.0 - weights[k]) * base_t_proba[:, k] + weights[k] * byol_t_aligned[:, k]
    blended = blended / blended.sum(axis=1, keepdims=True).clip(min=1e-12)

    # OOF blend (for Viterbi tuning) and diagnostic
    oof_blend = np.zeros_like(base_o_proba)
    for k in range(n_classes):
        oof_blend[:, k] = (1.0 - weights[k]) * base_o_proba[:, k] + weights[k] * byol_o_proba_aligned[:, k]
    oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True).clip(min=1e-12)

    blend_f1 = f1_score(base_o_y, oof_blend.argmax(axis=1), average="macro")
    base_macro = f1_score(base_o_y, base_o_pred, average="macro")
    print(f"\nOOF macro-F1: base={base_macro:.4f}  per-class-blend={blend_f1:.4f}  Δ={blend_f1-base_macro:+.4f}")

    if args.apply_viterbi:
        from evaluate_sequence_smoothing import estimate_transition_model, tune_viterbi_params, viterbi_predict_by_user
        classes = np.arange(n_classes, dtype=int)
        oof_uid = base_oof["user_id"].astype(str)
        class_w = np.ones(n_classes, dtype=np.float64)
        best = tune_viterbi_params(
            proba=oof_blend, y=base_o_y, classes=classes,
            file_ids=base_o_fid, user_ids=oof_uid, class_weights=class_w,
            alpha_grid=(0.1, 0.3, 1.0, 3.0),
            beta_grid=(0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.27),
            stay_grid=(0.0,),
        )
        print(f"Viterbi: α={best['alpha']} β={best['beta']} train_F1={best['train_macro_f1']:.4f}")
        transition, start = estimate_transition_model(base_o_y, classes, base_o_fid, oof_uid, alpha=best["alpha"])
        pred = viterbi_predict_by_user(
            blended, classes, base_t_fid, base_t_uid, class_w,
            transition, start, beta=best["beta"], stay_bonus=0.0,
        )
    else:
        pred = blended.argmax(axis=1)

    df = pd.DataFrame({"Id": base_t_fid, "Label": pred.astype(int)})
    sample = pd.read_csv("data/raw/sample_submission.csv")
    df = df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    sha = hashlib.sha256(args.output_csv.read_bytes()).hexdigest()
    print(f"\nWrote {args.output_csv}")
    print(f"SHA: {sha}")
    print(f"Label counts: {df['Label'].value_counts().sort_index().to_dict()}")

    # Compare to anchor
    anchor = pd.read_csv("submissions/submission_h8_v3_sslv2_weighted.csv")
    diff = (df.set_index("Id")["Label"] != anchor.set_index("Id")["Label"]).sum()
    print(f"Diff vs anchor 0.8248: {diff} rows")

    metadata = {
        "byol_max_weight": args.max_weight,
        "competitive_gap": args.competitive_gap,
        "per_class_weights": [float(w) for w in weights],
        "base_per_class_f1": [float(x) for x in base_f1],
        "byol_per_class_f1": [float(x) for x in byol_f1],
        "oof_macro_base": float(base_macro),
        "oof_macro_blend": float(blend_f1),
        "sha256": sha,
        "diff_vs_anchor_0_8248": int(diff),
    }
    meta_path = Path(str(args.output_csv).replace(".csv", "_metadata.json"))
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
