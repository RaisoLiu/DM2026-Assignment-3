#!/usr/bin/env python3
"""Fallback: blend BYOL LGBM probas at a fixed LOW weight on top of the
existing 4-source weighted-ensemble test probas, then Viterbi + emit submission.

Rationale: search-optimal blend can collapse on a rare class if the new source
has a class-2 like 0.03 OOF F1. A fixed 5% weight bounds the damage while still
testing whether BYOL adds any signal at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-test", type=Path, default=Path("artifacts/weighted_ensemble/test.npz"),
                   help="Existing 4-source test probas (pre-Viterbi).")
    p.add_argument("--byol-test", type=Path, default=Path("artifacts/byol_lgbm/test_proba.npz"))
    p.add_argument("--byol-weight", type=float, default=0.05)
    p.add_argument("--oof-base", type=Path, default=Path("artifacts/weighted_ensemble/oof.npz"))
    p.add_argument("--oof-byol", type=Path, default=Path("artifacts/byol_lgbm/oof_proba.npz"))
    p.add_argument("--output-csv", type=Path, default=Path("submissions/submission_byol_low5_blend.csv"))
    p.add_argument("--apply-viterbi", action="store_true", default=True)
    return p.parse_args()


def align_proba(target_fids: np.ndarray, src_proba: np.ndarray, src_fids: np.ndarray) -> np.ndarray:
    idx = {int(f): i for i, f in enumerate(src_fids)}
    order = np.array([idx[int(f)] for f in target_fids], dtype=int)
    return src_proba[order]


def main() -> None:
    args = parse_args()

    base = np.load(args.base_test, allow_pickle=True)
    byol = np.load(args.byol_test, allow_pickle=True)

    base_proba = base["proba"].astype(np.float64)
    base_fid = base["file_id"].astype(int)
    base_uid = base["user_id"].astype(str) if "user_id" in base.files else None

    byol_proba_aligned = align_proba(base_fid, byol["proba"].astype(np.float64), byol["file_id"].astype(int))

    # Blend at fixed low weight
    w = float(args.byol_weight)
    blended = (1.0 - w) * base_proba + w * byol_proba_aligned
    blended = blended / blended.sum(axis=1, keepdims=True).clip(min=1e-12)
    print(f"Blend weight: base={1-w:.3f}, byol={w:.3f}")

    # OOF comparison for diagnostics
    from sklearn.metrics import f1_score, classification_report
    oof_base = np.load(args.oof_base, allow_pickle=True)
    oof_byol = np.load(args.oof_byol, allow_pickle=True)
    base_o_proba = oof_base["proba"].astype(np.float64)
    base_o_y = oof_base["label"].astype(int)
    base_o_fid = oof_base["file_id"].astype(int)
    byol_o_proba_aligned = align_proba(base_o_fid, oof_byol["proba"].astype(np.float64), oof_byol["file_id"].astype(int))
    oof_blend = (1.0 - w) * base_o_proba + w * byol_o_proba_aligned
    oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True).clip(min=1e-12)
    f1 = f1_score(base_o_y, oof_blend.argmax(axis=1), average="macro")
    f1_base = f1_score(base_o_y, base_o_proba.argmax(axis=1), average="macro")
    print(f"OOF macro-F1: base={f1_base:.4f}  blend(w={w})={f1:.4f}  Δ={f1-f1_base:+.4f}")
    print("Blend per-class:")
    print(classification_report(base_o_y, oof_blend.argmax(axis=1), digits=4))

    if args.apply_viterbi:
        from evaluate_sequence_smoothing import estimate_transition_model, tune_viterbi_params, viterbi_predict_by_user
        classes = np.arange(6, dtype=int)
        oof_uid = oof_base["user_id"].astype(str)
        class_w = np.ones(6, dtype=np.float64)
        best = tune_viterbi_params(
            proba=oof_blend, y=base_o_y, classes=classes,
            file_ids=base_o_fid, user_ids=oof_uid, class_weights=class_w,
            alpha_grid=(0.1, 0.3, 1.0, 3.0),
            beta_grid=(0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.27),
            stay_grid=(0.0,),
        )
        print(f"Viterbi: α={best['alpha']} β={best['beta']} train_F1={best['train_macro_f1']:.4f}")
        transition, start = estimate_transition_model(base_o_y, classes, base_o_fid, oof_uid, alpha=best["alpha"])
        if base_uid is None:
            raise SystemExit("Need user_id in test npz")
        pred = viterbi_predict_by_user(
            blended, classes, base_fid, base_uid, class_w,
            transition, start, beta=best["beta"], stay_bonus=0.0,
        )
    else:
        pred = blended.argmax(axis=1)

    df = pd.DataFrame({"Id": base_fid, "Label": pred.astype(int)})
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


if __name__ == "__main__":
    main()
