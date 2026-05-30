#!/usr/bin/env python3
"""HOS-12 evaluator: given an OOF probability matrix, restrict to the 12
fresh holdout users (`holdout12_seed2027.csv`) and compute macro-F1 with
fold-fair Viterbi tuned on the remaining 48 users.

Usage:
    from scripts.eval_hos12 import eval_hos12_from_proba

    score = eval_hos12_from_proba(
        proba=oof['proba'],
        file_ids=oof['file_id'],
        labels=oof['label'],
        user_ids=oof['user_id'],
        classes=oof['classes'],
        class_weights=cw,        # length-6 vector
        alpha=1.0, beta=0.12,    # Viterbi params
    )
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sequence_smoothing import (  # noqa: E402
    estimate_transition_model,
    viterbi_predict_by_user,
)


def _load_hos12() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts/folds/holdout12_seed2027.csv")


def eval_hos12_from_proba(
    proba: np.ndarray,
    file_ids: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    classes: np.ndarray,
    class_weights: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.12,
    return_per_class: bool = True,
):
    """Compute HOS-12 macro-F1 with Viterbi smoothing.

    Tunes transition probabilities on the 48 non-HOS-12 users (fold-fair).
    """
    proba = np.asarray(proba, dtype=float)
    file_ids = np.asarray(file_ids, dtype=int)
    labels = np.asarray(labels, dtype=int)
    user_ids = np.asarray(user_ids, dtype=str)
    classes = np.asarray(classes, dtype=int)

    if class_weights is None:
        class_weights = np.ones(len(classes), dtype=float)
    else:
        class_weights = np.asarray(class_weights, dtype=float)

    hos = _load_hos12()
    hos_fids = set(int(f) for f in hos["file_id"])

    hos_mask = np.array([int(f) in hos_fids for f in file_ids])
    train_mask = ~hos_mask
    if not hos_mask.any():
        raise ValueError("No file_ids in HOS-12.")

    transition, start = estimate_transition_model(
        y=labels[train_mask],
        classes=classes,
        file_ids=file_ids[train_mask],
        user_ids=user_ids[train_mask],
        alpha=alpha,
    )
    pred_hos = viterbi_predict_by_user(
        proba=proba[hos_mask],
        classes=classes,
        file_ids=file_ids[hos_mask],
        user_ids=user_ids[hos_mask],
        class_weights=class_weights,
        transition=transition,
        start=start,
        beta=beta,
        stay_bonus=0.0,
    )
    y_hos = labels[hos_mask]
    macro = float(f1_score(y_hos, pred_hos, average="macro"))
    out = {"hos12_macro_f1": macro, "n_rows": int(hos_mask.sum())}
    if return_per_class:
        per = f1_score(y_hos, pred_hos, average=None, labels=list(range(len(classes))))
        out["per_class_f1"] = {int(c): float(per[i]) for i, c in enumerate(classes)}
        # Confusion
        out["pred_counts"] = pd.Series(pred_hos).value_counts().sort_index().to_dict()
        out["label_counts"] = pd.Series(y_hos).value_counts().sort_index().to_dict()
    return out


def eval_hos12_from_argmax(pred: np.ndarray, file_ids: np.ndarray, labels: np.ndarray):
    """Evaluate HOS-12 macro-F1 directly from class-argmax predictions (no Viterbi)."""
    pred = np.asarray(pred, dtype=int)
    file_ids = np.asarray(file_ids, dtype=int)
    labels = np.asarray(labels, dtype=int)
    hos = _load_hos12()
    hos_fids = set(int(f) for f in hos["file_id"])
    hos_mask = np.array([int(f) in hos_fids for f in file_ids])
    y = labels[hos_mask]
    p = pred[hos_mask]
    macro = float(f1_score(y, p, average="macro"))
    per = f1_score(y, p, average=None, labels=list(range(6)))
    return {
        "hos12_macro_f1": macro,
        "per_class_f1": {int(c): float(per[i]) for i, c in enumerate(range(6))},
        "pred_counts": pd.Series(p).value_counts().sort_index().to_dict(),
        "label_counts": pd.Series(y).value_counts().sort_index().to_dict(),
        "n_rows": int(hos_mask.sum()),
    }


if __name__ == "__main__":
    # Smoke test: compute baseline weighted ensemble HOS-12 macro
    we = np.load(ROOT / "artifacts/weighted_ensemble/oof.npz", allow_pickle=True)
    cm = np.load(ROOT / "artifacts/blend_search/oof_blend_centered_meta_round2_best.npz", allow_pickle=True)
    cw = cm["class_weights"].astype(float) if "class_weights" in cm.files else np.ones(6)
    r = eval_hos12_from_proba(
        proba=we["proba"], file_ids=we["file_id"], labels=we["label"],
        user_ids=we["user_id"], classes=we["classes"], class_weights=cw,
        alpha=1.0, beta=0.12,
    )
    print(f"Baseline weighted ensemble HOS-12 macro: {r['hos12_macro_f1']:.4f}")
    print(f"Per-class: {r['per_class_f1']}")
