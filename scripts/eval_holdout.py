#!/usr/bin/env python3
"""Apply the pre-registered 4-condition gate.

Reads:
  - artifacts/decision/criterion.json (pre-registered rules + baselines)
  - The new blend OOF (lambda, alpha, beta encoded in the npz)
  - InceptionTime predictions on the 8 held-out users
  - The existing centered-meta predictions on the 8 held-out users

Computes:
  - new_blend OOF macro-F1 (apples-to-apples on the 52-user 5-fold seed-2026)
  - new_blend HOS macro-F1 (with Viterbi if applicable)
  - bootstrap CI on (new_blend - baseline) OOF delta
  - per-class HOS F1 for class 2 / class 5

Writes:
  - artifacts/decision/holdout_decision_report.json with upload_decision
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sequence_smoothing import (
    estimate_transition_model,
    tune_viterbi_params,
    viterbi_predict_by_user,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply pre-registered 4-condition gate.")
    parser.add_argument("--criterion", type=Path, default=Path("artifacts/decision/criterion.json"))
    parser.add_argument(
        "--new-blend-oof",
        type=Path,
        default=Path("artifacts/inception_blend/avg_v2/oof_blend_centered_meta_with_inception.npz"),
        help="OOF blend npz from build_inception_blend_submission.py.",
    )
    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=Path("artifacts/sequence_smoothing_centered_meta_round2_best/centered_meta_viterbi_predictions.npz"),
        help="Existing centered-meta Viterbi predictions (for OOF baseline + HOS baseline).",
    )
    parser.add_argument(
        "--inception-holdout",
        type=Path,
        default=Path("artifacts/inception_full/m52_holdout/holdout_proba_seed2026.npz"),
        help="InceptionTime predictions on the 8 held-out users.",
    )
    parser.add_argument(
        "--centered-meta-oof-source",
        type=Path,
        default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"),
        help="Source for centered-meta proba on HOS rows.",
    )
    parser.add_argument(
        "--holdout-file",
        type=Path,
        default=Path("artifacts/folds/holdout8_seed2026.csv"),
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/decision/holdout_decision_report.json"),
    )
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--alpha-grid",
        default="0.1,0.3,1.0,3.0",
    )
    parser.add_argument(
        "--beta-grid",
        default="0.0,0.02,0.05,0.08,0.12,0.18,0.27,0.40,0.60,0.90,1.30",
    )
    return parser.parse_args()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    f1s = f1_score(y_true, y_pred, average=None, labels=list(range(6)))
    return {int(k): float(v) for k, v in enumerate(f1s)}


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    criterion = json.loads(args.criterion.read_text())
    rules = criterion["rules"]
    baselines = criterion["baselines"]

    print("Pre-registered criterion (rules SHA: " + criterion.get("rules_sha256", "?") + ")")
    print(f"  rule_1 OOF threshold: >= {rules['rule_1_oof_lift']['threshold']:.6f}")
    print(f"  rule_2 HOS threshold: >= {rules['rule_2_hos_lift']['threshold']:.6f}")
    print(
        f"  rule_4 HOS class-2 floor: >= "
        f"{rules['rule_4_no_rare_regression']['class_2_min_f1'] - rules['rule_4_no_rare_regression']['class_2_tolerance']:.4f}"
    )
    print(
        f"  rule_4 HOS class-5 floor: >= "
        f"{rules['rule_4_no_rare_regression']['class_5_min_f1'] - rules['rule_4_no_rare_regression']['class_5_tolerance']:.4f}"
    )

    # --- Compute new blend OOF macro-F1 (52-user, fold-fair Viterbi) ---
    new_blend = np.load(args.new_blend_oof, allow_pickle=True)
    new_proba = new_blend["proba"].astype(float)
    new_labels = new_blend["label"].astype(int)
    new_file_id = new_blend["file_id"].astype(int)
    new_user_id = new_blend["user_id"].astype(str)
    new_classes = new_blend["classes"].astype(int)
    new_class_weights = new_blend["class_weights"].astype(float)
    new_alpha = float(new_blend["alpha"]) if "alpha" in new_blend.files else 1.0
    new_beta = float(new_blend["beta"]) if "beta" in new_blend.files else 0.12

    folds = pd.read_csv(args.fold_file)
    fold_lookup = dict(zip(folds["file_id"].astype(int), folds["fold"].astype(int)))
    fold_ids_new = np.array([fold_lookup[int(f)] for f in new_file_id], dtype=int)
    n_folds = int(fold_ids_new.max())

    alpha_grid = tuple(float(x) for x in args.alpha_grid.split(","))
    beta_grid = tuple(float(x) for x in args.beta_grid.split(","))

    # Compute fold-fair Viterbi macro-F1 on the 52-user CV for the new blend
    fold_macros = []
    for k in range(1, n_folds + 1):
        train_mask = fold_ids_new != k
        valid_mask = fold_ids_new == k
        if not train_mask.any() or not valid_mask.any():
            continue
        params_k = tune_viterbi_params(
            proba=new_proba[train_mask],
            y=new_labels[train_mask],
            classes=new_classes,
            file_ids=new_file_id[train_mask],
            user_ids=new_user_id[train_mask],
            class_weights=new_class_weights,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            stay_grid=(0.0,),
        )
        tr_k, st_k = estimate_transition_model(
            y=new_labels[train_mask],
            classes=new_classes,
            file_ids=new_file_id[train_mask],
            user_ids=new_user_id[train_mask],
            alpha=params_k["alpha"],
        )
        pred_k = viterbi_predict_by_user(
            proba=new_proba[valid_mask],
            classes=new_classes,
            file_ids=new_file_id[valid_mask],
            user_ids=new_user_id[valid_mask],
            class_weights=new_class_weights,
            transition=tr_k,
            start=st_k,
            beta=params_k["beta"],
            stay_bonus=params_k["stay_bonus"],
        )
        fold_macros.append(float(f1_score(new_labels[valid_mask], pred_k, average="macro")))
    new_oof_viterbi = float(np.mean(fold_macros))
    print(f"\nNew blend fold-fair Viterbi OOF (52-user): {new_oof_viterbi:.6f}")

    # --- Baseline OOF: same fold structure, lambda=0 (centered-meta only) ---
    base_proba = (1.0 - 0.0) * new_proba  # lambda=0 means we'd subtract inception contribution
    # Better: re-derive baseline by removing inception contribution
    # If new_proba = (1-lam)*cm_proba + lam*inc_proba, then cm_proba = (new_proba - lam*inc_proba) / (1-lam)
    lam = float(new_blend.get("lambda_inception", 0.0)) if "lambda_inception" in new_blend.files else 0.0

    # Use the existing centered-meta source npz directly for baseline
    cm_src = np.load(args.centered_meta_oof_source, allow_pickle=True)
    cm_index = {int(f): i for i, f in enumerate(cm_src["file_id"])}
    order_cm = np.array([cm_index[int(f)] for f in new_file_id], dtype=int)
    cm_proba_aligned = cm_src["proba"][order_cm].astype(float)
    cm_class_weights = cm_src["class_weights"].astype(float)
    # Tune fold-fair Viterbi on the centered-meta-only baseline (apples-to-apples)
    fold_macros_base = []
    for k in range(1, n_folds + 1):
        train_mask = fold_ids_new != k
        valid_mask = fold_ids_new == k
        params_k = tune_viterbi_params(
            proba=cm_proba_aligned[train_mask],
            y=new_labels[train_mask],
            classes=new_classes,
            file_ids=new_file_id[train_mask],
            user_ids=new_user_id[train_mask],
            class_weights=cm_class_weights,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            stay_grid=(0.0,),
        )
        tr_k, st_k = estimate_transition_model(
            y=new_labels[train_mask],
            classes=new_classes,
            file_ids=new_file_id[train_mask],
            user_ids=new_user_id[train_mask],
            alpha=params_k["alpha"],
        )
        pred_k = viterbi_predict_by_user(
            proba=cm_proba_aligned[valid_mask],
            classes=new_classes,
            file_ids=new_file_id[valid_mask],
            user_ids=new_user_id[valid_mask],
            class_weights=cm_class_weights,
            transition=tr_k,
            start=st_k,
            beta=params_k["beta"],
            stay_bonus=params_k["stay_bonus"],
        )
        fold_macros_base.append(float(f1_score(new_labels[valid_mask], pred_k, average="macro")))
    baseline_oof_samefold = float(np.mean(fold_macros_base))
    print(f"Baseline (centered-meta only on same 52-user 5-fold): {baseline_oof_samefold:.6f}")

    delta_oof = new_oof_viterbi - baseline_oof_samefold
    print(f"Δ_OOF: {delta_oof:+.6f}")

    # --- Bootstrap on file_id-level OOF to estimate prob(new > baseline) ---
    rng = np.random.default_rng(args.seed)
    n = len(new_file_id)
    rule3_pos = 0
    # Pre-compute per-fold predictions for both new and baseline so bootstrap is cheap.
    # We use base argmax (no Viterbi) for bootstrap because Viterbi requires sequence structure
    # that breaks under random resampling. Bootstrap on argmax is a reasonable proxy.
    new_pred = new_classes[np.argmax(new_proba * new_class_weights.reshape(1, -1), axis=1)]
    base_pred = new_classes[np.argmax(cm_proba_aligned * cm_class_weights.reshape(1, -1), axis=1)]
    new_correct = (new_pred == new_labels).astype(np.int8)
    base_correct = (base_pred == new_labels).astype(np.int8)
    new_macro_full = macro_f1(new_labels, new_pred)
    base_macro_full = macro_f1(new_labels, base_pred)
    print(
        f"Argmax (no Viterbi) macro-F1: new={new_macro_full:.6f}, baseline={base_macro_full:.6f}"
    )

    boot_n_pos = 0
    for b in range(args.bootstrap_n):
        idx = rng.choice(n, size=n, replace=True)
        new_b = macro_f1(new_labels[idx], new_pred[idx])
        base_b = macro_f1(new_labels[idx], base_pred[idx])
        if new_b > base_b:
            boot_n_pos += 1
    boot_pct = boot_n_pos / args.bootstrap_n
    print(f"Bootstrap pct(new > baseline) on argmax-based macro-F1: {boot_pct:.3f}")

    # --- Compute new blend HOS macro-F1 ---
    hos_csv = pd.read_csv(args.holdout_file)
    hos_file_ids = hos_csv["file_id"].astype(int).to_numpy()
    hos_users = hos_csv["user_id"].astype(str).to_numpy()
    hos_labels = hos_csv["label"].astype(int).to_numpy()

    inception_hos = np.load(args.inception_holdout, allow_pickle=True)
    inc_index = {int(f): i for i, f in enumerate(inception_hos["file_id"])}
    inc_order = np.array([inc_index[int(f)] for f in hos_file_ids], dtype=int)
    inc_proba_hos = inception_hos["proba"][inc_order].astype(float)

    cm_index_hos = {int(f): i for i, f in enumerate(cm_src["file_id"])}
    cm_order_hos = np.array([cm_index_hos[int(f)] for f in hos_file_ids], dtype=int)
    cm_proba_hos = cm_src["proba"][cm_order_hos].astype(float)

    new_blend_hos = (1.0 - lam) * cm_proba_hos + lam * inc_proba_hos
    print(f"\nHOS lambda used: {lam}")

    # Predict HOS using a "global" Viterbi tuned on the 52-user OOF (which is the closest available
    # tuning for HOS users — they cannot be in the OOF tuning set).
    # We use the new_blend's stored alpha/beta as the canonical Viterbi parameters.
    transition, start = estimate_transition_model(
        y=new_labels,
        classes=new_classes,
        file_ids=new_file_id,
        user_ids=new_user_id,
        alpha=new_alpha,
    )
    new_pred_hos = viterbi_predict_by_user(
        proba=new_blend_hos,
        classes=new_classes,
        file_ids=hos_file_ids,
        user_ids=hos_users,
        class_weights=new_class_weights,
        transition=transition,
        start=start,
        beta=new_beta,
        stay_bonus=0.0,
    )
    new_hos_macro = macro_f1(hos_labels, new_pred_hos)
    new_hos_per_class = per_class_f1(hos_labels, new_pred_hos)
    print(f"New blend HOS macro-F1 (Viterbi-smoothed): {new_hos_macro:.6f}")
    print(f"New blend HOS per-class F1: {new_hos_per_class}")

    # Baseline HOS: existing centered-meta Viterbi predictions on HOS users
    base_pred_obj = np.load(args.baseline_oof, allow_pickle=True)
    base_pred_full = base_pred_obj["pred"].astype(int)  # fold-fair Viterbi pred (the headline 0.7693)
    base_users = base_pred_obj["user_id"].astype(str)
    base_file_ids = base_pred_obj["file_id"].astype(int)
    base_labels = base_pred_obj["label"].astype(int)
    base_index = {int(f): i for i, f in enumerate(base_file_ids)}
    base_order = np.array([base_index[int(f)] for f in hos_file_ids], dtype=int)
    base_pred_hos = base_pred_full[base_order]
    base_label_hos = base_labels[base_order]
    assert np.array_equal(base_label_hos, hos_labels), "HOS label mismatch"
    base_hos_macro = macro_f1(hos_labels, base_pred_hos)
    base_hos_per_class = per_class_f1(hos_labels, base_pred_hos)
    print(f"Baseline HOS macro-F1: {base_hos_macro:.6f}")
    print(f"Baseline HOS per-class F1: {base_hos_per_class}")

    delta_hos = new_hos_macro - base_hos_macro
    print(f"Δ_HOS: {delta_hos:+.6f}")

    # --- Apply 4 rules ---
    rule_1 = new_oof_viterbi >= rules["rule_1_oof_lift"]["threshold"]
    rule_2 = new_hos_macro >= rules["rule_2_hos_lift"]["threshold"]
    rule_3a = (delta_oof > 0) == (delta_hos > 0) and (delta_oof > 0)
    rule_3b = boot_pct >= rules["rule_3_sign_and_bootstrap"]["bootstrap_pct"]
    rule_3 = rule_3a and rule_3b
    rule_4_c2 = new_hos_per_class[2] >= (
        rules["rule_4_no_rare_regression"]["class_2_min_f1"] - rules["rule_4_no_rare_regression"]["class_2_tolerance"]
    )
    rule_4_c5 = new_hos_per_class[5] >= (
        rules["rule_4_no_rare_regression"]["class_5_min_f1"] - rules["rule_4_no_rare_regression"]["class_5_tolerance"]
    )
    rule_4 = rule_4_c2 and rule_4_c5
    pass_all = bool(rule_1 and rule_2 and rule_3 and rule_4)

    if pass_all:
        upload_decision = "new_pipeline"
        selected_csv = criterion["primary_csv_if_passes"]
    else:
        upload_decision = "fallback_0.8156"
        selected_csv = criterion["fallback_csv"]

    report = {
        "upload_decision": upload_decision,
        "selected_csv": selected_csv,
        "criterion_rules_sha": criterion.get("rules_sha256"),
        "criterion_baselines_sha": criterion.get("baselines_sha256"),
        "metrics": {
            "new_oof_viterbi": new_oof_viterbi,
            "baseline_oof_samefold": baseline_oof_samefold,
            "delta_oof": delta_oof,
            "new_hos_macro_f1": new_hos_macro,
            "baseline_hos_macro_f1": base_hos_macro,
            "delta_hos": delta_hos,
            "bootstrap_pct_new_gt_baseline": boot_pct,
            "new_oof_argmax_macro": new_macro_full,
            "baseline_oof_argmax_macro": base_macro_full,
            "new_hos_per_class_f1": new_hos_per_class,
            "baseline_hos_per_class_f1": base_hos_per_class,
            "lambda_inception": lam,
            "viterbi_alpha": new_alpha,
            "viterbi_beta": new_beta,
        },
        "rule_results": {
            "rule_1_oof_lift": {
                "pass": bool(rule_1),
                "value": new_oof_viterbi,
                "threshold": rules["rule_1_oof_lift"]["threshold"],
            },
            "rule_2_hos_lift": {
                "pass": bool(rule_2),
                "value": new_hos_macro,
                "threshold": rules["rule_2_hos_lift"]["threshold"],
            },
            "rule_3_sign_and_bootstrap": {
                "pass": bool(rule_3),
                "sign_match": bool(rule_3a),
                "bootstrap_pct": boot_pct,
                "bootstrap_threshold": rules["rule_3_sign_and_bootstrap"]["bootstrap_pct"],
            },
            "rule_4_no_rare_regression": {
                "pass": bool(rule_4),
                "class_2_pass": bool(rule_4_c2),
                "class_5_pass": bool(rule_4_c5),
                "class_2_value": new_hos_per_class[2],
                "class_5_value": new_hos_per_class[5],
            },
        },
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    print("\n=== DECISION ===")
    print(f"  rule_1: {'PASS' if rule_1 else 'FAIL'}  (OOF {new_oof_viterbi:.6f} >= {rules['rule_1_oof_lift']['threshold']:.6f})")
    print(f"  rule_2: {'PASS' if rule_2 else 'FAIL'}  (HOS {new_hos_macro:.6f} >= {rules['rule_2_hos_lift']['threshold']:.6f})")
    print(f"  rule_3: {'PASS' if rule_3 else 'FAIL'}  (signs ok={rule_3a}, bootstrap {boot_pct:.3f} >= 0.80)")
    print(
        f"  rule_4: {'PASS' if rule_4 else 'FAIL'}  (c2 {new_hos_per_class[2]:.4f}>=floor, c5 {new_hos_per_class[5]:.4f}>=floor)"
    )
    print(f"  → upload_decision = {upload_decision}")
    print(f"  → selected_csv = {selected_csv}")


if __name__ == "__main__":
    main()
