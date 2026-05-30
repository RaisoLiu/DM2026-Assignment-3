#!/usr/bin/env python3
"""Validator gate for Generator candidates.

Given a candidate CSV (test predictions) and optionally an OOF/holdout npz with
the same modeling pipeline applied to training data, emit a single-line verdict:

  APPROVE | filename | HOS-12 macro | c2 F1 | diff | label-counts | reason

Anchors used as references:
  0.8248 -> submissions/submission_h8_v3_sslv2_weighted.csv  (primary)
  0.8244 -> submissions/submission_h8_uncertain_refined.csv
  0.8240 -> submissions/submission_ssl_hybrid_recover.csv

HOS-12 reference baseline (centered-meta Viterbi fold-fair pred): 0.7849
HOS-12 fold file: artifacts/folds/holdout12_seed2027.csv (12 users, 2231 rows)

Phase-1 calibration table (see VALIDATOR report) gives observed regressors:
  -0.0062 manual_rules:   diff=14 BUT unanim_override=5,  c2 delta=+9
  -0.0093 byol_low5:      diff=193,    unanim_override=175, c2 delta=-102
  -0.0125 sslv3_5src:     diff=106,    unanim_override=43,  c2 delta=-1

Verdict thresholds (calibrated, see report):
  REJECT if any of:
    - HOS-12 macro (if available) drops > 0.005 vs anchor's HOS-12 (0.7849)
    - c2 count diff vs 0.8248 anchor (184) > 30  (byol regressor was -102)
    - diff-vs-0.8248 > 120 rows                  (byol=193, sslv3=106 -> 120 mid)
    - unanim_override > 30                       (byol=175, sslv3=43)
    - c5 count diff vs 0.8248 anchor (257) > 40  (sslv3 had c5=285, +28)
  APPROVE-WITH-CAVEAT if HOS-12 within +/-0.005 of baseline AND any of:
    - diff-vs-0.8248 in (30, 120]
    - c2 count diff in (15, 30]
    - unanim_override in (5, 30]
    - HOS-12 c2 F1 drops > 0.02 vs cm_baseline (0.3758)
  APPROVE if all thresholds clear AND HOS-12 macro >= 0.7849 (when available).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]

ANCHOR_PRIMARY = ROOT / "submissions/submission_h8_v3_sslv2_weighted.csv"  # 0.8248
ANCHOR_244 = ROOT / "submissions/submission_h8_uncertain_refined.csv"
ANCHOR_240 = ROOT / "submissions/submission_ssl_hybrid_recover.csv"
HOS12_FOLD = ROOT / "artifacts/folds/holdout12_seed2027.csv"
BASELINE_HOS12_JSON = ROOT / "artifacts/decision/baseline_hos12_v3.json"

# Anchor label-count reference (from 0.8248)
ANCHOR_LABEL_COUNTS = {0: 2809, 1: 3055, 2: 184, 3: 470, 4: 74, 5: 257}
HOS12_BASELINE_MACRO = 0.784897  # cm_v fold-fair Viterbi
HOS12_BASELINE_C2 = 0.3758
HOS12_BASELINE_C5 = 0.8224

# Calibrated thresholds
THR_HOS12_MACRO_DROP_REJECT = 0.005   # > drop = REJECT
THR_HOS12_MACRO_DROP_CAVEAT = 0.0005  # tolerance band around baseline -> APPROVE
THR_C2_COUNT_DIFF_REJECT = 30
THR_C2_COUNT_DIFF_CAVEAT = 15
THR_C5_COUNT_DIFF_REJECT = 40
THR_DIFF_REJECT = 120
THR_DIFF_CAVEAT = 30
THR_UNANIM_OVERRIDE_REJECT = 30
THR_UNANIM_OVERRIDE_CAVEAT = 0   # anchors had 0; any override on the 98.8% consensus rows is a flag
THR_HOS12_C2_F1_DROP_CAVEAT = 0.02


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values("Id").reset_index(drop=True)


def hos12_score(pred_arr: np.ndarray, file_ids: np.ndarray) -> tuple[float, dict, dict]:
    """Score predictions on HOS-12 fold. pred_arr aligned with file_ids."""
    hos = pd.read_csv(HOS12_FOLD)
    fid_to_pred = dict(zip(file_ids.tolist(), pred_arr.tolist()))
    missing = [int(f) for f in hos["file_id"] if int(f) not in fid_to_pred]
    if missing:
        return float("nan"), {}, {"error": f"missing {len(missing)} HOS-12 file_ids in OOF"}
    hos_pred = np.array([fid_to_pred[int(f)] for f in hos["file_id"]])
    hos_label = hos["label"].values
    macro = float(f1_score(hos_label, hos_pred, average="macro"))
    per_class = {int(i): float(v) for i, v in enumerate(
        f1_score(hos_label, hos_pred, average=None, labels=list(range(6)))
    )}
    counts = {int(k): int(v) for k, v in pd.Series(hos_pred).value_counts().sort_index().items()}
    return macro, per_class, counts


def diagnose_csv(csv_path: Path) -> dict:
    """CSV-level diagnostics: diff vs anchors, label counts, unanim override count."""
    cand = load_csv(csv_path)
    a248 = load_csv(ANCHOR_PRIMARY)
    a244 = load_csv(ANCHOR_244)
    a240 = load_csv(ANCHOR_240)

    if not np.array_equal(cand["Id"].values, a248["Id"].values):
        raise ValueError(f"Id mismatch between {csv_path} and primary anchor")

    cand_lab = cand["Label"].values
    a248_lab = a248["Label"].values
    a244_lab = a244["Label"].values
    a240_lab = a240["Label"].values

    diff_248 = int((cand_lab != a248_lab).sum())
    diff_244 = int((cand_lab != a244_lab).sum())
    diff_240 = int((cand_lab != a240_lab).sum())

    unanim = (a248_lab == a244_lab) & (a244_lab == a240_lab)
    unanim_override = int((unanim & (cand_lab != a248_lab)).sum())
    mixed_change = int((~unanim & (cand_lab != a248_lab)).sum())

    counts = {int(k): int(v) for k, v in pd.Series(cand_lab).value_counts().sort_index().items()}
    c2_count = counts.get(2, 0)
    c2_diff = c2_count - ANCHOR_LABEL_COUNTS[2]
    c5_count = counts.get(5, 0)
    c5_diff = c5_count - ANCHOR_LABEL_COUNTS[5]

    # Agreement with each anchor
    agree_248 = int((cand_lab == a248_lab).sum())
    agree_244 = int((cand_lab == a244_lab).sum())
    agree_240 = int((cand_lab == a240_lab).sum())
    # Cross-anchor agreement count for the row (0/1/2/3 anchors agree with candidate per row)
    per_row_agree = (
        (cand_lab == a248_lab).astype(int)
        + (cand_lab == a244_lab).astype(int)
        + (cand_lab == a240_lab).astype(int)
    )
    rows_0agree = int((per_row_agree == 0).sum())
    rows_1agree = int((per_row_agree == 1).sum())

    return {
        "csv": str(csv_path),
        "n_rows": int(len(cand)),
        "label_counts": counts,
        "c2_count": c2_count,
        "c2_count_diff": c2_diff,
        "c5_count": c5_count,
        "c5_count_diff": c5_diff,
        "diff_vs_0.8248": diff_248,
        "diff_vs_0.8244": diff_244,
        "diff_vs_0.8240": diff_240,
        "unanim_override": unanim_override,
        "mixed_zone_change": mixed_change,
        "agree_0.8248": agree_248,
        "agree_0.8244": agree_244,
        "agree_0.8240": agree_240,
        "rows_0_anchor_agree": rows_0agree,
        "rows_1_anchor_agree": rows_1agree,
    }


def verdict_from_diagnostics(d: dict, hos12: dict | None) -> tuple[str, list[str]]:
    reasons = []
    # Hard rejections
    if hos12 is not None and hos12.get("macro_f1") is not None and not np.isnan(hos12["macro_f1"]):
        macro_drop = HOS12_BASELINE_MACRO - hos12["macro_f1"]
        if macro_drop > THR_HOS12_MACRO_DROP_REJECT:
            reasons.append(f"HOS-12 macro {hos12['macro_f1']:.4f} drops {macro_drop:+.4f} vs baseline {HOS12_BASELINE_MACRO:.4f}")
    if abs(d["c2_count_diff"]) > THR_C2_COUNT_DIFF_REJECT:
        reasons.append(f"c2 count {d['c2_count']} differs {d['c2_count_diff']:+d} from anchor {ANCHOR_LABEL_COUNTS[2]} (>{THR_C2_COUNT_DIFF_REJECT})")
    if abs(d["c5_count_diff"]) > THR_C5_COUNT_DIFF_REJECT:
        reasons.append(f"c5 count {d['c5_count']} differs {d['c5_count_diff']:+d} from anchor {ANCHOR_LABEL_COUNTS[5]} (>{THR_C5_COUNT_DIFF_REJECT})")
    if d["diff_vs_0.8248"] > THR_DIFF_REJECT:
        reasons.append(f"diff vs 0.8248 = {d['diff_vs_0.8248']} > {THR_DIFF_REJECT}")
    if d["unanim_override"] > THR_UNANIM_OVERRIDE_REJECT:
        reasons.append(f"unanim_override {d['unanim_override']} > {THR_UNANIM_OVERRIDE_REJECT}")

    if reasons:
        return "REJECT", reasons

    # Caveats
    if hos12 is not None and hos12.get("macro_f1") is not None and not np.isnan(hos12["macro_f1"]):
        macro_drop = HOS12_BASELINE_MACRO - hos12["macro_f1"]
        if macro_drop > THR_HOS12_MACRO_DROP_CAVEAT:
            reasons.append(f"HOS-12 macro {hos12['macro_f1']:.4f} below baseline {HOS12_BASELINE_MACRO:.4f}")
        c2_f1 = hos12.get("per_class", {}).get(2)
        if c2_f1 is not None:
            c2_f1_drop = HOS12_BASELINE_C2 - c2_f1
            if c2_f1_drop > THR_HOS12_C2_F1_DROP_CAVEAT:
                reasons.append(f"HOS-12 c2 F1 {c2_f1:.4f} drops {c2_f1_drop:+.4f} vs baseline {HOS12_BASELINE_C2:.4f}")
    if d["diff_vs_0.8248"] > THR_DIFF_CAVEAT:
        reasons.append(f"diff vs 0.8248 = {d['diff_vs_0.8248']} > {THR_DIFF_CAVEAT}")
    if abs(d["c2_count_diff"]) > THR_C2_COUNT_DIFF_CAVEAT:
        reasons.append(f"c2 count diff {d['c2_count_diff']:+d} > {THR_C2_COUNT_DIFF_CAVEAT}")
    if d["unanim_override"] > THR_UNANIM_OVERRIDE_CAVEAT:
        reasons.append(f"unanim_override {d['unanim_override']} > {THR_UNANIM_OVERRIDE_CAVEAT}")

    if reasons:
        return "APPROVE-WITH-CAVEAT", reasons

    return "APPROVE", ["clears all thresholds"]


def main():
    parser = argparse.ArgumentParser(description="Validate a candidate Kaggle submission CSV.")
    parser.add_argument("csv", type=Path, help="Candidate test-submission CSV")
    parser.add_argument(
        "--oof-npz",
        type=Path,
        default=None,
        help="Optional OOF/holdout predictions npz with 'pred' (or 'proba'+'classes') + 'file_id'. "
        "Used to compute HOS-12 macro. Without this, HOS-12 is skipped.",
    )
    parser.add_argument("--oof-pred-key", default="pred", help="npz key for prediction array (default 'pred'). If absent and 'proba'+'classes' present, argmax is used.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write JSON diagnostic.")
    args = parser.parse_args()

    diag = diagnose_csv(args.csv)

    hos12 = None
    if args.oof_npz is not None and args.oof_npz.exists():
        f = np.load(args.oof_npz, allow_pickle=True)
        if args.oof_pred_key in f.files:
            pred = np.asarray(f[args.oof_pred_key]).astype(int)
        elif "proba" in f.files and "classes" in f.files:
            pred = np.asarray(f["classes"])[np.argmax(f["proba"], axis=1)].astype(int)
        else:
            raise ValueError(f"OOF npz must contain '{args.oof_pred_key}' or ('proba'+'classes'). Got: {list(f.files)}")
        if "file_id" not in f.files:
            raise ValueError("OOF npz must contain 'file_id'")
        macro, per_class, counts_or_err = hos12_score(pred, np.asarray(f["file_id"]).astype(int))
        hos12 = {
            "macro_f1": macro,
            "per_class": per_class,
            "hos12_pred_counts": counts_or_err,
            "source": str(args.oof_npz),
        }

    verdict, reasons = verdict_from_diagnostics(diag, hos12)

    # One-line summary
    fname = Path(diag["csv"]).name
    hos_str = "NA"
    c2_f1_str = "NA"
    if hos12 is not None and not np.isnan(hos12.get("macro_f1", float("nan"))):
        hos_str = f"{hos12['macro_f1']:.4f}"
        c2_f1_str = f"{hos12['per_class'].get(2, float('nan')):.4f}"
    summary = (
        f"{verdict} | {fname} | HOS12={hos_str} | c2F1={c2_f1_str} | "
        f"diff={diag['diff_vs_0.8248']} | c2_cnt={diag['c2_count']} (Δ{diag['c2_count_diff']:+d}) | "
        f"unanim_override={diag['unanim_override']} | reason: {reasons[0] if reasons else 'clears'}"
    )
    print(summary)

    out = {"verdict": verdict, "reasons": reasons, "diagnostics": diag, "hos12": hos12}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str))
    return 0 if verdict in ("APPROVE", "APPROVE-WITH-CAVEAT") else 1


if __name__ == "__main__":
    sys.exit(main())
