#!/usr/bin/env python3
"""H4+: Apply class specialists as both promote and demote signals on top of
a base multi-anchor (or hybrid) CSV.

Mechanism:
  - For each rare class K (2, 3, 4, 5), have a binary specialist confidence per row.
  - PROMOTE: where base predicts non-K and specialist_K >= promote_threshold_K → switch to K (cap per class).
  - DEMOTE: where base predicts K and specialist_K < demote_threshold_K → demote to argmax-of-other.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Specialist-aware hybrid override.")
    parser.add_argument("--base-csv", type=Path, required=True, help="Base CSV (e.g., multi_anchor_vote).")
    parser.add_argument("--ensemble-proba-npz", type=Path, default=Path("artifacts/weighted_ensemble/test_viterbi.npz"))
    # Specialist probas — each class is a separate npz
    parser.add_argument("--spec-c2", type=Path, default=Path("artifacts/class_specialist/test_c2.npz"))
    parser.add_argument("--spec-c3", type=Path, default=Path("artifacts/class_specialist/test_c3.npz"))
    parser.add_argument("--spec-c4", type=Path, default=Path("artifacts/class_specialist/test_c4.npz"))
    parser.add_argument("--spec-c5", type=Path, default=Path("artifacts/class_specialist/test_c5.npz"))
    # Promote thresholds (per class)
    parser.add_argument("--promote-c2", type=float, default=0.70, help="High threshold (c2 weak)")
    parser.add_argument("--promote-c3", type=float, default=0.60)
    parser.add_argument("--promote-c4", type=float, default=0.70)
    parser.add_argument("--promote-c5", type=float, default=0.65)
    # Caps on promotions
    parser.add_argument("--cap-c2", type=int, default=20)
    parser.add_argument("--cap-c3", type=int, default=30)
    parser.add_argument("--cap-c4", type=int, default=15)
    parser.add_argument("--cap-c5", type=int, default=20)
    # Demote thresholds (if base predicts K but specialist says <demote_K)
    parser.add_argument("--demote-c2", type=float, default=0.10)
    parser.add_argument("--demote-c3", type=float, default=0.30)
    parser.add_argument("--demote-c4", type=float, default=0.30)
    parser.add_argument("--demote-c5", type=float, default=0.30)
    parser.add_argument("--enable-demote", action="store_true", default=False)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base_csv)
    base_ids = base["Id"].astype(int).to_numpy()
    base_lbl = base["Label"].astype(int).to_numpy()
    pred = base_lbl.copy()
    n = len(pred)
    print(f"Base label counts: {pd.Series(base_lbl).value_counts().sort_index().to_dict()}")

    # Load ensemble proba (for argmax-of-other in demote)
    ens = np.load(args.ensemble_proba_npz, allow_pickle=True)
    ens_proba = ens["proba"].astype(float)
    ens_idx = {int(f): i for i, f in enumerate(ens["file_id"])}
    ens_order = np.array([ens_idx[int(i)] for i in base_ids], dtype=int)
    ens_proba = ens_proba[ens_order]

    def load_spec(path: Path) -> np.ndarray:
        d = np.load(path, allow_pickle=True)
        idx = {int(f): i for i, f in enumerate(d["file_id"])}
        order = np.array([idx[int(i)] for i in base_ids], dtype=int)
        return d["proba"][order].astype(float)

    spec = {
        2: load_spec(args.spec_c2),
        3: load_spec(args.spec_c3),
        4: load_spec(args.spec_c4),
        5: load_spec(args.spec_c5),
    }

    promote_thresh = {2: args.promote_c2, 3: args.promote_c3, 4: args.promote_c4, 5: args.promote_c5}
    promote_cap = {2: args.cap_c2, 3: args.cap_c3, 4: args.cap_c4, 5: args.cap_c5}
    demote_thresh = {2: args.demote_c2, 3: args.demote_c3, 4: args.demote_c4, 5: args.demote_c5}

    # PROMOTE: where base says non-K and specialist_K >= threshold
    promotes = {}
    for k in (4, 2, 5, 3):  # process strongest first (c4 most reliable)
        candidates = np.where((pred != k) & (spec[k] >= promote_thresh[k]))[0]
        # Top-cap by specialist confidence
        candidates_sorted = candidates[np.argsort(-spec[k][candidates])]
        candidates_top = candidates_sorted[: promote_cap[k]]
        pred[candidates_top] = k
        promotes[k] = len(candidates_top)
        if len(candidates_top) > 0:
            print(f"  PROMOTE c{k}: {len(candidates_top)} (thresh {promote_thresh[k]:.2f}, cap {promote_cap[k]}) — proba range {spec[k][candidates_top].min():.3f}-{spec[k][candidates_top].max():.3f}")
        else:
            print(f"  PROMOTE c{k}: 0")

    # DEMOTE: where base says K and specialist_K < threshold
    demotes = {}
    if args.enable_demote:
        for k in (2, 3, 4, 5):
            candidates = np.where((pred == k) & (spec[k] < demote_thresh[k]))[0]
            for idx in candidates:
                # Argmax of ensemble proba excluding class k
                p = ens_proba[idx].copy()
                p[k] = -np.inf
                pred[idx] = int(p.argmax())
            demotes[k] = int(len(candidates))
            print(f"  DEMOTE c{k}: {len(candidates)} (thresh {demote_thresh[k]:.2f})")
    else:
        print("  DEMOTE: disabled")

    # Write CSV
    out = pd.DataFrame({"Id": base_ids.astype(int), "Label": pred.astype(int)})
    sample = pd.read_csv("data/raw/sample_submission.csv")
    out = out.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    out["Label"] = out["Label"].astype(int)
    out.to_csv(args.output, index=False)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    final_counts = out["Label"].value_counts().sort_index().to_dict()
    n_diff_base = int((pred != base_lbl).sum())
    print(f"\nWrote {args.output}")
    print(f"SHA: {sha}")
    print(f"Final label counts: {final_counts}")
    print(f"Diff from base: {n_diff_base}")

    metadata = {
        "base_csv": str(args.base_csv),
        "promotes": {int(k): int(v) for k, v in promotes.items()},
        "demotes": {int(k): int(v) for k, v in demotes.items()},
        "thresholds": {f"promote_c{k}": promote_thresh[k] for k in promote_thresh},
        "caps": {f"cap_c{k}": promote_cap[k] for k in promote_cap},
        "final_label_counts": {int(k): int(v) for k, v in final_counts.items()},
        "diff_from_base": int(n_diff_base),
        "sha256": sha,
    }
    Path(str(args.output).replace(".csv", "_metadata.json")).write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
