#!/usr/bin/env python3
"""Apply BYOL probas only on the 144 uncertain test rows (≤3/5 anchor agreement).

For each uncertain row:
  combined = 0.4 * base_ensemble_proba + 0.3 * byol_proba + 0.3 * anchor_vote_count_proba
  new_label = argmax(combined)

Conservative: caps changes at 30. Falls back to anchor if BYOL says class-2
(since BYOL likely has c2 F1 < 0.2; not trustworthy for c2).

This mirrors the winning H8_uncertain_refined mechanism (which scored +0.0004
on May 13) but injects BYOL as a third opinion alongside the existing
weighted ensemble and the anchor vote count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-csv", type=Path, default=Path("submissions/submission_h8_v3_sslv2_weighted.csv"))
    p.add_argument("--summary-csv", type=Path, default=Path("artifacts/uncertain_viz/summary.csv"))
    p.add_argument("--byol-test", type=Path, default=Path("artifacts/byol_lgbm/test_proba.npz"))
    p.add_argument("--byol-oof", type=Path, default=Path("artifacts/byol_lgbm/oof_proba.npz"),
                   help="Used to check per-class F1; classes with F1<0.2 are not promoted to.")
    p.add_argument("--ensemble-test", type=Path, default=Path("artifacts/weighted_ensemble/test_viterbi.npz"))
    p.add_argument("--max-changes", type=int, default=30)
    p.add_argument("--w-ensemble", type=float, default=0.40)
    p.add_argument("--w-byol", type=float, default=0.30)
    p.add_argument("--w-vote", type=float, default=0.30)
    p.add_argument("--byol-min-class-f1", type=float, default=0.20,
                   help="Don't switch to class K if BYOL's OOF F1 on K is below this.")
    p.add_argument("--output", type=Path, default=Path("submissions/submission_byol_refine_uncertain.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    base_dict = dict(zip(base["Id"].astype(int), base["Label"].astype(int)))
    print(f"Base label counts: {pd.Series(list(base_dict.values())).value_counts().sort_index().to_dict()}")

    summary = pd.read_csv(args.summary_csv)
    print(f"Uncertain rows: {len(summary)}")

    byol = np.load(args.byol_test, allow_pickle=True)
    byol_fid = byol["file_id"].astype(int)
    byol_proba = byol["proba"].astype(np.float64)
    byol_lookup = {int(f): byol_proba[i] for i, f in enumerate(byol_fid)}

    ens = np.load(args.ensemble_test, allow_pickle=True)
    ens_fid = ens["file_id"].astype(int)
    ens_proba = ens["proba"].astype(np.float64)
    ens_lookup = {int(f): ens_proba[i] for i, f in enumerate(ens_fid)}

    # Per-class BYOL F1 → mask classes BYOL can't trust
    from sklearn.metrics import f1_score
    byol_oof = np.load(args.byol_oof, allow_pickle=True)
    byol_o_y = byol_oof["label"].astype(int) if "label" in byol_oof.files else byol_oof["y"].astype(int)
    byol_o_proba = byol_oof["proba"].astype(np.float64)
    byol_class_f1 = f1_score(byol_o_y, byol_o_proba.argmax(axis=1), average=None)
    print(f"BYOL per-class OOF F1: {[f'{x:.3f}' for x in byol_class_f1]}")
    trusted_classes = {k: bool(byol_class_f1[k] >= args.byol_min_class_f1) for k in range(len(byol_class_f1))}
    print(f"BYOL trusted classes (F1≥{args.byol_min_class_f1}): {trusted_classes}")

    changes = []
    for _, row in summary.iterrows():
        fid = int(row["file_id"])
        h8 = int(row["H8_label"])

        if fid not in byol_lookup or fid not in ens_lookup:
            continue

        # Anchor vote distribution
        anchors = [int(row[f"anchor_{name}"]) for name in ["ssl_hybrid", "inception_blend", "consensus_greedy", "centered_meta", "lgbm63"]]
        vote_proba = np.zeros(6)
        for a in anchors:
            vote_proba[a] += 1.0 / len(anchors)

        combined = (
            args.w_ensemble * ens_lookup[fid]
            + args.w_byol * byol_lookup[fid]
            + args.w_vote * vote_proba
        )

        # Mask classes BYOL can't trust — but only the BYOL portion
        for k in range(6):
            if not trusted_classes[k]:
                # Subtract BYOL contribution for untrusted class
                combined[k] -= args.w_byol * byol_lookup[fid][k]

        new_label = int(combined.argmax())

        if new_label != h8:
            changes.append({
                "file_id": fid,
                "user": row["user"],
                "from": h8,
                "to": new_label,
                "byol_proba_to": float(byol_lookup[fid][new_label]),
                "ens_proba_to": float(ens_lookup[fid][new_label]),
                "vote_to": float(vote_proba[new_label]),
                "combined_max": float(combined.max()),
                "combined_h8": float(combined[h8]),
                "margin": float(combined.max() - combined[h8]),
            })

    # Sort by margin, take top-K
    changes.sort(key=lambda c: -c["margin"])
    changes = changes[: args.max_changes]
    print(f"Proposing {len(changes)} changes (cap {args.max_changes})")

    # Apply
    pred = dict(base_dict)
    for c in changes:
        pred[c["file_id"]] = c["to"]

    out_df = pd.DataFrame({"Id": list(pred.keys()), "Label": list(pred.values())})
    out_df["Id"] = out_df["Id"].astype(int)
    out_df["Label"] = out_df["Label"].astype(int)
    sample = pd.read_csv("data/raw/sample_submission.csv")
    out_df = out_df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"\nWrote {args.output}")
    print(f"SHA: {sha}")
    print(f"Final label counts: {out_df['Label'].value_counts().sort_index().to_dict()}")
    if changes:
        print("\nChanges:")
        for c in changes[:20]:
            print(f"  fid={c['file_id']} (user={c['user']}) {c['from']}→{c['to']}  margin={c['margin']:.3f}")

    metadata = {
        "base_csv": str(args.base_csv),
        "uncertain_rows_total": int(len(summary)),
        "byol_class_f1": [float(x) for x in byol_class_f1],
        "byol_trusted": trusted_classes,
        "weights": {"ensemble": args.w_ensemble, "byol": args.w_byol, "vote": args.w_vote},
        "byol_min_class_f1": args.byol_min_class_f1,
        "n_changes": len(changes),
        "final_label_counts": {int(k): int(v) for k, v in out_df["Label"].value_counts().sort_index().to_dict().items()},
        "sha256": sha,
        "changes": changes,
    }
    meta_path = Path(str(args.output).replace(".csv", "_metadata.json"))
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
