#!/usr/bin/env python3
"""H1: Multi-anchor public-score-weighted vote + hybrid v3 recovery layer.

Inputs: N anchor CSVs (each with known public score), one new-pipeline soft-proba npz.

For each test row:
  votes[class_c] = sum over anchors of weight_i * indicator(anchor_i predicts c)
  winner = argmax_c votes[class_c]   (tiebreak: highest-weighted anchor)

Then apply hybrid v3 layer: where winner disagrees with the strongest anchor on a rare
class (2/3/5) AND the new pipeline's soft proba for that class >= threshold, restore the
strongest anchor's prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-anchor weighted vote + hybrid v3 recovery.")
    parser.add_argument(
        "--anchors",
        nargs="+",
        required=True,
        help="anchor CSV path with appended public score, e.g. 'submissions/X.csv:0.8240'",
    )
    parser.add_argument("--hybrid-proba-npz", type=Path, default=Path("artifacts/weighted_ensemble/test_viterbi.npz"))
    parser.add_argument(
        "--anchor-for-hybrid",
        type=Path,
        default=Path("submissions/submission_ssl_hybrid_recover.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c2-threshold", type=float, default=0.20)
    parser.add_argument("--c3-threshold", type=float, default=0.25)
    parser.add_argument("--c5-threshold", type=float, default=0.25)
    parser.add_argument("--max-c2", type=int, default=35)
    parser.add_argument("--max-c3", type=int, default=25)
    parser.add_argument("--max-c5", type=int, default=10)
    parser.add_argument("--apply-hybrid", action="store_true", default=True)
    parser.add_argument("--no-hybrid", dest="apply_hybrid", action="store_false")
    return parser.parse_args()


def parse_anchors(specs: list[str]) -> list[tuple[str, float, Path]]:
    out = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"anchor spec must include ':<public_score>', got {spec}")
        path, score = spec.rsplit(":", 1)
        out.append((Path(path).stem, float(score), Path(path)))
    return out


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    anchors = parse_anchors(args.anchors)
    print(f"Loaded {len(anchors)} anchors:")
    for name, score, path in anchors:
        print(f"  {name}: weight={score:.4f} from {path}")

    # Load anchor CSVs
    anchor_dfs = []
    ref_ids = None
    for name, score, path in anchors:
        df = pd.read_csv(path)
        if ref_ids is None:
            ref_ids = df["Id"].astype(int).to_numpy()
        else:
            assert (df["Id"].astype(int).to_numpy() == ref_ids).all(), f"Id mismatch in {path}"
        anchor_dfs.append((name, score, df["Label"].astype(int).to_numpy()))

    n = len(ref_ids)
    n_classes = 6

    # Compute weighted votes per row per class
    votes = np.zeros((n, n_classes), dtype=float)
    for name, w, lbl in anchor_dfs:
        for c in range(n_classes):
            votes[lbl == c, c] += w

    # Argmax with tiebreak: when two classes tie, prefer the strongest anchor's pick
    strongest_anchor_lbl = anchor_dfs[0][2]  # first anchor (assumed strongest)
    vote_pred = np.zeros(n, dtype=int)
    max_vote = votes.max(axis=1)
    for i in range(n):
        candidates = np.where(votes[i] == max_vote[i])[0]
        if len(candidates) == 1:
            vote_pred[i] = candidates[0]
        else:
            # Tiebreak: pick the strongest anchor's class if it's in candidates
            if strongest_anchor_lbl[i] in candidates:
                vote_pred[i] = strongest_anchor_lbl[i]
            else:
                vote_pred[i] = candidates[0]

    # Stats
    vote_counts = pd.Series(vote_pred).value_counts().sort_index().to_dict()
    print(f"\nMulti-anchor vote label counts: {vote_counts}")
    for i, (name, score, lbl) in enumerate(anchor_dfs):
        ac = pd.Series(lbl).value_counts().sort_index().to_dict()
        agree = int((vote_pred == lbl).sum())
        print(f"  {name} ({score:.4f}): {agree}/{n} agree ({100*agree/n:.1f}%) | counts {ac}")

    pred = vote_pred.copy()

    # Apply hybrid v3 recovery layer
    if args.apply_hybrid and args.hybrid_proba_npz.exists():
        npz = np.load(args.hybrid_proba_npz, allow_pickle=True)
        proba_full = npz["proba"].astype(float)
        proba_fids = npz["file_id"].astype(int)
        # Align proba to ref_ids
        proba_idx = {int(f): k for k, f in enumerate(proba_fids)}
        order = np.array([proba_idx[int(i)] for i in ref_ids], dtype=int)
        proba = proba_full[order]

        # Load hybrid anchor (the "from where to restore" reference)
        ha_csv = pd.read_csv(args.anchor_for_hybrid)
        ha_idx = {int(i): k for k, i in enumerate(ha_csv["Id"])}
        anchor_for_hyb = np.array([ha_csv["Label"].iloc[ha_idx[int(i)]] for i in ref_ids], dtype=int)

        def recover(pred, target, anchor, proba, threshold, cap, name):
            disagree = (anchor == target) & (pred != target)
            cand = np.where(disagree & (proba[:, target] >= threshold))[0]
            cand_sorted = cand[np.argsort(-proba[cand, target])]
            cand_top = cand_sorted[:cap]
            pred[cand_top] = target
            print(f"  hybrid c{target} ({name}): {len(cand_top)} (thresh {threshold}, cap {cap})")
            return len(cand_top)

        print(f"\nApplying hybrid v3 layer (anchor: {args.anchor_for_hybrid.name})")
        n2 = recover(pred, 2, anchor_for_hyb, proba, args.c2_threshold, args.max_c2, "rare burst")
        n3 = recover(pred, 3, anchor_for_hyb, proba, args.c3_threshold, args.max_c3, "rare activity")
        n5 = recover(pred, 5, anchor_for_hyb, proba, args.c5_threshold, args.max_c5, "rare burst")
        n_hybrid_changes = n2 + n3 + n5
    else:
        n_hybrid_changes = 0
        print("\nSkipping hybrid layer (--no-hybrid or proba npz missing)")

    # Compose CSV
    out_df = pd.DataFrame({"Id": ref_ids.astype(int), "Label": pred.astype(int)})
    sample = pd.read_csv("data/raw/sample_submission.csv")
    out_df = out_df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
    out_df["Label"] = out_df["Label"].astype(int)
    out_df.to_csv(args.output, index=False)

    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    final_counts = out_df["Label"].value_counts().sort_index().to_dict()
    diff_vote = int((pred != vote_pred).sum())
    n_diff_strongest = int((pred != strongest_anchor_lbl).sum())
    print(f"\nWrote {args.output}")
    print(f"SHA256: {sha}")
    print(f"Final label counts: {final_counts}")
    print(f"Diff from vote (pre-hybrid): {diff_vote}")
    print(f"Diff from strongest anchor:  {n_diff_strongest}")

    # Metadata
    metadata = {
        "csv": str(args.output),
        "sha256": sha,
        "anchors": [{"name": n_, "weight": s, "path": str(p)} for n_, s, p in anchors],
        "final_label_counts": {int(k): int(v) for k, v in final_counts.items()},
        "vote_label_counts": {int(k): int(v) for k, v in vote_counts.items()},
        "diff_vote_to_final": diff_vote,
        "diff_to_strongest_anchor": n_diff_strongest,
        "hybrid_changes": int(n_hybrid_changes),
        "thresholds": {"c2": args.c2_threshold, "c3": args.c3_threshold, "c5": args.c5_threshold},
        "caps": {"c2": args.max_c2, "c3": args.max_c3, "c5": args.max_c5},
    }
    meta_path = Path(str(args.output).replace(".csv", "_metadata.json"))
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
