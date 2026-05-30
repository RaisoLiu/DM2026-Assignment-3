#!/usr/bin/env python3
"""H_visual: hand-crafted rules from visual inspection of 144 uncertain test rows.

Rules derived from visualization agent's analysis (artifacts/uncertain_viz/):
  Rule B: H8==3 AND max_vote==4 AND frac_high_std>=0.95 AND frac_low_std<=0.02 → 4
  Rule C: H8==3 AND max_vote==2 AND frac_high_std<0.65  AND frac_low_std>=0.18 → 2
  Rule D: H8==1 AND max_vote==2 AND std_xyz_mean>0.24 AND mean_x_ptp>1.2 → 2
  Rule E: H8==1 AND max_vote∈{5,3} AND specific spike pattern → 5/3

Applied ONLY on uncertain rows (agreement≤3/5). Other rows kept as H8.

Output: submissions/submission_manual_rules.csv  +  metadata.json
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
    p.add_argument("--test-seq", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--output", type=Path, default=Path("submissions/submission_manual_rules.csv"))
    p.add_argument("--apply-rules", default="B,C,D,E",
                   help="Comma-separated rules to enable.")
    return p.parse_args()


def compute_features(x: np.ndarray) -> dict:
    """x: (6, 300). Returns visual-pattern features used by rules."""
    mean_x, mean_y, mean_z = x[0], x[1], x[2]
    std_x, std_y, std_z = x[3], x[4], x[5]
    std_total = std_x + std_y + std_z
    return {
        "std_xyz_mean": float(std_total.mean()),
        "frac_high_std": float((std_total > 0.2).mean()),
        "frac_low_std": float((std_total < 0.05).mean()),
        "mean_x_ptp": float(np.ptp(mean_x)),
        "max_total_std": float(std_total.max()),
        "med_total_std": float(np.median(std_total)),
        "spike_ratio": float(std_total.max() / max(np.median(std_total), 1e-6)),
        "n_high_spikes": int((std_total > 0.3).sum()),
    }


def main() -> None:
    args = parse_args()
    rules = set(args.apply_rules.upper().split(","))
    print(f"Applying rules: {sorted(rules)}")

    base = pd.read_csv(args.base_csv)
    base_lbl = base.set_index("Id")["Label"].to_dict()
    print(f"Base label counts: {pd.Series(list(base_lbl.values())).value_counts().sort_index().to_dict()}")

    summary = pd.read_csv(args.summary_csv)
    print(f"Uncertain rows: {len(summary)}")

    test = np.load(args.test_seq, allow_pickle=True)
    test_x = test["x"]
    test_fid = test["file_ids"].astype(int)
    fid_to_idx = {int(f): i for i, f in enumerate(test_fid)}

    changes = []
    rule_counts = {"B": 0, "C": 0, "D": 0, "E": 0}

    for _, row in summary.iterrows():
        fid = int(row["file_id"])
        if fid not in fid_to_idx:
            continue
        x = test_x[fid_to_idx[fid]]
        feat = compute_features(x)
        h8 = int(row["H8_label"])
        max_vote = int(row["max_vote_class"])

        new_label = h8

        # Rule B: H8==3 AND max_vote==4 AND saturated high-std
        if "B" in rules and h8 == 3 and max_vote == 4:
            if feat["frac_high_std"] >= 0.95 and feat["frac_low_std"] <= 0.02:
                new_label = 4
                rule_counts["B"] += 1

        # Rule C: H8==3 AND max_vote==2 AND intermittent activity
        elif "C" in rules and h8 == 3 and max_vote == 2:
            if feat["frac_high_std"] < 0.65 and feat["frac_low_std"] >= 0.18:
                new_label = 2
                rule_counts["C"] += 1

        # Rule D: H8==1 AND max_vote==2 AND mid-energy + level shifts
        elif "D" in rules and h8 == 1 and max_vote == 2:
            if feat["std_xyz_mean"] > 0.24 and feat["mean_x_ptp"] > 1.2:
                new_label = 2
                rule_counts["D"] += 1

        # Rule E: H8==1 with isolated spike pattern → max_vote class (5 or 3)
        elif "E" in rules and h8 == 1 and max_vote in (5, 3):
            anchors_5 = (int(row["anchor_ssl_hybrid"]) == 5) + (int(row["anchor_inception_blend"]) == 5)
            anchors_3 = (int(row["anchor_ssl_hybrid"]) == 3) + (int(row["anchor_inception_blend"]) == 3)
            # SSL anchors agreeing on rare class
            cond_5 = max_vote == 5 and anchors_5 >= 1
            cond_3 = max_vote == 3 and anchors_3 >= 1
            isolated = feat["frac_high_std"] < 0.10 and feat["spike_ratio"] > 8.0 and feat["n_high_spikes"] <= 6
            if (cond_5 or cond_3) and isolated:
                new_label = max_vote
                rule_counts["E"] += 1

        if new_label != h8:
            changes.append({
                "file_id": fid,
                "user": row["user"],
                "from": h8,
                "to": new_label,
                "rule": (
                    "B" if (h8 == 3 and new_label == 4) else
                    "C" if (h8 == 3 and new_label == 2) else
                    "D" if (h8 == 1 and new_label == 2) else
                    "E"
                ),
                **{k: round(v, 3) for k, v in feat.items()},
                "max_vote": max_vote,
            })

    # Apply changes
    pred = dict(base_lbl)
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
    final_counts = out_df["Label"].value_counts().sort_index().to_dict()

    print(f"\nWrote {args.output}")
    print(f"SHA: {sha}")
    print(f"Final label counts: {final_counts}")
    print(f"Rule counts: {rule_counts}")
    print(f"Total changes: {len(changes)}")
    if changes:
        print("\nChanges:")
        for c in changes:
            print(f"  {c['rule']}: fid={c['file_id']} {c['from']}→{c['to']} (user={c['user']}, max_vote={c['max_vote']}, std_mean={c['std_xyz_mean']:.3f}, frac_high={c['frac_high_std']:.3f}, frac_low={c['frac_low_std']:.3f})")

    metadata = {
        "base_csv": str(args.base_csv),
        "rules_applied": sorted(rules),
        "rule_counts": rule_counts,
        "n_changes": len(changes),
        "final_label_counts": {int(k): int(v) for k, v in final_counts.items()},
        "sha256": sha,
        "changes": changes,
    }
    out_meta = Path(str(args.output).replace(".csv", "_metadata.json"))
    out_meta.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {out_meta}")


if __name__ == "__main__":
    main()
