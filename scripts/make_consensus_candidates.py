#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative consensus candidates from public-strong submissions.")
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/blend_search/consensus_candidates_may10.json"))
    parser.add_argument("--base", type=Path, default=Path("submissions/submission_centered_meta_viterbi_oof07693.csv"))
    parser.add_argument("--lgbm", type=Path, default=Path("submissions/submission_lgbm_leaves63_calibrated.csv"))
    parser.add_argument("--mini", type=Path, default=Path("submissions/submission_blend_minirocket_oof07597.csv"))
    parser.add_argument(
        "--centered-test-npz",
        type=Path,
        default=Path("artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz"),
    )
    parser.add_argument(
        "--round2-test-npz",
        type=Path,
        default=Path("artifacts/blend_search/test_blend_round2_oofbest.npz"),
    )
    parser.add_argument(
        "--threshold-summary-json",
        type=Path,
        default=Path("artifacts/proba_threshold_public_best_rare_sources/summary.json"),
    )
    parser.add_argument(
        "--threshold-summary-csv",
        type=Path,
        default=Path("artifacts/proba_threshold_public_best_rare_sources/summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    base = load_submission(args.base)
    lgbm = load_submission(args.lgbm)
    mini = load_submission(args.mini)
    for name, frame in (("lgbm", lgbm), ("mini", mini)):
        if not base["Id"].equals(frame["Id"]):
            raise ValueError(f"{name} Id order differs from base")
    threshold_context = load_threshold_context(args, base)
    greedy_groups = [(5, 3), (2, 5), (1, 2), (3, 1), (4, 5), (1, 5), (3, 4)]

    specs = [
        {
            "name": "consensus_rare_b",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on target {3,5} from non-2 source only.",
            "non2_targets": {3, 5},
            "include_source2_to5": False,
            "oof_delta_reference": 0.0023496721369845597,
        },
        {
            "name": "consensus_rare_bplusc",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on target {3,5} from non-2 source, plus source-2 to 5.",
            "non2_targets": {3, 5},
            "include_source2_to5": True,
            "oof_delta_reference": 0.0034055174082483486,
        },
        {
            "name": "consensus_rare_c",
            "description": "Base centered-meta Viterbi; only source-2 to class 5 when LGBM/Mini agree.",
            "non2_targets": set(),
            "include_source2_to5": True,
            "oof_delta_reference": 0.0010687423017115005,
        },
        {
            "name": "consensus_rare_f",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on target {3} from non-2 source only.",
            "non2_targets": {3},
            "include_source2_to5": False,
            "oof_delta_reference": 0.001636538355973327,
        },
        {
            "name": "consensus_rare_g",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on target {5} from non-2 source only.",
            "non2_targets": {5},
            "include_source2_to5": False,
            "oof_delta_reference": 0.0007266362650257818,
        },
        {
            "name": "consensus_rare_aplusc",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on target {2,3,4,5} from non-2 source, plus source-2 to 5.",
            "non2_targets": {2, 3, 4, 5},
            "include_source2_to5": True,
            "oof_delta_reference": 0.0035470995001685157,
        },
        {
            "name": "consensus_rare_a",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on target {2,3,4,5} from non-2 source only.",
            "non2_targets": {2, 3, 4, 5},
            "include_source2_to5": False,
            "oof_delta_reference": 0.0024972713176577566,
        },
    ]

    payload: list[dict[str, object]] = []
    consensus_predictions: dict[str, pd.Series] = {}
    for spec in specs:
        pred, mask, mask_non2, mask_source2_to5 = apply_consensus_rule(base, lgbm, mini, spec)
        consensus_predictions[str(spec["name"])] = pred.copy()
        out = base[["Id"]].copy()
        out["Label"] = pred.astype(int)
        output_path = args.output_dir / f"submission_{spec['name']}.csv"
        out.to_csv(output_path, index=False)
        record = {
            "name": spec["name"],
            "output": str(output_path),
            "sha256": sha256_file(output_path),
            "description": spec["description"],
            "oof_delta_reference": spec["oof_delta_reference"],
            "changes_vs_base": int(mask.sum()),
            "non2_changes": int(mask_non2.sum()),
            "source2_to5_changes": int(mask_source2_to5.sum() if spec["include_source2_to5"] else 0),
            "label_counts": {int(k): int(v) for k, v in out["Label"].value_counts().sort_index().items()},
            "changed_from_to": changed_from_to(base["Label"], pred, mask),
            "base_public_reference": 0.8130,
            "lgbm_public_reference": 0.8106,
            "mini_public_reference": 0.8087,
        }
        payload.append(record)
        print(f"Wrote {output_path}: changes={record['changes_vs_base']} counts={record['label_counts']}")

    greedy_pred, greedy_group_counts = apply_consensus_groups(base["Label"], lgbm, mini, greedy_groups)
    threshold3_groups = [(5, 3), (2, 5), (1, 2), (3, 1), (1, 5), (3, 4)]
    threshold3_seed, threshold3_group_counts = apply_consensus_groups(base["Label"], lgbm, mini, threshold3_groups)
    greedy_threshold3_pred, greedy_threshold3_rules = apply_threshold_rules(
        threshold3_seed,
        threshold_context,
        max_rules=3,
        threshold_level="q75",
    )
    alt_threshold4_groups = [(2, 5), (1, 2), (3, 1), (1, 5), (3, 4)]
    alt_threshold4_seed, alt_threshold4_group_counts = apply_consensus_groups(base["Label"], lgbm, mini, alt_threshold4_groups)
    alt_threshold4_pred, alt_threshold4_rules = apply_threshold_rules(
        alt_threshold4_seed,
        threshold_context,
        max_rules=4,
        threshold_level="q75",
    )
    greedy_specs = [
        {
            "name": "consensus_oof_greedy",
            "description": "Base centered-meta Viterbi; LGBM/Mini agree on OOF-greedy positive from-to groups.",
            "pred": greedy_pred,
            "oof_delta_reference": 0.005338015410896402,
            "groups": greedy_group_counts,
        },
        {
            "name": "consensus_oof_greedy_threshold3",
            "description": "OOF-optimized LGBM/Mini consensus group subset, then the first three restricted threshold rules.",
            "pred": greedy_threshold3_pred,
            "oof_delta_reference": 0.007700429,
            "groups": threshold3_group_counts,
            "threshold_rules": greedy_threshold3_rules,
            "threshold_order": "consensus_then_first3_threshold_q75",
        },
        {
            "name": "consensus_oof_alt_threshold4",
            "description": "Alternative OOF-optimized LGBM/Mini group subset, then the first four restricted threshold rules.",
            "pred": alt_threshold4_pred,
            "oof_delta_reference": 0.007451285,
            "groups": alt_threshold4_group_counts,
            "threshold_rules": alt_threshold4_rules,
            "threshold_order": "consensus_then_first4_threshold_q75",
        },
    ]
    for spec in greedy_specs:
        out = base[["Id"]].copy()
        out["Label"] = spec["pred"].astype(int)
        output_path = args.output_dir / f"submission_{spec['name']}.csv"
        out.to_csv(output_path, index=False)
        record = {
            "name": spec["name"],
            "output": str(output_path),
            "sha256": sha256_file(output_path),
            "description": spec["description"],
            "oof_delta_reference": spec["oof_delta_reference"],
            "changes_vs_base": int(out["Label"].ne(base["Label"]).sum()),
            "label_counts": {int(k): int(v) for k, v in out["Label"].value_counts().sort_index().items()},
            "changed_from_to": changed_from_to(base["Label"], out["Label"], out["Label"].ne(base["Label"])),
            "consensus_groups": spec["groups"],
            "base_public_reference": 0.8130,
            "lgbm_public_reference": 0.8106,
            "mini_public_reference": 0.8087,
        }
        if "threshold_rules" in spec:
            record["threshold_rules"] = spec["threshold_rules"]
            record["threshold_order"] = spec["threshold_order"]
        payload.append(record)
        print(f"Wrote {output_path}: changes={record['changes_vs_base']} counts={record['label_counts']}")

    threshold_specs = [
        {
            "name": "public_best_threshold",
            "description": "Base centered-meta Viterbi plus five median threshold rules selected on current public-best OOF.",
            "start": base["Label"],
            "threshold_order": "threshold_only",
            "oof_delta_reference": 0.005026137,
        },
        {
            "name": "consensus_rare_c_threshold",
            "description": "Consensus rare C seed, then five median threshold rules selected on current public-best OOF.",
            "start": consensus_predictions["consensus_rare_c"],
            "threshold_order": "consensus_then_threshold",
            "oof_delta_reference": 0.006095512,
        },
        {
            "name": "consensus_rare_bplusc_threshold",
            "description": "Five median threshold rules selected on current public-best OOF, then consensus rare B+C.",
            "start": base["Label"],
            "post_consensus_spec": next(spec for spec in specs if spec["name"] == "consensus_rare_bplusc"),
            "threshold_order": "threshold_then_consensus",
            "oof_delta_reference": 0.005435765,
        },
    ]
    for spec in threshold_specs:
        pred, rules = apply_threshold_rules(spec["start"], threshold_context)
        if "post_consensus_spec" in spec:
            pred, _, _, _ = apply_consensus_rule(pd.DataFrame({"Label": pred}), lgbm, mini, spec["post_consensus_spec"])
        out = base[["Id"]].copy()
        out["Label"] = pred.astype(int)
        output_path = args.output_dir / f"submission_{spec['name']}.csv"
        out.to_csv(output_path, index=False)
        record = {
            "name": spec["name"],
            "output": str(output_path),
            "sha256": sha256_file(output_path),
            "description": spec["description"],
            "oof_delta_reference": spec["oof_delta_reference"],
            "changes_vs_base": int(out["Label"].ne(base["Label"]).sum()),
            "label_counts": {int(k): int(v) for k, v in out["Label"].value_counts().sort_index().items()},
            "changed_from_to": changed_from_to(base["Label"], out["Label"], out["Label"].ne(base["Label"])),
            "threshold_rules": rules,
            "threshold_order": spec["threshold_order"],
            "base_public_reference": 0.8130,
            "lgbm_public_reference": 0.8106,
            "mini_public_reference": 0.8087,
            "threshold_note": "Rules use median fold thresholds from the restricted current-public-best OOF probe.",
        }
        payload.append(record)
        print(f"Wrote {output_path}: changes={record['changes_vs_base']} counts={record['label_counts']}")

    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.metadata}")


def load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["Id", "Label"]:
        raise ValueError(f"Invalid submission columns in {path}: {frame.columns.tolist()}")
    if frame["Id"].duplicated().any():
        raise ValueError(f"Duplicate Id values in {path}")
    frame["Label"] = frame["Label"].astype(int)
    return frame


def apply_consensus_rule(
    base: pd.DataFrame,
    lgbm: pd.DataFrame,
    mini: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    pred = base["Label"].copy()
    agree = lgbm["Label"].eq(mini["Label"])
    target = lgbm["Label"]
    source = base["Label"]
    mask_non2 = agree & target.ne(source) & source.ne(2) & target.isin(sorted(spec["non2_targets"]))
    mask_source2_to5 = agree & target.ne(source) & source.eq(2) & target.eq(5)
    mask = mask_non2 | (mask_source2_to5 if spec["include_source2_to5"] else False)
    pred.loc[mask] = target.loc[mask]
    return pred, mask, mask_non2, mask_source2_to5


def apply_consensus_groups(
    start: pd.Series,
    lgbm: pd.DataFrame,
    mini: pd.DataFrame,
    groups: list[tuple[int, int]],
) -> tuple[pd.Series, dict[str, int]]:
    pred = start.copy()
    group_counts: dict[str, int] = {}
    for from_label, to_label in groups:
        mask = pred.eq(from_label) & lgbm["Label"].eq(to_label) & mini["Label"].eq(to_label)
        pred.loc[mask] = to_label
        group_counts[f"{from_label}->{to_label}"] = int(mask.sum())
    return pred, group_counts


def load_threshold_context(args: argparse.Namespace, base: pd.DataFrame) -> dict[str, object]:
    centered = np.load(args.centered_test_npz, allow_pickle=True)
    round2 = np.load(args.round2_test_npz, allow_pickle=True)
    if "pred" not in centered.files:
        raise ValueError(f"{args.centered_test_npz} does not contain pred")
    if not np.array_equal(centered["pred"].astype(int), base["Label"].to_numpy(dtype=int)):
        raise ValueError("Centered test predictions differ from base submission order")
    if not np.array_equal(centered["classes"].astype(int), round2["classes"].astype(int)):
        raise ValueError("Class order differs between centered and round2 test bundles")
    if not np.array_equal(centered["file_id"].astype(int), round2["file_id"].astype(int)):
        raise ValueError("File order differs between centered and round2 test bundles")

    summary_payload = json.loads(args.threshold_summary_json.read_text(encoding="utf-8"))
    summary = pd.read_csv(args.threshold_summary_csv)
    classes = centered["classes"].astype(int)
    return {
        "centered": centered,
        "round2": round2,
        "summary": summary,
        "selected": [str(name) for name in summary_payload["selected"]],
        "classes": classes,
        "class_to_idx": {int(label): idx for idx, label in enumerate(classes)},
    }


def apply_threshold_rules(
    start: pd.Series,
    context: dict[str, object],
    max_rules: int | None = None,
    threshold_level: str = "median",
) -> tuple[pd.Series, list[dict[str, object]]]:
    pred = start.to_numpy(dtype=int).copy()
    selected = context["selected"] if max_rules is None else context["selected"][:max_rules]
    summary = context["summary"]
    rules: list[dict[str, object]] = []
    for name in selected:
        row = summary.loc[summary["name"] == name]
        if row.empty:
            raise ValueError(f"Missing threshold rule in summary CSV: {name}")
        rule = row.iloc[0]
        source_name = str(rule["source"])
        score_name = str(rule["score_name"])
        from_label = int(rule["from"])
        to_label = int(rule["to"])
        threshold = fold_threshold(str(rule["logs"]), threshold_level)
        proba = threshold_source_proba(source_name, context)
        scores = score_vector(proba, score_name, from_label, to_label, context["class_to_idx"])
        before = pred.copy()
        mask = (pred == from_label) & (scores >= threshold)
        pred[mask] = to_label
        rules.append(
            {
                "name": name,
                "source": source_name,
                "score_name": score_name,
                "from": from_label,
                "to": to_label,
                "threshold": float(threshold),
                "threshold_level": threshold_level,
                "step_changes": int(np.sum(pred != before)),
            }
        )
    return pd.Series(pred, index=start.index), rules


def threshold_source_proba(source_name: str, context: dict[str, object]) -> np.ndarray:
    centered = context["centered"]
    round2 = context["round2"]
    if source_name.startswith("blend_search/oof_blend_centered_meta_round2_best.npz:"):
        key = source_name.rsplit(":", 1)[-1]
        if key not in centered.files:
            raise ValueError(f"{key} missing from centered test bundle")
        return centered[key].astype(float)
    if source_name == "model_search_position_proba/oof_position_lgbm_leaves63.npz:proba":
        return round2["lgbm_leaves63_proba"].astype(float)
    raise ValueError(f"No test probability mapping for threshold source: {source_name}")


def fold_threshold(raw_logs: str, level: str) -> float:
    logs = ast.literal_eval(raw_logs)
    values = np.array([entry["threshold"] for entry in logs if entry.get("threshold") is not None], dtype=float)
    if values.size == 0:
        raise ValueError("Threshold rule has no finite fold thresholds")
    if level == "median":
        return float(statistics.median(values.tolist()))
    if level == "q75":
        return float(np.quantile(values, 0.75))
    if level == "q90":
        return float(np.quantile(values, 0.90))
    if level == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown threshold level: {level}")


def score_vector(
    proba: np.ndarray,
    score_name: str,
    from_label: int,
    to_label: int,
    class_to_idx: dict[int, int],
) -> np.ndarray:
    from_idx = class_to_idx[from_label]
    to_idx = class_to_idx[to_label]
    if score_name == "p_to":
        return proba[:, to_idx]
    if score_name == "margin_from":
        return proba[:, to_idx] - proba[:, from_idx]
    if score_name == "margin_other":
        return proba[:, to_idx] - np.max(np.delete(proba, to_idx, axis=1), axis=1)
    if score_name == "log_margin_from":
        log_proba = np.log(np.clip(proba, 1e-9, 1.0))
        return log_proba[:, to_idx] - log_proba[:, from_idx]
    raise ValueError(f"Unknown score type: {score_name}")


def changed_from_to(base: pd.Series, pred: pd.Series, mask: pd.Series) -> dict[str, int]:
    out: dict[str, int] = {}
    for from_label, to_label in zip(base.loc[mask].astype(int), pred.loc[mask].astype(int), strict=True):
        key = f"{from_label}->{to_label}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
