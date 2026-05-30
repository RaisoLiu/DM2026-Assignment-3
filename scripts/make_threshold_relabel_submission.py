#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from explore_models import META_COLS, add_context_features, aligned_proba, fit_model, make_models
from row_level_lgbm import SIGNAL_COLS, add_row_features, aggregate_by_file, load_or_build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate selected OOF threshold relabel rules to a test CSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--base-test-npz", type=Path, default=Path("artifacts/blend_search/test_blend_local_best_oof0774907.npz"))
    parser.add_argument("--centered-test-npz", type=Path, default=Path("artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/local_best_oof_0802784/summary.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("artifacts/proba_threshold_all_pairs_all_sources_after_0797693/summary.csv"))
    parser.add_argument("--train-feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--test-feature-cache", type=Path, default=Path("artifacts/features/test_features.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--save-npz", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-row-lgbm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = np.load(args.base_test_npz, allow_pickle=True)
    centered = np.load(args.centered_test_npz, allow_pickle=True)
    classes = base["classes"].astype(int)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    file_id = base["file_id"].astype(int)
    user_id = base["user_id"].astype(str) if "user_id" in base.files else centered["user_id"].astype(str)
    pred = base["pred"].astype(int).copy()
    original_pred = pred.copy()

    train_frame = pd.read_csv(args.train_feature_cache)
    test_frame = pd.read_csv(args.test_feature_cache)
    source_cache: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "base_test_npz": str(args.base_test_npz),
        "centered_test_npz": str(args.centered_test_npz),
        "summary_json": str(args.summary_json),
        "summary_csv": str(args.summary_csv),
        "oof_reference_macro_f1": json.loads(args.summary_json.read_text(encoding="utf-8"))["macro_f1"],
        "rules": [],
        "skipped_rules": [],
    }

    selected = json.loads(args.summary_json.read_text(encoding="utf-8"))["selected"]
    summary = pd.read_csv(args.summary_csv)
    for rule_name in selected:
        row = summary.loc[summary["name"] == rule_name]
        if row.empty:
            metadata["skipped_rules"].append({"name": rule_name, "reason": "missing from summary csv"})
            continue
        rule = row.iloc[0]
        source_name = str(rule["source"])
        score_name = str(rule["score_name"])
        from_label = int(rule["from"])
        to_label = int(rule["to"])
        threshold = median_threshold(str(rule["logs"]))
        if threshold is None:
            metadata["skipped_rules"].append({"name": rule_name, "reason": "no finite threshold"})
            continue
        proba = load_source_proba(
            source_name=source_name,
            base=base,
            centered=centered,
            classes=classes,
            file_id=file_id,
            train_frame=train_frame,
            test_frame=test_frame,
            data_dir=args.data_dir,
            seed=args.seed,
            source_cache=source_cache,
            skip_row_lgbm=args.skip_row_lgbm,
        )
        if proba is None:
            metadata["skipped_rules"].append({"name": rule_name, "source": source_name, "reason": "no test translation"})
            continue
        scores = score_vector(proba, score_name, from_label, to_label, class_to_idx)
        mask = (pred == from_label) & (scores >= threshold)
        before = pred.copy()
        pred[mask] = to_label
        metadata["rules"].append(
            {
                "name": rule_name,
                "source": source_name,
                "score_name": score_name,
                "from": from_label,
                "to": to_label,
                "threshold": float(threshold),
                "changes": int(np.sum(before != pred)),
                "source_note": source_note(source_name),
            }
        )

    submission = merge_submission(args.data_dir, file_id, pred)
    validate_submission(args.data_dir, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    digest = sha256_file(args.output)
    counts = {int(k): int(v) for k, v in submission["Label"].value_counts().sort_index().items()}
    metadata.update(
        {
            "output": str(args.output),
            "sha256": digest,
            "base_label_counts": counts_from_array(original_pred),
            "label_counts": counts,
            "total_changes_vs_base": int(np.sum(pred != original_pred)),
            "note": "Leaderboard probe: selected OOF threshold rules translated with median fold thresholds; some sources are full-train or approximate test counterparts.",
        }
    )
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            pred=pred,
            base_pred=original_pred,
            classes=classes,
            file_id=file_id,
            user_id=user_id,
        )

    print(f"Wrote {args.output} rows={len(submission)} sha256={digest}", flush=True)
    print(f"changes_vs_base={metadata['total_changes_vs_base']}", flush=True)
    print(f"label_counts={counts}", flush=True)
    print(f"rules_applied={len(metadata['rules'])} skipped={len(metadata['skipped_rules'])}", flush=True)


def median_threshold(raw_logs: str) -> float | None:
    logs = ast.literal_eval(raw_logs)
    values = [entry["threshold"] for entry in logs if entry.get("threshold") is not None]
    return float(statistics.median(values)) if values else None


def load_source_proba(
    *,
    source_name: str,
    base: np.lib.npyio.NpzFile,
    centered: np.lib.npyio.NpzFile,
    classes: np.ndarray,
    file_id: np.ndarray,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    data_dir: Path,
    seed: int,
    source_cache: dict[str, np.ndarray],
    skip_row_lgbm: bool,
) -> np.ndarray | None:
    if source_name in source_cache:
        return source_cache[source_name]

    if source_name.startswith("blend_search/oof_blend_centered_meta_round2_best.npz:"):
        key = source_name.rsplit(":", 1)[-1]
        proba = centered[key].astype(float) if key in centered.files else None
    elif source_name == "blend_search/oof_blend_centered_meta_additive_sidecars_best.npz:proba":
        proba = centered["proba"].astype(float)
    elif source_name == "model_search_position_shift/oof_position_shift_lgbm_base.npz:proba":
        proba = train_tabular_source(train_frame, test_frame, file_id, classes, "position_shift", "lgbm_base", seed)
    elif source_name == "model_search_position_cw_2x5/oof_position_lgbm_leaves63.npz:proba":
        proba = train_tabular_source(
            train_frame,
            test_frame,
            file_id,
            classes,
            "position",
            "lgbm_leaves63",
            seed,
            class_weight_multipliers={2: 5.0},
        )
    elif source_name == "model_search_position_cat_depth8/oof_position_catboost_depth8.npz:proba":
        proba = train_tabular_source(train_frame, test_frame, file_id, classes, "position", "catboost_depth8", seed)
    elif source_name == "blend_search/oof_blend_centered_meta_with_row_lgbm_source.npz:row_lgbm_proba":
        proba = None if skip_row_lgbm else train_row_lgbm_source(data_dir, file_id, classes, seed)
    elif source_name == "model_search_position_proba/oof_position_lgbm_leaves63.npz:proba":
        proba = train_tabular_source(train_frame, test_frame, file_id, classes, "position", "lgbm_leaves63", seed)
    elif source_name == "class5_softmix_position_lgbm/oof_softmix_lam0.075.npz:proba":
        position = load_source_proba(
            source_name="model_search_position_proba/oof_position_lgbm_leaves63.npz:proba",
            base=base,
            centered=centered,
            classes=classes,
            file_id=file_id,
            train_frame=train_frame,
            test_frame=test_frame,
            data_dir=data_dir,
            seed=seed,
            source_cache=source_cache,
            skip_row_lgbm=skip_row_lgbm,
        )
        proba = class5_softmix(centered["proba"].astype(float), position, lam=0.075) if position is not None else None
    else:
        proba = None

    if proba is not None:
        source_cache[source_name] = proba
    return proba


def train_tabular_source(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    file_id: np.ndarray,
    classes: np.ndarray,
    context: str,
    model_name: str,
    seed: int,
    class_weight_multipliers: dict[int, float] | None = None,
) -> np.ndarray:
    train = add_context_features(train_frame.copy(), context)
    test = add_context_features(test_frame.copy(), context)
    features = [col for col in train.columns if col not in META_COLS]
    for col in [col for col in features if col not in test.columns]:
        test[col] = 0.0
    aligned_test = align_frame(test, file_id)
    y = train["label"].astype(int).to_numpy()
    model = clone(make_models(seed)[model_name])
    sample_weight = compute_sample_weight("balanced", y)
    if class_weight_multipliers:
        sample_weight = sample_weight * np.array([class_weight_multipliers.get(int(label), 1.0) for label in y])
    print(f"Training {source_note_from_parts(context, model_name)}: train={train[features].shape}; test={aligned_test[features].shape}", flush=True)
    fit_model(model, train[features], y, sample_weight)
    return aligned_proba(model, aligned_test[features], classes)


def train_row_lgbm_source(data_dir: Path, file_id: np.ndarray, classes: np.ndarray, seed: int) -> np.ndarray:
    from lightgbm import LGBMClassifier

    rows, files = load_or_build(data_dir, Path("artifacts/row_level"))
    test_rows, test_files = load_test_rows(data_dir)
    feature_cols = [col for col in rows.columns if col not in {"file_id", "user_id", "label"}]
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.02,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    y = rows["label"].astype(int).to_numpy()
    sample_weight = compute_sample_weight("balanced", y)
    print(f"Training row_lgbm: train_rows={len(rows)}; test_rows={len(test_rows)}", flush=True)
    model.fit(rows[feature_cols], y, sample_weight=sample_weight)
    row_proba = model.predict_proba(test_rows[feature_cols])
    agg = aggregate_by_file(row_proba, test_rows["file_id"].to_numpy(), classes)
    out = np.zeros((len(test_files), len(classes)), dtype=float)
    for idx, fid in enumerate(test_files["file_id"].astype(int).to_numpy()):
        out[idx] = agg[int(fid)]
    aligned_files = test_files["file_id"].astype(int).to_numpy()
    order = pd.Series(np.arange(len(aligned_files)), index=aligned_files).loc[file_id].to_numpy()
    return out[order]


def load_test_rows(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    files = []
    for path in sorted((data_dir / "test").rglob("*.csv")):
        df = pd.read_csv(path).sort_values("index").reset_index(drop=True)
        fid = int(df["file_id"].iloc[0])
        user = path.parent.name
        row = df[["index", *SIGNAL_COLS]].copy()
        row["file_id"] = fid
        row["user_id"] = user
        add_row_features(row)
        rows.append(row)
        files.append({"file_id": fid, "user_id": user})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(files)


def class5_softmix(base: np.ndarray, position: np.ndarray, lam: float) -> np.ndarray:
    proba = base.copy()
    p5 = (1.0 - lam) * base[:, 5] + lam * position[:, 5]
    scale = (1.0 - p5) / np.clip(1.0 - base[:, 5], 1e-12, None)
    proba[:, :5] *= scale.reshape(-1, 1)
    proba[:, 5] = p5
    return proba / proba.sum(axis=1, keepdims=True)


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


def align_frame(frame: pd.DataFrame, file_id: np.ndarray) -> pd.DataFrame:
    order = pd.Series(np.arange(len(frame)), index=frame["file_id"].astype(int)).loc[file_id].to_numpy()
    return frame.iloc[order].reset_index(drop=True)


def merge_submission(data_dir: Path, file_id: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    pred_by_id = pd.DataFrame({"_join_id": pd.Series(file_id).map(normalize_id).astype(str), "Label": pred})
    sample = load_sample_submission(data_dir)
    submission = sample[["Id"]].copy()
    submission["_join_id"] = submission["Id"].map(normalize_id).astype(str)
    submission = submission.merge(pred_by_id, on="_join_id", how="left").drop(columns=["_join_id"])
    if submission["Label"].isna().any():
        missing_ids = submission.loc[submission["Label"].isna(), "Id"].head().tolist()
        raise ValueError(f"Missing predictions for sample_submission IDs: {missing_ids}")
    submission["Label"] = submission["Label"].astype(int)
    return submission


def validate_submission(data_dir: Path, submission: pd.DataFrame) -> None:
    sample = load_sample_submission(data_dir)
    if list(submission.columns) != ["Id", "Label"]:
        raise ValueError(f"Invalid columns: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, got {len(submission)}")
    if submission["Id"].duplicated().any():
        raise ValueError("Duplicate Id values in submission")
    if not submission["Id"].equals(sample["Id"]):
        raise ValueError("Submission Id order does not match sample_submission.csv")


def counts_from_array(values: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(values.astype(int), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_note(source_name: str) -> str:
    if "additive_sidecars" in source_name:
        return "approximated with centered-meta full-train proba"
    if "class5_softmix" in source_name:
        return "approximated from centered-meta and full-train position LGBM"
    if "row_lgbm" in source_name:
        return "full-train row LGBM"
    if source_name.startswith("model_search_"):
        return "full-train tabular counterpart"
    return "saved full-train counterpart"


def source_note_from_parts(context: str, model_name: str) -> str:
    return f"{context}/{model_name}"


if __name__ == "__main__":
    main()
