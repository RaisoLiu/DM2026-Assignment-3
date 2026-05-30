#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dm2026_asg3.data import load_sample_submission, normalize_id
from evaluate_sequence_smoothing import estimate_transition_model, tune_viterbi_params, viterbi_predict_by_user
from explore_models import META_COLS, add_context_features, aligned_proba, fit_model, make_models
from train_oof_stacker import build_meta_features, fit_with_optional_weight, aligned_proba as stacker_aligned_proba


COMPONENT_NAMES = (
    "xgb",
    "cat",
    "xgb_d6",
    "mini10",
    "mini20",
    "miniraw",
    "multi",
    "event_lgbm",
    "meta_lgbm",
    "meta_xgb",
    "lgbm47",
)
META_INPUT_NAMES = (
    "xgb",
    "cat",
    "xgb_d6",
    "mini10",
    "mini20",
    "miniraw",
    "multi",
    "event_lgbm",
    "lgbm47",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the centered-meta Viterbi submission from saved OOF artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-feature-cache", type=Path, default=Path("artifacts/features/train_features.csv"))
    parser.add_argument("--test-feature-cache", type=Path, default=Path("artifacts/features/test_features.csv"))
    parser.add_argument("--oof-blend", type=Path, default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"))
    parser.add_argument("--test-base-proba", type=Path, default=Path("artifacts/blend_search/test_blend_rocket_event_oof07610.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-proba", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oof = np.load(args.oof_blend, allow_pickle=True)
    classes = oof["classes"].astype(int)
    y = oof["label"].astype(int)
    oof_file_id = oof["file_id"].astype(int)
    oof_user_id = oof["user_id"].astype(str)
    model_names = [str(name) for name in oof["model_names"]]
    model_weights = oof["model_weights"].astype(float)
    class_weights = oof["class_weights"].astype(float)
    if model_names != list(COMPONENT_NAMES):
        raise ValueError(f"Unexpected component order: {model_names}")

    test_bundle = np.load(args.test_base_proba, allow_pickle=True)
    test_file_id = test_bundle["file_id"].astype(int)
    test_frame = add_context_features(pd.read_csv(args.test_feature_cache), "position")
    train_frame = add_context_features(pd.read_csv(args.train_feature_cache), "position")
    test_user_id = align_series(test_frame, test_file_id, "user_id").astype(str)

    train_components = {name: oof[f"{name}_proba"].astype(float) for name in COMPONENT_NAMES}
    test_components = load_saved_test_components(test_bundle, classes, test_file_id)
    test_components["lgbm47"] = train_full_lgbm47(train_frame, test_frame, test_file_id, classes, args.seed)

    meta_train = build_meta_matrix(train_components, oof_file_id, oof_user_id)
    meta_test = build_meta_matrix(test_components, test_file_id, test_user_id)
    meta_lgbm, meta_xgb = train_full_meta_models(meta_train, meta_test, y, classes, args.seed)
    test_components["meta_lgbm"] = meta_lgbm
    test_components["meta_xgb"] = meta_xgb

    test_proba = np.zeros((len(test_file_id), len(classes)), dtype=float)
    for name, weight in zip(model_names, model_weights, strict=True):
        test_proba += float(weight) * test_components[name]

    params = tune_viterbi_params(
        proba=oof["proba"].astype(float),
        y=y,
        classes=classes,
        file_ids=oof_file_id,
        user_ids=oof_user_id,
        class_weights=class_weights,
        alpha_grid=(0.1, 0.3, 1.0, 3.0),
        beta_grid=(0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30),
        stay_grid=(0.0,),
    )
    transition, start = estimate_transition_model(
        y=y,
        classes=classes,
        file_ids=oof_file_id,
        user_ids=oof_user_id,
        alpha=params["alpha"],
    )
    pred = viterbi_predict_by_user(
        proba=test_proba,
        classes=classes,
        file_ids=test_file_id,
        user_ids=test_user_id,
        class_weights=class_weights,
        transition=transition,
        start=start,
        beta=params["beta"],
        stay_bonus=params["stay_bonus"],
    )

    submission = merge_submission(args.data_dir, test_file_id, pred)
    validate_submission(args.data_dir, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    digest = sha256_file(args.output)
    label_counts = {int(k): int(v) for k, v in submission["Label"].value_counts().sort_index().items()}

    global_viterbi_pred = viterbi_predict_by_user(
        proba=oof["proba"].astype(float),
        classes=classes,
        file_ids=oof_file_id,
        user_ids=oof_user_id,
        class_weights=class_weights,
        transition=transition,
        start=start,
        beta=params["beta"],
        stay_bonus=params["stay_bonus"],
    )
    metadata = {
        "source_oof_blend": str(args.oof_blend),
        "fold_fair_viterbi_macro_f1_reference": 0.769308432188306,
        "global_oof_viterbi_macro_f1": float(f1_score(y, global_viterbi_pred, average="macro")),
        "global_oof_base_macro_f1": float(f1_score(y, classes[np.argmax(oof["proba"], axis=1)], average="macro")),
        "global_oof_class_weight_macro_f1": float(
            f1_score(y, classes[np.argmax(oof["proba"] * class_weights.reshape(1, -1), axis=1)], average="macro")
        ),
        "model_names": model_names,
        "model_weights": [float(v) for v in model_weights],
        "class_weights": [float(v) for v in class_weights],
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "stay_bonus": float(params["stay_bonus"]),
        "label_counts": label_counts,
        "sha256": digest,
        "output": str(args.output),
        "note": "Full-train test translation of centered-meta blend; meta_lgbm/meta_xgb are retrained on all OOF meta features.",
    }
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_proba:
        args.save_proba.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_proba,
            proba=test_proba,
            classes=classes,
            file_id=test_file_id,
            user_id=test_user_id,
            pred=pred,
            model_names=np.array(model_names, dtype=object),
            model_weights=model_weights,
            class_weights=class_weights,
            **{f"{name}_proba": test_components[name] for name in COMPONENT_NAMES},
        )

    print(f"Wrote {args.output} rows={len(submission)} sha256={digest}", flush=True)
    print(f"global_oof_viterbi_macro_f1={metadata['global_oof_viterbi_macro_f1']:.6f}", flush=True)
    print(f"alpha={params['alpha']} beta={params['beta']} stay_bonus={params['stay_bonus']}", flush=True)
    print(f"label_counts={label_counts}", flush=True)


def load_saved_test_components(bundle: np.lib.npyio.NpzFile, classes: np.ndarray, file_id: np.ndarray) -> dict[str, np.ndarray]:
    key_map = {
        "xgb": "xgb_proba",
        "cat": "cat_proba",
        "xgb_d6": "xgb_d6_proba",
        "mini10": "mini10_proba",
        "mini20": "mini20_proba",
        "miniraw": "miniraw_proba",
        "multi": "multi_proba",
        "event_lgbm": "event_lgbm_proba",
    }
    if not np.array_equal(bundle["classes"].astype(int), classes):
        raise ValueError("Class order mismatch in saved test components")
    if not np.array_equal(bundle["file_id"].astype(int), file_id):
        raise ValueError("File order mismatch in saved test components")
    return {name: bundle[key].astype(float) for name, key in key_map.items()}


def train_full_lgbm47(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_file_id: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> np.ndarray:
    features = [col for col in train_frame.columns if col not in META_COLS]
    for col in [col for col in features if col not in test_frame.columns]:
        test_frame[col] = 0.0
    test_aligned = align_frame(test_frame, test_file_id)
    y = train_frame["label"].astype(int).to_numpy()
    model = clone(make_models(seed)["lgbm_leaves47"])
    sample_weight = compute_sample_weight("balanced", y)
    print(f"Training lgbm_leaves47 full model: train={train_frame[features].shape}; test={test_aligned[features].shape}", flush=True)
    fit_model(model, train_frame[features], y, sample_weight)
    return aligned_proba(model, test_aligned[features], classes)


def build_meta_matrix(components: dict[str, np.ndarray], file_id: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    bundles = [{"proba": components[name]} for name in META_INPUT_NAMES]
    return build_meta_features(
        bundles,
        file_id,
        user_id,
        include_log=True,
        include_aggregate=True,
        include_seq_context=True,
        neighbor_shifts=[],
        neighbor_rolling=[],
    )


def train_full_meta_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    specs = make_meta_models(seed)
    out = []
    sample_weight = compute_sample_weight("balanced", y)
    for name in ("meta_lgbm", "meta_xgb"):
        model = specs[name]
        print(f"Training {name}: train={x_train.shape}; test={x_test.shape}", flush=True)
        fit_with_optional_weight(model, x_train, y, sample_weight)
        out.append(stacker_aligned_proba(model, x_test, classes))
    return out[0], out[1]


def make_meta_models(seed: int) -> dict[str, object]:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "meta_lgbm": LGBMClassifier(
            objective="multiclass",
            num_class=6,
            n_estimators=650,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.45,
            random_state=seed + 205,
            n_jobs=-1,
            verbosity=-1,
        ),
        "meta_xgb": XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=420,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.04,
            reg_lambda=0.45,
            random_state=seed + 206,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        ),
    }


def align_frame(frame: pd.DataFrame, file_id: np.ndarray) -> pd.DataFrame:
    frame = frame.copy()
    order = {int(fid): idx for idx, fid in enumerate(file_id)}
    frame["_order"] = frame["file_id"].astype(int).map(order)
    if frame["_order"].isna().any():
        raise ValueError("Feature cache is missing some requested file_id values")
    frame = frame.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    if not np.array_equal(frame["file_id"].astype(int).to_numpy(), file_id.astype(int)):
        raise ValueError("Failed to align feature cache to file_id order")
    return frame


def align_series(frame: pd.DataFrame, file_id: np.ndarray, column: str) -> pd.Series:
    return align_frame(frame, file_id)[column]


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
    labels = set(submission["Label"].astype(int).unique())
    if not labels.issubset(set(range(6))):
        raise ValueError(f"Invalid labels: {sorted(labels)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
