#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


DEFAULT_SOURCES = (
    "artifacts/blend_search/oof_blend_centered_meta_round2_best.npz",
    "artifacts/blend_search/oof_blend_centered_meta_additive_sidecars_best.npz",
    "artifacts/model_search_position_xgb/oof_position_xgb_base.npz",
    "artifacts/model_search_position_proba/oof_position_lgbm_leaves63.npz",
    "artifacts/model_search_position_fixedfold_tree_extra/oof_position_extra_trees_500.npz",
    "artifacts/model_search_position_cat_xgb/oof_position_catboost_base.npz",
    "artifacts/model_search_position_xgb_gpu/oof_position_xgb_gpu_depth5.npz",
    "artifacts/model_search_event_position/oof_position_lgbm_base.npz",
    "artifacts/rocket_probe/oof_minirocket_augmented_k3000.npz",
    "artifacts/rocket_oof/oof_minirocket_augmented_k20000.npz",
    "artifacts/oof_logit_stacker_components/oof_logit_C0.003.npz",
    "artifacts/oof_logit_stacker_components/oof_logit_C3.npz",
    "artifacts/catch22_oof/oof_catch22_raw_lgbm_c22.npz",
    "artifacts/sequence_cnn_aug_gpu_shuffle_cw/oof_sequence_cnn.npz",
    "artifacts/sequence_cnn_aug_gpu_sampler_weighted_nocw/oof_sequence_cnn.npz",
)

DEFAULT_TRANSITIONS = "2:1,2:3,2:5,5:2,5:3,1:2,1:3,1:5,1:0,3:1,3:2,3:5,5:1,0:1,3:4,4:3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-aware probability-threshold relabel probe.")
    parser.add_argument("--base-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", default=None, help="Aligned OOF npz source; repeatable.")
    parser.add_argument("--all-sources", action="store_true", help="Scan artifacts/**/*.npz for aligned probability sources.")
    parser.add_argument("--transitions", default=DEFAULT_TRANSITIONS, help="Comma list of from:to label transitions.")
    parser.add_argument("--max-rules", type=int, default=10)
    parser.add_argument("--min-quantile", type=float, default=0.55)
    parser.add_argument("--max-quantile", type=float, default=0.995)
    parser.add_argument("--n-quantiles", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(args.base_npz, allow_pickle=True)
    y = base["label"].astype(int)
    base_pred = base["pred"].astype(int)
    file_ids = base["file_id"].astype(int)
    user_ids = base["user_id"].astype(str)
    folds = base["fold"].astype(int)
    classes = base["classes"].astype(int)
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    y_idx = np.array([class_to_idx[int(value)] for value in y], dtype=int)
    base_score = float(f1_score(y, base_pred, average="macro"))

    source_paths = discover_sources() if args.all_sources else tuple(args.source) if args.source else DEFAULT_SOURCES
    sources = load_sources(source_paths, y, file_ids, user_ids, classes)
    transitions = parse_transitions(args.transitions)
    quantiles = np.linspace(args.min_quantile, args.max_quantile, args.n_quantiles)

    rows: list[dict[str, object]] = []
    pred_outputs: dict[str, np.ndarray] = {}
    for source_name, proba in sources:
        log_proba = np.log(np.clip(proba, 1e-9, 1.0))
        for from_label, to_label in transitions:
            candidate_idx = np.flatnonzero(base_pred == from_label)
            if len(candidate_idx) == 0:
                continue
            score_map = {
                "p_to": proba[:, class_to_idx[to_label]],
                "margin_from": proba[:, class_to_idx[to_label]] - proba[:, class_to_idx[from_label]],
                "margin_other": proba[:, class_to_idx[to_label]]
                - np.max(np.delete(proba, class_to_idx[to_label], axis=1), axis=1),
                "log_margin_from": log_proba[:, class_to_idx[to_label]]
                - log_proba[:, class_to_idx[from_label]],
            }
            for score_name, scores in score_map.items():
                pred, logs = evaluate_rule(
                    y=y,
                    y_idx=y_idx,
                    base_pred=base_pred,
                    folds=folds,
                    classes=classes,
                    candidate_idx=candidate_idx,
                    scores=scores,
                    from_label=from_label,
                    to_label=to_label,
                    from_idx=class_to_idx[from_label],
                    to_idx=class_to_idx[to_label],
                    quantiles=quantiles,
                )
                score = float(f1_score(y, pred, average="macro"))
                changes = int(np.sum(pred != base_pred))
                if score > base_score + 1e-12 or changes:
                    name = f"{source_name}:{score_name}:{from_label}->{to_label}"
                    rows.append(
                        {
                            "name": name,
                            "source": source_name,
                            "score_name": score_name,
                            "from": from_label,
                            "to": to_label,
                            "macro_f1": score,
                            "delta": score - base_score,
                            "changes": changes,
                            "logs": logs,
                        }
                    )
                    pred_outputs[name] = pred

    rows = sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    best_pred, selected = greedy_combine(rows, pred_outputs, y, base_pred, base_score, args.max_rules)
    best_score = float(f1_score(y, best_pred, average="macro"))

    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(classification_report(y, best_pred, output_dict=True, zero_division=0)).T.to_csv(
        args.output_dir / "best_report.csv"
    )
    np.savez_compressed(
        args.output_dir / "best_predictions.npz",
        pred=best_pred,
        base_pred=base_pred,
        label=y,
        file_id=file_ids,
        user_id=user_ids,
        fold=folds,
        classes=classes,
    )
    payload = {
        "macro_f1": best_score,
        "base_macro_f1": base_score,
        "delta": best_score - base_score,
        "changes": int(np.sum(best_pred != base_pred)),
        "selected": selected,
        "top_singles": rows[:30],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(pd.DataFrame(rows).head(30).to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def load_sources(
    paths: tuple[str, ...],
    y: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    classes: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    sources: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for raw_path in paths:
        path = Path(raw_path)
        data = np.load(path, allow_pickle=True)
        if not aligned(data, y, file_ids, user_ids, classes):
            continue
        candidates: list[tuple[str, np.ndarray]] = []
        if "proba" in data.files and data["proba"].shape == (len(y), len(classes)):
            candidates.append(("proba", data["proba"].astype(float)))
        if "model_names" in data.files:
            for model_name in data["model_names"].astype(str).tolist():
                key = f"{model_name}_proba"
                if key in data.files and data[key].shape == (len(y), len(classes)):
                    candidates.append((key, data[key].astype(float)))
        for key, proba in candidates:
            fingerprint = np.round(proba, 6).tobytes()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            sources.append((f"{path.parent.name}/{path.name}:{key}", proba))
    return sources


def discover_sources() -> tuple[str, ...]:
    return tuple(str(path) for path in sorted(Path("artifacts").glob("**/*.npz")))


def aligned(
    data: np.lib.npyio.NpzFile,
    y: np.ndarray,
    file_ids: np.ndarray,
    user_ids: np.ndarray,
    classes: np.ndarray,
) -> bool:
    return (
        all(key in data.files for key in ("label", "file_id", "user_id", "classes"))
        and np.array_equal(data["label"].astype(int), y)
        and np.array_equal(data["file_id"].astype(int), file_ids)
        and np.array_equal(data["user_id"].astype(str), user_ids)
        and np.array_equal(data["classes"].astype(int), classes)
    )


def parse_transitions(raw: str) -> tuple[tuple[int, int], ...]:
    transitions = []
    for item in raw.split(","):
        if not item.strip():
            continue
        left, right = item.split(":", 1)
        transitions.append((int(left), int(right)))
    return tuple(transitions)


def evaluate_rule(
    *,
    y: np.ndarray,
    y_idx: np.ndarray,
    base_pred: np.ndarray,
    folds: np.ndarray,
    classes: np.ndarray,
    candidate_idx: np.ndarray,
    scores: np.ndarray,
    from_label: int,
    to_label: int,
    from_idx: int,
    to_idx: int,
    quantiles: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    pred = base_pred.copy()
    logs: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds)):
        train_rows = folds != fold
        train_idx = candidate_idx[folds[candidate_idx] != fold]
        valid_idx = candidate_idx[folds[candidate_idx] == fold]
        threshold, train_score = tune_threshold(
            y=y,
            y_idx=y_idx,
            base_pred=base_pred,
            train_rows=train_rows,
            train_idx=train_idx,
            scores=scores,
            classes=classes,
            from_idx=from_idx,
            to_idx=to_idx,
            quantiles=quantiles,
        )
        if threshold is None:
            logs.append({"fold": int(fold), "threshold": None, "changes": 0, "train_score": train_score})
            continue
        changed = valid_idx[scores[valid_idx] >= threshold]
        pred[changed] = to_label
        logs.append(
            {
                "fold": int(fold),
                "threshold": float(threshold),
                "changes": int(len(changed)),
                "train_score": train_score,
            }
        )
    return pred, logs


def tune_threshold(
    *,
    y: np.ndarray,
    y_idx: np.ndarray,
    base_pred: np.ndarray,
    train_rows: np.ndarray,
    train_idx: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    from_idx: int,
    to_idx: int,
    quantiles: np.ndarray,
) -> tuple[float | None, float]:
    cm = confusion_matrix(y[train_rows], base_pred[train_rows], labels=classes).astype(int)
    best_score = macro_f1_from_cm(cm)
    best_threshold: float | None = None
    if len(train_idx) == 0:
        return None, best_score
    train_scores = scores[train_idx]
    for threshold in np.unique(np.quantile(train_scores, quantiles)):
        selected = train_idx[train_scores >= threshold]
        if len(selected) == 0:
            continue
        counts = np.bincount(y_idx[selected], minlength=len(classes))
        trial = cm.copy()
        trial[:, from_idx] -= counts
        trial[:, to_idx] += counts
        score = macro_f1_from_cm(trial)
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def macro_f1_from_cm(cm: np.ndarray) -> float:
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    true_pos = np.diag(cm)
    denom = rows + cols
    f1 = np.divide(2 * true_pos, denom, out=np.zeros(len(cm), dtype=float), where=denom > 0)
    return float(f1.mean())


def greedy_combine(
    rows: list[dict[str, object]],
    pred_outputs: dict[str, np.ndarray],
    y: np.ndarray,
    base_pred: np.ndarray,
    base_score: float,
    max_rules: int,
) -> tuple[np.ndarray, list[str]]:
    best_pred = base_pred.copy()
    best_score = base_score
    selected: list[str] = []
    locked = np.zeros(len(y), dtype=bool)
    positives = [row for row in rows if float(row["macro_f1"]) > base_score + 1e-12]
    while len(selected) < max_rules:
        step_name = ""
        step_pred = None
        step_score = best_score
        for row in positives:
            name = str(row["name"])
            if name in selected:
                continue
            candidate = (pred_outputs[name] != base_pred) & ~locked
            if not np.any(candidate):
                continue
            trial = best_pred.copy()
            trial[candidate] = pred_outputs[name][candidate]
            score = float(f1_score(y, trial, average="macro"))
            if score > step_score + 1e-12:
                step_name = name
                step_pred = trial
                step_score = score
        if step_pred is None:
            break
        candidate = (pred_outputs[step_name] != base_pred) & ~locked
        locked[candidate] = True
        best_pred = step_pred
        best_score = step_score
        selected.append(step_name)
    return best_pred, selected


if __name__ == "__main__":
    main()
