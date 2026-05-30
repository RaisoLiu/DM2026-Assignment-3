#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a grouped-OOF linear-chain CRF over saved OOF probability features.")
    parser.add_argument("--oof-npz", type=Path, default=Path("artifacts/blend_search/oof_blend_centered_meta_round2_best.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequence_crf_centered_meta"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=650)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=0.20)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--feature-mode", choices=["base", "components"], default="components")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


class LinearChainCRF(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(n_features, n_classes)
        self.start = nn.Parameter(torch.zeros(n_classes))
        self.transition = nn.Parameter(torch.zeros(n_classes, n_classes))

    def emission_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.emission(self.dropout(x))

    def neg_log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emit = self.emission_scores(x)
        log_z = self._log_partition(emit)
        gold = self.start[y[0]] + emit[0, y[0]]
        if len(y) > 1:
            gold = gold + self.transition[y[:-1], y[1:]].sum() + emit[1:, :].gather(1, y[1:, None]).sum()
        return log_z - gold

    def _log_partition(self, emit: torch.Tensor) -> torch.Tensor:
        score = self.start + emit[0]
        for t in range(1, emit.shape[0]):
            score = torch.logsumexp(score[:, None] + self.transition + emit[t][None, :], dim=0)
        return torch.logsumexp(score, dim=0)

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        emit = self.emission_scores(x)
        score = self.start + emit[0]
        back = []
        for t in range(1, emit.shape[0]):
            candidates = score[:, None] + self.transition
            prev = torch.argmax(candidates, dim=0)
            score = candidates[prev, torch.arange(emit.shape[1], device=emit.device)] + emit[t]
            back.append(prev)
        path = [int(torch.argmax(score).item())]
        for prev in reversed(back):
            path.append(int(prev[path[-1]].item()))
        path.reverse()
        return torch.tensor(path, dtype=torch.long, device=x.device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = np.load(args.oof_npz, allow_pickle=True)
    classes = bundle["classes"].astype(int)
    y = bundle["label"].astype(int)
    folds = bundle["fold"].astype(int)
    file_ids = bundle["file_id"].astype(int)
    users = bundle["user_id"].astype(str)
    features = make_features(bundle, args.feature_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; feature_mode={args.feature_mode}; X={features.shape}; classes={classes.tolist()}", flush=True)

    oof_pred = np.empty_like(y)
    oof_emission = np.zeros((len(y), len(classes)), dtype=np.float32)
    fold_rows = []
    fold_values = sorted(np.unique(folds))
    if args.max_folds:
        fold_values = fold_values[: args.max_folds]
    for fold in fold_values:
        train_mask = folds != fold
        valid_mask = folds == fold
        train_sequences = make_sequences(features, y, file_ids, users, train_mask)
        valid_sequences = make_sequences(features, y, file_ids, users, valid_mask)
        if args.standardize:
            mean = features[train_mask].mean(axis=0, keepdims=True)
            std = features[train_mask].std(axis=0, keepdims=True) + 1e-6
        else:
            mean = np.zeros((1, features.shape[1]), dtype=np.float32)
            std = np.ones((1, features.shape[1]), dtype=np.float32)
        train_tensors = to_tensors(train_sequences, mean, std, device)
        valid_tensors = to_tensors(valid_sequences, mean, std, device)
        model = LinearChainCRF(features.shape[1], len(classes), args.dropout).to(device)
        initialize_model(model, train_tensors, classes, y[train_mask])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        class_weights = torch.tensor(
            compute_class_weight("balanced", classes=classes, y=y[train_mask]),
            dtype=torch.float32,
            device=device,
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = torch.zeros((), device=device)
            token_count = 0
            order = torch.randperm(len(train_tensors)).tolist()
            optimizer.zero_grad(set_to_none=True)
            for idx in order:
                x_seq, y_seq, _ = train_tensors[idx]
                nll = model.neg_log_likelihood(x_seq, y_seq) / len(y_seq)
                emit = model.emission_scores(x_seq)
                ce = nn.functional.cross_entropy(emit, y_seq, weight=class_weights)
                loss = nll + args.ce_weight * ce
                loss.backward()
                total = total + loss.detach() * len(y_seq)
                token_count += len(y_seq)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            if epoch in {1, 25, 50, 100, 200, 400, args.epochs}:
                print(f"fold {fold} epoch {epoch}: loss={float(total / token_count):.5f}", flush=True)
        pred, emit_proba = predict_sequences(model, valid_tensors, len(classes), device)
        for seq_pred, seq_proba, (_, _, rows) in zip(pred, emit_proba, valid_tensors, strict=True):
            oof_pred[rows] = classes[seq_pred]
            oof_emission[rows] = seq_proba
        score = float(f1_score(y[valid_mask], oof_pred[valid_mask], average="macro"))
        fold_rows.append({"fold": int(fold), "macro_f1": score, "n_valid": int(valid_mask.sum())})
        print(f"fold {fold}: macro-F1={score:.6f}", flush=True)

    covered = np.isin(folds, fold_values)
    macro = float(f1_score(y[covered], oof_pred[covered], average="macro"))
    print("\nOOF macro-F1", macro, flush=True)
    print(classification_report(y[covered], oof_pred[covered], digits=4, zero_division=0), flush=True)
    stem = f"sequence_crf_{args.feature_mode}"
    np.savez_compressed(
        args.output_dir / f"oof_{stem}.npz",
        proba=oof_emission,
        pred=oof_pred,
        classes=classes,
        label=y,
        fold=folds,
        file_id=file_ids,
        user_id=users,
        covered=covered,
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / f"{stem}_folds.csv", index=False)
    pd.DataFrame(classification_report(y[covered], oof_pred[covered], output_dict=True, zero_division=0)).T.to_csv(
        args.output_dir / f"{stem}_report.csv"
    )
    payload = {
        "feature_mode": args.feature_mode,
        "macro_f1": macro,
        "fold_scores": fold_rows,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ce_weight": args.ce_weight,
        "dropout": args.dropout,
    }
    (args.output_dir / f"{stem}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(args.output_dir / f"{stem}_metrics.csv", index=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_features(bundle: np.lib.npyio.NpzFile, mode: str) -> np.ndarray:
    blocks = [log_block(bundle["proba"].astype(float))]
    if mode == "components":
        for key in bundle.files:
            if key.endswith("_proba") and bundle[key].ndim == 2:
                blocks.append(log_block(bundle[key].astype(float)))
    file_ids = bundle["file_id"].astype(int)
    users = bundle["user_id"].astype(str)
    blocks.append(position_features(file_ids, users))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def log_block(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-6, 1.0))


def position_features(file_ids: np.ndarray, users: np.ndarray) -> np.ndarray:
    out = np.zeros((len(file_ids), 4), dtype=np.float32)
    frame = pd.DataFrame({"row": np.arange(len(file_ids)), "file_id": file_ids, "user_id": users})
    for _, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False):
        rows = group["row"].to_numpy(dtype=int)
        denom = max(1, len(rows) - 1)
        pos = np.arange(len(rows), dtype=np.float32) / denom
        out[rows, 0] = pos
        out[rows, 1] = 1.0 - pos
        out[rows, 2] = np.sin(2 * np.pi * pos)
        out[rows, 3] = np.cos(2 * np.pi * pos)
    return out


def make_sequences(
    features: np.ndarray,
    y: np.ndarray,
    file_ids: np.ndarray,
    users: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    frame = pd.DataFrame({"row": np.flatnonzero(mask), "file_id": file_ids[mask], "user_id": users[mask]})
    sequences = []
    for _, group in frame.sort_values(["user_id", "file_id"]).groupby("user_id", sort=False):
        rows = group["row"].to_numpy(dtype=int)
        sequences.append((features[rows], y[rows], rows))
    return sequences


def to_tensors(sequences, mean: np.ndarray, std: np.ndarray, device: torch.device):
    tensors = []
    for x, labels, rows in sequences:
        x_std = (x - mean) / std
        tensors.append(
            (
                torch.tensor(x_std, dtype=torch.float32, device=device),
                torch.tensor(labels, dtype=torch.long, device=device),
                rows,
            )
        )
    return tensors


def initialize_model(model: LinearChainCRF, train_tensors, classes: np.ndarray, y_train: np.ndarray) -> None:
    with torch.no_grad():
        nn.init.zeros_(model.emission.weight)
        nn.init.zeros_(model.emission.bias)
        n_classes = len(classes)
        for cls_idx in range(n_classes):
            model.emission.weight[cls_idx, cls_idx] = 0.7
        counts = torch.ones(n_classes, n_classes, device=model.transition.device) * 0.5
        starts = torch.ones(n_classes, device=model.transition.device) * 0.5
        for _, y_seq, _ in train_tensors:
            starts[y_seq[0]] += 1.0
            if len(y_seq) > 1:
                for prev, cur in zip(y_seq[:-1], y_seq[1:], strict=False):
                    counts[prev, cur] += 1.0
        trans = torch.log(counts / counts.sum(dim=1, keepdim=True))
        start = torch.log(starts / starts.sum())
        model.transition.copy_(0.10 * trans)
        model.start.copy_(0.10 * start)


@torch.no_grad()
def predict_sequences(model: LinearChainCRF, tensors, n_classes: int, device: torch.device):
    model.eval()
    preds = []
    probas = []
    for x_seq, _, _ in tensors:
        path = model.decode(x_seq).detach().cpu().numpy()
        emit = model.emission_scores(x_seq)
        proba = torch.softmax(emit, dim=1).detach().cpu().numpy().astype(np.float32)
        if proba.shape[1] != n_classes:
            raise RuntimeError("Unexpected emission dimension")
        preds.append(path)
        probas.append(proba)
    return preds, probas


if __name__ == "__main__":
    main()
