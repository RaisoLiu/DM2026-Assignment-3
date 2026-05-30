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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_aeon_rocket import make_representation


SIGNAL_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1D CNN sequence model with grouped CV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/sequence"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequence_cnn"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--representation", choices=["raw", "augmented"], default="raw")
    parser.add_argument("--sampler-mode", choices=["weighted", "shuffle"], default="weighted")
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--max-folds", type=int, default=0, help="Debug: limit number of folds; 0 means all.")
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class SequenceCNN(nn.Module):
    def __init__(self, in_channels: int, channels: int, n_classes: int, dropout: float):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, channels, 9, dropout),
            nn.MaxPool1d(2),
            ConvBlock(channels, channels * 2, 7, dropout),
            nn.MaxPool1d(2),
            ConvBlock(channels * 2, channels * 2, 5, dropout),
            nn.MaxPool1d(2),
            ConvBlock(channels * 2, channels * 3, 3, dropout),
        )
        dim = channels * 3 * 2
        self.head = nn.Sequential(
            nn.Linear(dim, channels * 2),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, n_classes),
        )

    def forward(self, x):
        x = self.blocks(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.head(pooled)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y, users, file_ids = load_or_build_cache(args.data_dir, args.cache_dir)
    x = make_representation(x.astype(np.float32), args.representation)
    classes = np.array(sorted(np.unique(y)))
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}; representation={args.representation}; x={x.shape}; "
        f"classes={classes.tolist()}; users={len(np.unique(users))}",
        flush=True,
    )
    oof = np.zeros((len(x), len(classes)), dtype=np.float32)
    fold_scores: list[float] = []
    splits = list(cv.split(x, y, users))
    if args.max_folds:
        splits = splits[: args.max_folds]
    for fold, (tr, va) in enumerate(splits, start=1):
        print(f"\n=== fold {fold} ===", flush=True)
        x_tr, x_va = x[tr].copy(), x[va].copy()
        mean = x_tr.mean(axis=(0, 2), keepdims=True)
        std = x_tr.std(axis=(0, 2), keepdims=True) + 1e-6
        x_tr = (x_tr - mean) / std
        x_va = (x_va - mean) / std
        model = SequenceCNN(x.shape[1], args.channels, len(classes), args.dropout).to(device)
        weight_tensor = None
        if args.class_weight_mode == "balanced":
            class_weights = compute_class_weight("balanced", classes=classes, y=y[tr])
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = FocalCrossEntropyLoss(weight=weight_tensor, gamma=args.focal_gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        train_loader = make_train_loader(x_tr, y[tr], args.batch_size, args.sampler_mode)
        valid_loader = DataLoader(SequenceDataset(x_va, y[va]), batch_size=args.batch_size * 2, shuffle=False, num_workers=2)
        best_score = -1.0
        best_state = None
        bad_epochs = 0
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            proba = predict_proba(model, valid_loader, device)
            pred = classes[np.argmax(proba, axis=1)]
            score = float(f1_score(y[va], pred, average="macro"))
            print(f"fold {fold} epoch {epoch:02d}: macro-F1={score:.5f}", flush=True)
            if score > best_score + 1e-5:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        model.load_state_dict(best_state)
        proba = predict_proba(model, valid_loader, device)
        oof[va] = proba
        pred = classes[np.argmax(proba, axis=1)]
        score = float(f1_score(y[va], pred, average="macro"))
        fold_scores.append(score)
        print(f"fold {fold} best macro-F1={score:.5f}", flush=True)
    covered = oof.sum(axis=1) > 0
    base_pred = classes[np.argmax(oof[covered], axis=1)]
    base_score = float(f1_score(y[covered], base_pred, average="macro"))
    print("\nOOF base macro-F1", base_score, flush=True)
    print(classification_report(y[covered], base_pred, digits=4, zero_division=0), flush=True)
    np.savez_compressed(
        args.output_dir / "oof_sequence_cnn.npz",
        proba=oof,
        label=y,
        user_id=users,
        file_id=file_ids,
        classes=classes,
        covered=covered,
    )
    pd.DataFrame(
        {
            "model": ["sequence_cnn"],
            "representation": [args.representation],
            "sampler_mode": [args.sampler_mode],
            "class_weight_mode": [args.class_weight_mode],
            "focal_gamma": [args.focal_gamma],
            "base_macro_f1": [base_score],
            "fold_scores": [",".join(f"{s:.5f}" for s in fold_scores)],
        }
    ).to_csv(args.output_dir / "cv_metrics.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"representation": args.representation, "base_macro_f1": base_score, "fold_scores": fold_scores}, indent=2),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_or_build_cache(data_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "train_sequences.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data["x"], data["y"], data["users"].astype(str), data["file_ids"]
    rows = []
    labels = []
    users = []
    file_ids = []
    for csv_path in sorted((data_dir / "train").rglob("*.csv")):
        df = pd.read_csv(csv_path)
        df = df.sort_values("index")
        arr = df[SIGNAL_COLS].interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        if arr.shape[0] != 300:
            arr = resample_to_300(arr)
        rows.append(arr.T)
        labels.append(int(df["label"].iloc[0]))
        users.append(csv_path.parent.name)
        file_ids.append(int(df["file_id"].iloc[0]))
    x = np.stack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    users_arr = np.array(users)
    file_ids_arr = np.array(file_ids)
    np.savez_compressed(path, x=x, y=y, users=users_arr, file_ids=file_ids_arr)
    return x, y, users_arr, file_ids_arr


def resample_to_300(arr: np.ndarray) -> np.ndarray:
    old = np.linspace(0, 1, len(arr))
    new = np.linspace(0, 1, 300)
    return np.vstack([np.interp(new, old, arr[:, i]) for i in range(arr.shape[1])]).T.astype(np.float32)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        if self.gamma <= 0:
            return loss.mean()
        pt = torch.exp(-loss.detach()).clamp(1e-6, 1.0)
        return (((1.0 - pt) ** self.gamma) * loss).mean()


def make_train_loader(x: np.ndarray, y: np.ndarray, batch_size: int, sampler_mode: str) -> DataLoader:
    if sampler_mode == "shuffle":
        return DataLoader(
            SequenceDataset(x, y),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
    class_counts = np.bincount(y)
    sample_weights = 1.0 / np.maximum(class_counts[y], 1)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(
        SequenceDataset(x, y),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> None:
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    main()
