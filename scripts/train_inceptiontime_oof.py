#!/usr/bin/env python3
"""InceptionTime OOF training over the 52-user fold scheme.

Saves OOF probabilities in npz format compatible with
``scripts/evaluate_sequence_smoothing.py`` (keys: proba, classes, label,
file_id, user_id). Optional auxiliary binary head for class-2 vs rest.

Usage:
    .venv/bin/python scripts/train_inceptiontime_oof.py \
        --fold-file artifacts/folds/sgkf_seed2026_train52.csv \
        --output-dir artifacts/inception_oof/seed2026 \
        --epochs 25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SIGNAL_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
N_CLASSES = 6


# -------------------- Data --------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HAR sequence model per fold and emit OOF.")
    parser.add_argument(
        "--model",
        choices=["inception", "patchtst", "tcn", "resnet1d"],
        default="inception",
        help="Which architecture to train.",
    )
    parser.add_argument(
        "--pseudo-labels",
        type=Path,
        default=None,
        help="Optional consensus pseudo-label CSV (Id,Label) for test rows; merged into training with --pseudo-weight.",
    )
    parser.add_argument(
        "--pseudo-weight",
        type=float,
        default=0.4,
        help="Sample-weight for pseudo-labeled rows (real labels get 1.0).",
    )
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
        help="CSV with file_id,user_id,label,fold (52-user partition).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--seq-cache",
        type=Path,
        default=Path("artifacts/sequence/train_sequences.npz"),
    )
    parser.add_argument(
        "--test-seq-cache",
        type=Path,
        default=Path("artifacts/sequence/test_sequences.npz"),
        help="Cache for test sequences; rebuilt if missing for per-user z-score stats.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/inception_oof/seed2026"),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", type=str, default="all", help="all or comma-separated fold ids")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--n-filters", type=int, default=32)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--kernel-sizes", type=str, default="9,19,39")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument(
        "--sampler",
        choices=["shuffle", "soft", "balanced"],
        default="soft",
        help="shuffle: natural; soft: counts^-0.5; balanced: 1/counts",
    )
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Path to a SimCLR pretraining checkpoint (encoder_state_dict). Backbone weights loaded with strict=False.",
    )
    parser.add_argument(
        "--per-user-zscore",
        action="store_true",
        default=True,
        help="Apply per-user z-score using train+test stats. On by default.",
    )
    parser.add_argument(
        "--no-per-user-zscore",
        dest="per_user_zscore",
        action="store_false",
    )
    return parser.parse_args()


def build_test_sequence_cache(data_dir: Path, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["x"].astype(np.float32), d["users"].astype(str), d["file_ids"].astype(int)
    rows, users, file_ids = [], [], []
    for csv_path in sorted((data_dir / "test").rglob("*.csv")):
        df = pd.read_csv(csv_path).sort_values("index")
        arr = (
            df[SIGNAL_COLS]
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        if arr.shape[0] != 300:
            old = np.linspace(0, 1, len(arr))
            new = np.linspace(0, 1, 300)
            arr = np.vstack([np.interp(new, old, arr[:, i]) for i in range(arr.shape[1])]).T.astype(np.float32)
        rows.append(arr.T)
        users.append(csv_path.parent.name)
        file_ids.append(int(df["file_id"].iloc[0]) if "file_id" in df.columns else int(csv_path.stem))
    x = np.stack(rows).astype(np.float32)
    np.savez_compressed(cache, x=x, users=np.array(users), file_ids=np.array(file_ids, dtype=int))
    return x, np.array(users), np.array(file_ids, dtype=int)


def compute_per_user_zscore(
    x_train: np.ndarray,
    users_train: np.ndarray,
    x_test: np.ndarray | None,
    users_test: np.ndarray | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns user_id -> (mean[6], std[6]) computed over train+test windows."""
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pieces: list[tuple[np.ndarray, np.ndarray]] = [(x_train, users_train)]
    if x_test is not None:
        pieces.append((x_test, users_test))
    by_user: dict[str, list[np.ndarray]] = {}
    for x, u in pieces:
        for i, user in enumerate(u):
            by_user.setdefault(str(user), []).append(x[i])
    for user, arrays in by_user.items():
        stack = np.stack(arrays, axis=0)  # (n, 6, 300)
        flat = stack.transpose(1, 0, 2).reshape(stack.shape[1], -1)  # (6, n*300)
        mean = flat.mean(axis=1).astype(np.float32)
        std = flat.std(axis=1).astype(np.float32) + 1e-6
        stats[user] = (mean, std)
    return stats


def apply_zscore(x: np.ndarray, users: np.ndarray, stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = x.copy()
    for i, user in enumerate(users):
        m, s = stats[str(user)]
        out[i] = (x[i] - m[:, None]) / s[:, None]
    return out


# -------------------- Augmentation --------------------


def time_warp(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Time-warp a (C, T) tensor by factor (0.9..1.1) via linear interp, returning (C, T)."""
    c, t = x.shape
    new_t = max(1, int(round(t * factor)))
    x_new = F.interpolate(x.unsqueeze(0), size=new_t, mode="linear", align_corners=False).squeeze(0)
    if new_t == t:
        return x_new
    if new_t > t:
        start = (new_t - t) // 2
        return x_new[:, start : start + t]
    pad = t - new_t
    left = pad // 2
    return F.pad(x_new, (left, pad - left), mode="replicate")


def augment_window(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Apply time-warp, magnitude scale, jitter, and cyclic shift with p=0.7 each."""
    if rng.random() < 0.7:
        factor = float(rng.uniform(0.9, 1.1))
        x = time_warp(x, factor)
    if rng.random() < 0.7:
        scale = float(rng.uniform(0.9, 1.1))
        x = x * scale
    if rng.random() < 0.7:
        x = x + torch.randn_like(x) * 0.01
    if rng.random() < 0.7:
        shift = int(rng.integers(-5, 6))
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
    return x


class WindowDataset(Dataset):
    """Returns (x, y, user_idx, weight) for class-balanced sampling, intra-user mixup,
    and optional per-sample weighting (used for pseudo-labels at lower weight than real labels)."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        users: np.ndarray,
        train: bool,
        augment_seed: int,
        intra_user_mixup_alpha: float = 0.0,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.user_idx = pd.Categorical(users).codes.astype(np.int64)
        self.train = train
        self.rng = np.random.default_rng(augment_seed)
        self.mixup_alpha = float(intra_user_mixup_alpha)
        if sample_weights is None:
            self.sample_weights = torch.ones(len(y), dtype=torch.float32)
        else:
            self.sample_weights = torch.from_numpy(sample_weights.astype(np.float32))
        self.user_index_map: dict[int, list[int]] = {}
        if train and self.mixup_alpha > 0:
            for i, code in enumerate(self.user_idx):
                self.user_index_map.setdefault(int(code), []).append(int(i))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = int(self.y[idx])
        if self.train:
            x = augment_window(x.clone(), self.rng)
            # Intra-user mixup on classes 0/1 only
            if (
                self.mixup_alpha > 0
                and y in (0, 1)
                and self.rng.random() < 0.5
            ):
                code = int(self.user_idx[idx])
                pool = [j for j in self.user_index_map.get(code, []) if int(self.y[j]) in (0, 1)]
                if len(pool) > 1:
                    j = int(self.rng.choice([k for k in pool if k != idx]))
                    lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    x = lam * x + (1 - lam) * augment_window(self.x[j].clone(), self.rng)
        return x, y, int(self.user_idx[idx]), float(self.sample_weights[idx])


# -------------------- Model --------------------


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        bottleneck: int,
        kernel_sizes: Iterable[int],
        use_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
            conv_in = bottleneck
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels
        self.convs = nn.ModuleList(
            [nn.Conv1d(conv_in, n_filters, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes]
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        out_channels = n_filters * (len(self.convs) + 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        outs = [conv(z) for conv in self.convs]
        outs.append(self.maxpool_conv(self.maxpool(x)))
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(out))


class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        n_blocks: int = 6,
        n_filters: int = 32,
        bottleneck: int = 32,
        kernel_sizes: Iterable[int] = (9, 19, 39),
        n_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        block_out = n_filters * (len(self.kernel_sizes) + 1)
        prev_channels = in_channels
        residual_in = in_channels
        for i in range(n_blocks):
            self.blocks.append(
                InceptionBlock(prev_channels, n_filters, bottleneck, self.kernel_sizes)
            )
            prev_channels = block_out
            if (i + 1) % 3 == 0:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(residual_in, prev_channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(prev_channels),
                    )
                )
                residual_in = prev_channels
            else:
                self.shortcuts.append(None)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_share = nn.Sequential(
            nn.Linear(prev_channels, prev_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_main = nn.Linear(prev_channels, n_classes)
        self.head_aux = nn.Linear(prev_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = block(x)
            if shortcut is not None:
                x = x + shortcut(residual)
                residual = x
        z = self.gap(x).squeeze(-1)  # (B, C)
        h = self.head_share(z)
        return self.head_main(h), self.head_aux(h).squeeze(-1)


# -------------------- Loss --------------------


class FocalCE(nn.Module):
    def __init__(self, gamma: float = 0.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        if self.gamma > 0:
            with torch.no_grad():
                pt = torch.exp(-loss).clamp(1e-6, 1.0)
            loss = ((1.0 - pt) ** self.gamma) * loss
        if sample_weight is None:
            return loss.mean()
        # Weighted mean
        return (loss * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)


def class_balanced_weights(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Soft class-balanced weights based on sqrt of effective number.
    Floored at 0.7 to avoid starving the majority classes when paired with a
    rebalancing sampler. Designed to give a *gentle* nudge toward minorities,
    not the aggressive 1/count weighting that destroys class 1 learning.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / np.sqrt(np.maximum(eff, 1.0))
    weights = weights * (N_CLASSES / weights.sum())
    weights = np.maximum(weights, 0.7)
    return weights.astype(np.float32)


# -------------------- Training loop --------------------


def make_train_loader(
    x: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    batch_size: int,
    seed: int,
    mixup_alpha: float,
    num_workers: int,
    sampler_mode: str = "soft",
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    """Sampler modes:
    - "shuffle": natural class distribution (preserves majority signal)
    - "soft": counts^-0.5 weights (mild rebalance, doesn't starve majorities)
    - "balanced": 1/counts (strong rebalance, can starve majorities)
    sample_weights: per-row loss weights (e.g., 1.0 for real, 0.4 for pseudo).
    """
    dataset = WindowDataset(
        x, y, users, train=True, augment_seed=seed,
        intra_user_mixup_alpha=mixup_alpha,
        sample_weights=sample_weights,
    )
    if sampler_mode == "shuffle":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    if sampler_mode == "balanced":
        sample_weight = 1.0 / np.maximum(counts[y], 1.0)
    else:  # "soft"
        sample_weight = 1.0 / np.sqrt(np.maximum(counts[y], 1.0))
    sampler = WeightedRandomSampler(
        sample_weight, num_samples=len(sample_weight), replacement=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def make_eval_loader(x: np.ndarray, y: np.ndarray, users: np.ndarray, batch_size: int, num_workers: int) -> DataLoader:
    dataset = WindowDataset(x, y, users, train=False, augment_seed=0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def train_fold(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    fold: int,
    sample_weights_for_train: np.ndarray | None = None,
    extra_train_x: np.ndarray | None = None,
    extra_train_y: np.ndarray | None = None,
    extra_train_users: np.ndarray | None = None,
    extra_train_weights: np.ndarray | None = None,
) -> np.ndarray:
    device = torch.device(args.device)
    kernels = tuple(int(k) for k in args.kernel_sizes.split(","))

    train_x = x[train_idx]
    train_y = y[train_idx]
    train_u = users[train_idx]
    valid_x = x[valid_idx]
    valid_y = y[valid_idx]
    valid_u = users[valid_idx]

    # Build per-sample weights (1.0 for real, pseudo_weight for pseudo)
    train_sample_weights = np.ones(len(train_y), dtype=np.float32)
    if extra_train_x is not None and len(extra_train_x) > 0:
        train_x = np.concatenate([train_x, extra_train_x], axis=0)
        train_y = np.concatenate([train_y, extra_train_y], axis=0)
        train_u = np.concatenate([train_u, extra_train_users], axis=0)
        ew = extra_train_weights if extra_train_weights is not None else np.full(len(extra_train_x), 0.4, dtype=np.float32)
        train_sample_weights = np.concatenate([train_sample_weights, ew], axis=0)
        print(
            f"  Fold {fold}: train + pseudo = {len(train_y)} rows "
            f"(real={len(train_y) - len(extra_train_x)} weight=1.0, pseudo={len(extra_train_x)} avg-weight={ew.mean():.2f})",
            flush=True,
        )

    cls_w_np = class_balanced_weights(train_y, beta=0.999)
    # Cap rare-class boost at 2.0 (per the strategy spec)
    cls_w_np = np.clip(cls_w_np, a_min=None, a_max=2.0).astype(np.float32)
    print(f"  Fold {fold}: class weights {np.round(cls_w_np, 3).tolist()}", flush=True)

    train_loader = make_train_loader(
        train_x, train_y, train_u, args.batch_size, args.seed + fold,
        args.mixup_alpha, args.num_workers,
        sampler_mode=args.sampler,
        sample_weights=train_sample_weights,
    )
    valid_loader = make_eval_loader(valid_x, valid_y, valid_u, args.batch_size, args.num_workers)

    if args.model == "inception":
        model = InceptionTime(
            in_channels=6,
            n_blocks=args.n_blocks,
            n_filters=args.n_filters,
            bottleneck=args.bottleneck,
            kernel_sizes=kernels,
            n_classes=N_CLASSES,
            dropout=args.dropout,
        ).to(device)
    else:
        # Defer import to avoid circular import (har_models imports InceptionTime from this file)
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from har_models import build_model
        model = build_model(args.model, dropout=args.dropout).to(device)
    if args.init_from_checkpoint and args.init_from_checkpoint.exists():
        ckpt = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        encoder_state = ckpt.get("encoder_state_dict", ckpt)
        # Filter to keys present in the InceptionTime model and matching shapes
        own_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        for k, v in encoder_state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
        print(
            f"  Fold {fold}: SSL init loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} from {args.init_from_checkpoint}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cls_w = torch.from_numpy(cls_w_np).to(device)
    main_loss = FocalCE(gamma=args.focal_gamma, weight=cls_w)
    aux_loss = nn.BCEWithLogitsLoss()
    use_amp = (not args.no_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_amp:
        print("  bf16 AMP enabled", flush=True)

    best_macro = -1.0
    best_proba = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        t_epoch = time.time()
        for xb, yb, _, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            target_aux = (yb == 2).float()
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else _NullCtx()
            with ctx:
                logits, aux_logits = model(xb)
                main_loss_val = main_loss(logits, yb, sample_weight=wb)
                aux_loss_raw = F.binary_cross_entropy_with_logits(aux_logits, target_aux, reduction="none")
                aux_loss_val = (aux_loss_raw * wb).sum() / wb.sum().clamp(min=1e-6)
                loss = main_loss_val + args.aux_weight * aux_loss_val
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * xb.size(0)
            n_seen += xb.size(0)
        scheduler.step()
        avg_loss = epoch_loss / max(1, n_seen)

        # Eval
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _, _, _ in valid_loader:
                xb = xb.to(device, non_blocking=True)
                ctx = torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else _NullCtx()
                with ctx:
                    logits, _ = model(xb)
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        pred = probs.argmax(axis=1)
        macro = f1_score(valid_y, pred, average="macro")
        per_class = f1_score(valid_y, pred, average=None, labels=list(range(N_CLASSES)))
        elapsed = time.time() - t_epoch
        print(
            f"  Fold {fold} epoch {epoch:02d}: loss={avg_loss:.4f} macroF1={macro:.4f} "
            f"perClass=[{', '.join(f'{p:.2f}' for p in per_class)}] {elapsed:.1f}s",
            flush=True,
        )
        if macro > best_macro:
            best_macro = float(macro)
            best_proba = probs

    assert best_proba is not None
    print(f"  Fold {fold}: best macroF1={best_macro:.4f}")
    return best_proba


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *_a):
        return False


# -------------------- Main --------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    folds_df = pd.read_csv(args.fold_file)
    folds_df["user_id"] = folds_df["user_id"].astype(str)

    seq = np.load(args.seq_cache, allow_pickle=True)
    x_full = seq["x"].astype(np.float32)
    y_full = seq["y"].astype(np.int64)
    users_full = seq["users"].astype(str)
    file_ids_full = seq["file_ids"].astype(int)

    # Filter to rows in fold_file
    keep_mask = np.isin(file_ids_full, folds_df["file_id"].to_numpy())
    x = x_full[keep_mask]
    y = y_full[keep_mask]
    users = users_full[keep_mask]
    file_ids = file_ids_full[keep_mask]

    # Reorder to match fold_file order
    file_to_idx = {int(fid): i for i, fid in enumerate(file_ids)}
    target_order = folds_df["file_id"].astype(int).to_numpy()
    order = np.array([file_to_idx[int(f)] for f in target_order], dtype=np.int64)
    x = x[order]
    y = y[order]
    users = users[order]
    file_ids = file_ids[order]
    fold_ids = folds_df["fold"].astype(int).to_numpy()

    print(
        f"Loaded {len(x)} train rows ({len(set(users))} users) from {args.seq_cache}",
        flush=True,
    )

    # Optional per-user z-score using train+test stats
    if args.per_user_zscore:
        if args.test_seq_cache.exists() or True:
            x_test, users_test, _ = build_test_sequence_cache(args.data_dir, args.test_seq_cache)
            print(
                f"Loaded {len(x_test)} test rows ({len(set(users_test))} users) from {args.test_seq_cache}",
                flush=True,
            )
            stats = compute_per_user_zscore(x, users, x_test, users_test)
        else:
            stats = compute_per_user_zscore(x, users, None, None)
        x = apply_zscore(x, users, stats)
        print(f"Applied per-user z-score over {len(stats)} users", flush=True)

    # Determine fold subset
    all_folds = sorted(set(fold_ids.tolist()))
    if args.folds == "all":
        run_folds = all_folds
    else:
        run_folds = [int(f) for f in args.folds.split(",") if f.strip()]

    # Load pseudo-labels if provided
    extra_train_x = extra_train_y = extra_train_users = extra_train_weights = None
    if args.pseudo_labels is not None and args.pseudo_labels.exists():
        pseudo_df = pd.read_csv(args.pseudo_labels)
        pseudo_ids = pseudo_df["Id"].astype(int).to_numpy()
        pseudo_labels = pseudo_df["Label"].astype(int).to_numpy()
        # confidence_score column optional; defaults to 1.0
        if "confidence_score" in pseudo_df.columns:
            pseudo_conf = pseudo_df["confidence_score"].astype(float).to_numpy()
        else:
            pseudo_conf = np.ones(len(pseudo_df), dtype=float)
        # Match pseudo IDs to test sequences
        test_file_ids = file_ids_test if False else None
        # We already loaded x_test, users_test, test_file_ids in the z-score block; reload here for clarity
        x_test_local, users_test_local, file_ids_test_local = build_test_sequence_cache(args.data_dir, args.test_seq_cache)
        # Apply z-score to test (if per_user_zscore)
        if args.per_user_zscore:
            x_test_local = apply_zscore(x_test_local, users_test_local, stats)
        test_idx = {int(fid): i for i, fid in enumerate(file_ids_test_local)}
        sel_idx = np.array([test_idx[int(i)] for i in pseudo_ids if int(i) in test_idx], dtype=int)
        sel_pseudo_ids = np.array([int(i) for i in pseudo_ids if int(i) in test_idx], dtype=int)
        extra_train_x = x_test_local[sel_idx].astype(np.float32)
        # Map pseudo_labels by id to preserve order
        id_to_label = dict(zip(pseudo_ids.tolist(), pseudo_labels.tolist()))
        id_to_conf = dict(zip(pseudo_ids.tolist(), pseudo_conf.tolist()))
        extra_train_y = np.array([id_to_label[int(i)] for i in sel_pseudo_ids], dtype=np.int64)
        extra_train_users = users_test_local[sel_idx].astype(str)
        # Sample weight = args.pseudo_weight * confidence_score
        extra_train_weights = (np.array([id_to_conf[int(i)] for i in sel_pseudo_ids], dtype=np.float32) * args.pseudo_weight).astype(np.float32)
        print(
            f"Loaded {len(extra_train_x)} pseudo-labels from {args.pseudo_labels} (pseudo_weight={args.pseudo_weight})",
            flush=True,
        )

    proba_oof = np.full((len(x), N_CLASSES), np.nan, dtype=np.float32)
    fold_metrics: dict[int, dict[str, float]] = {}
    for fold in run_folds:
        print(f"\n=== Fold {fold} ===", flush=True)
        valid_mask = fold_ids == fold
        train_mask = ~valid_mask
        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(valid_mask)[0]
        proba = train_fold(
            args, x, y, users, train_idx, valid_idx, fold,
            extra_train_x=extra_train_x,
            extra_train_y=extra_train_y,
            extra_train_users=extra_train_users,
            extra_train_weights=extra_train_weights,
        )
        proba_oof[valid_idx] = proba
        pred = proba.argmax(axis=1)
        macro = f1_score(y[valid_idx], pred, average="macro")
        per_class = f1_score(y[valid_idx], pred, average=None, labels=list(range(N_CLASSES)))
        fold_metrics[int(fold)] = {
            "macro_f1": float(macro),
            "per_class": [float(v) for v in per_class.tolist()],
        }

    # Compute aggregate over the rows we actually predicted (others stay NaN)
    predicted_mask = ~np.isnan(proba_oof[:, 0])
    if predicted_mask.sum() == len(x):
        oof_pred = proba_oof.argmax(axis=1)
        macro_oof = float(f1_score(y, oof_pred, average="macro"))
        print(f"\nFull OOF macroF1 (base, no Viterbi): {macro_oof:.6f}", flush=True)
        print(classification_report(y, oof_pred, digits=4))
    else:
        macro_oof = float("nan")
        print(
            f"\nPartial OOF over {predicted_mask.sum()} / {len(x)} rows; full macro-F1 not reported.",
            flush=True,
        )

    out_npz = args.output_dir / f"oof_inception_seed{args.seed}.npz"
    np.savez_compressed(
        out_npz,
        proba=proba_oof.astype(np.float32),
        classes=np.arange(N_CLASSES, dtype=np.int64),
        label=y.astype(np.int64),
        file_id=file_ids.astype(np.int64),
        user_id=users.astype(str),
    )
    print(f"Wrote {out_npz}", flush=True)

    summary = {
        "seed": args.seed,
        "fold_file": str(args.fold_file),
        "n_train": int(len(x)),
        "folds_run": run_folds,
        "fold_metrics": fold_metrics,
        "oof_macro_f1_base": macro_oof,
        "smoke_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics.values()])) if fold_metrics else None,
        "epochs": args.epochs,
        "args": {
            "n_blocks": args.n_blocks,
            "n_filters": args.n_filters,
            "bottleneck": args.bottleneck,
            "kernel_sizes": args.kernel_sizes,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "focal_gamma": args.focal_gamma,
            "aux_weight": args.aux_weight,
            "mixup_alpha": args.mixup_alpha,
            "per_user_zscore": args.per_user_zscore,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
