#!/usr/bin/env python3
"""SimCLR contrastive pretraining of an InceptionTime encoder on combined train+test
sequences (17,869 total) for the DM2026-Asg-3 HAR task.

Trains the encoder with NT-Xent loss using two augmented views per sample.
Saves the encoder state_dict (excluding any classification head).

Usage:
    .venv/bin/python scripts/pretrain_simclr.py \
        --output artifacts/inception_ssl/encoder.pt \
        --epochs 150 --batch-size 256 --lr 5e-4

Then in the fine-tuning script, pass --init-from-checkpoint artifacts/inception_ssl/encoder.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_inceptiontime_oof import (
    InceptionBlock,
    apply_zscore,
    build_test_sequence_cache,
    compute_per_user_zscore,
)


# -------------------- Encoder (no classification head) --------------------


class InceptionEncoder(nn.Module):
    """Same backbone as InceptionTime in train_inceptiontime_oof.py but without heads."""

    def __init__(
        self,
        in_channels: int = 6,
        n_blocks: int = 6,
        n_filters: int = 32,
        bottleneck: int = 32,
        kernel_sizes: tuple = (9, 19, 39),
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
        self.feature_dim = prev_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = block(x)
            if shortcut is not None:
                x = x + shortcut(residual)
                residual = x
        return self.gap(x).squeeze(-1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------- Augmentations (returns torch tensor) --------------------


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


def time_warp_th(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Warp (C, T) by factor in (0.8..1.2)."""
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


def random_crop_resample(x: torch.Tensor, rng: np.random.Generator, min_frac: float = 0.7) -> torch.Tensor:
    c, t = x.shape
    crop = max(64, int(rng.uniform(min_frac, 1.0) * t))
    start = int(rng.integers(0, t - crop + 1))
    cropped = x[:, start : start + crop]
    return F.interpolate(cropped.unsqueeze(0), size=t, mode="linear", align_corners=False).squeeze(0)


def channel_mask(x: torch.Tensor, rng: np.random.Generator, p: float = 0.3) -> torch.Tensor:
    if rng.random() < p:
        out = x.clone()
        n_channels = x.shape[0]
        # Mask 1 channel pair (mean and std for the same axis)
        axis = int(rng.integers(0, 3))
        for offset in (0, 3):  # mean_x is index 0, std_x is index 3, etc.
            out[axis + offset] = 0.0
        return out
    return x


def channel_permute(x: torch.Tensor, rng: np.random.Generator, p: float = 0.3) -> torch.Tensor:
    if rng.random() < p:
        # Permute the (x, y, z) axes within the mean group and within the std group
        axes = list(range(3))
        rng.shuffle(axes)
        out = x.clone()
        for axis_target, axis_source in enumerate(axes):
            out[axis_target] = x[axis_source]
            out[axis_target + 3] = x[axis_source + 3]
        return out
    return x


def augment_view(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    out = x
    # Random crop+resample (always)
    out = random_crop_resample(out, rng, min_frac=0.6)
    # Time warp
    if rng.random() < 0.7:
        out = time_warp_th(out, float(rng.uniform(0.85, 1.15)))
    # Magnitude scale
    if rng.random() < 0.7:
        out = out * float(rng.uniform(0.85, 1.15))
    # Jitter
    if rng.random() < 0.7:
        out = out + torch.randn_like(out) * 0.02
    # Cyclic shift
    if rng.random() < 0.5:
        shift = int(rng.integers(-10, 11))
        if shift != 0:
            out = torch.roll(out, shifts=shift, dims=-1)
    out = channel_mask(out, rng, p=0.2)
    out = channel_permute(out, rng, p=0.2)
    return out


# -------------------- Dataset --------------------


class TwoViewDataset(Dataset):
    def __init__(self, x: np.ndarray, augment_seed: int) -> None:
        self.x = _to_tensor(x)
        self.rng = np.random.default_rng(augment_seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        v1 = augment_view(x.clone(), self.rng)
        v2 = augment_view(x.clone(), self.rng)
        return v1, v2


# -------------------- NT-Xent loss --------------------


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric NT-Xent contrastive loss, treating the diagonal block as positives."""
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2n, d)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / temperature  # (2n, 2n)
    mask_self = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask_self, -1e9)
    # Positive pairs: i with i+n, and i+n with i
    pos_idx = torch.arange(2 * n, device=z.device)
    pos_idx = (pos_idx + n) % (2 * n)
    return F.cross_entropy(sim, pos_idx)


# -------------------- Main --------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR pretraining of InceptionTime encoder.")
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
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/inception_ssl/encoder.pt"),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--n-filters", type=int, default=32)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--kernel-sizes", type=str, default="9,19,39")
    parser.add_argument("--projection-hidden", type=int, default=128)
    parser.add_argument("--projection-out", type=int, default=64)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    kernels = tuple(int(k) for k in args.kernel_sizes.split(","))

    # Load train + test sequences
    seq_train = np.load(args.seq_cache, allow_pickle=True)
    x_train = seq_train["x"].astype(np.float32)
    users_train = seq_train["users"].astype(str)

    x_test, users_test, _ = build_test_sequence_cache(args.data_dir, args.test_seq_cache)

    print(f"Train: {len(x_train)} from {len(set(users_train))} users")
    print(f"Test: {len(x_test)} from {len(set(users_test))} users")

    # Per-user z-score using combined stats
    stats = compute_per_user_zscore(x_train, users_train, x_test, users_test)
    x_train_z = apply_zscore(x_train, users_train, stats)
    x_test_z = apply_zscore(x_test, users_test, stats)

    # Combine
    x_all = np.concatenate([x_train_z, x_test_z], axis=0)
    print(f"Combined: {len(x_all)} sequences for SSL")

    dataset = TwoViewDataset(x_all, augment_seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    encoder = InceptionEncoder(
        in_channels=6,
        n_blocks=args.n_blocks,
        n_filters=args.n_filters,
        bottleneck=args.bottleneck,
        kernel_sizes=kernels,
    ).to(device)
    projector = ProjectionHead(
        in_dim=encoder.feature_dim,
        hidden_dim=args.projection_hidden,
        out_dim=args.projection_out,
    ).to(device)

    print(
        f"Encoder feature dim: {encoder.feature_dim}, projection out: {args.projection_out}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (not args.no_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_amp:
        print("bf16 AMP enabled", flush=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projector.train()
        t = time.time()
        loss_acc = 0.0
        n_seen = 0
        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else _NullCtx()
            with ctx:
                z1 = projector(encoder(v1))
                z2 = projector(encoder(v2))
                loss = nt_xent(z1.float(), z2.float(), temperature=args.temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), 5.0)
            optimizer.step()
            loss_acc += float(loss.detach().item()) * v1.size(0)
            n_seen += v1.size(0)
        scheduler.step()
        avg_loss = loss_acc / max(1, n_seen)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "projector_state_dict": projector.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "args": vars(args),
                    "feature_dim": encoder.feature_dim,
                },
                args.output,
            )
        print(
            f"Epoch {epoch:03d}: loss={avg_loss:.4f} best={best_loss:.4f} {time.time() - t:.1f}s",
            flush=True,
        )
        if epoch % args.checkpoint_every == 0:
            ckpt = args.output.parent / f"encoder_ep{epoch:03d}.pt"
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "projector_state_dict": projector.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "args": vars(args),
                    "feature_dim": encoder.feature_dim,
                },
                ckpt,
            )

    summary = {
        "epochs": args.epochs,
        "best_loss": best_loss,
        "n_combined": int(len(x_all)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.output.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} (best loss {best_loss:.4f})")


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *_a):
        return False


if __name__ == "__main__":
    main()
