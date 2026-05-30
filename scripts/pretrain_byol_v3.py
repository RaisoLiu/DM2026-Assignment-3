#!/usr/bin/env python3
"""BYOL v3: self-supervised pretraining of a deeper InceptionTime encoder.

Differences vs the existing SimCLR v1/v2:
  - Encoder: 12 blocks (vs 6), 32 filters, kernels (9,19,39,79). Deeper + multi-scale.
  - Objective: BYOL (Bootstrap Your Own Latent) — online + EMA target, predictor, cosine MSE.
                No negative pairs → batches as small as 128 work cleanly.
  - Epochs: 500 (vs 150/300).
  - Harder augmentations: channel-drop p=0.3, time-mask 20-60 steps p=0.4,
                         magnitude warp ±30%, Gaussian noise σ=0.05, cyclic shift ±20.

Input: combined train+test sequences (17,869 of shape (6, 300)).
Output: artifacts/inception_byol_v3/encoder.pt (state_dict of the 12-block encoder
        and projector/predictor for diagnostics).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


# ============================================================================
# Encoder: 12-block InceptionTime
# ============================================================================


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        bottleneck: int,
        kernel_sizes,
    ) -> None:
        super().__init__()
        self.use_bottleneck = in_channels > bottleneck and bottleneck > 0
        proj_channels = bottleneck if self.use_bottleneck else in_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
            if self.use_bottleneck
            else nn.Identity()
        )
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv1d(
                    proj_channels,
                    n_filters,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    bias=False,
                )
            )
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottlenecked = self.bottleneck(x)
        outputs = [conv(bottlenecked) for conv in self.convs]
        outputs.append(self.pool_conv(x))
        out = torch.cat(outputs, dim=1)
        return self.act(self.bn(out))


class InceptionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        n_blocks: int = 12,
        n_filters: int = 32,
        bottleneck: int = 32,
        kernel_sizes: tuple = (9, 19, 39, 79),
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


# ============================================================================
# BYOL heads (projector + predictor)
# ============================================================================


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def byol_loss(p: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Symmetric cosine-similarity loss from BYOL.
    p: predictor output on online branch
    z_target: target projector output (stop-grad)
    """
    p = F.normalize(p, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    return 2.0 - 2.0 * (p * z_target).sum(dim=-1).mean()


# ============================================================================
# Augmentations (harder than SimCLR v2)
# ============================================================================


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


def time_warp_th(x: torch.Tensor, factor: float) -> torch.Tensor:
    c, t = x.shape
    new_t = max(1, int(round(t * factor)))
    x_new = F.interpolate(x.unsqueeze(0), size=new_t, mode="linear", align_corners=False).squeeze(0)
    if new_t == t:
        return x_new
    if new_t > t:
        start = (new_t - t) // 2
        return x_new[:, start : start + t]
    pad = t - new_t
    return F.pad(x_new, (pad // 2, pad - pad // 2))


def random_crop_resample(x: torch.Tensor, rng: np.random.Generator, min_frac: float = 0.5) -> torch.Tensor:
    c, t = x.shape
    frac = float(rng.uniform(min_frac, 1.0))
    crop = max(8, int(round(t * frac)))
    start = int(rng.integers(0, t - crop + 1))
    cropped = x[:, start : start + crop]
    return F.interpolate(cropped.unsqueeze(0), size=t, mode="linear", align_corners=False).squeeze(0)


def time_mask(x: torch.Tensor, rng: np.random.Generator, p: float = 0.4) -> torch.Tensor:
    if rng.random() < p:
        c, t = x.shape
        mask_len = int(rng.integers(20, 61))
        start = int(rng.integers(0, t - mask_len + 1))
        out = x.clone()
        out[:, start : start + mask_len] = 0.0
        return out
    return x


def channel_drop(x: torch.Tensor, rng: np.random.Generator, p: float = 0.3) -> torch.Tensor:
    if rng.random() < p:
        out = x.clone()
        axis = int(rng.integers(0, 3))
        for offset in (0, 3):
            out[axis + offset] = 0.0
        return out
    return x


def channel_permute(x: torch.Tensor, rng: np.random.Generator, p: float = 0.2) -> torch.Tensor:
    if rng.random() < p:
        axes = list(range(3))
        rng.shuffle(axes)
        out = x.clone()
        for axis_target, axis_source in enumerate(axes):
            out[axis_target] = x[axis_source]
            out[axis_target + 3] = x[axis_source + 3]
        return out
    return x


def augment_view(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    out = random_crop_resample(x, rng, min_frac=0.5)
    if rng.random() < 0.7:
        out = time_warp_th(out, float(rng.uniform(0.8, 1.2)))
    if rng.random() < 0.7:
        out = out * float(rng.uniform(0.7, 1.3))  # magnitude warp ±30%
    if rng.random() < 0.7:
        out = out + torch.randn_like(out) * 0.05  # noise σ=0.05
    if rng.random() < 0.5:
        shift = int(rng.integers(-20, 21))
        if shift != 0:
            out = torch.roll(out, shifts=shift, dims=-1)
    out = time_mask(out, rng, p=0.4)
    out = channel_drop(out, rng, p=0.3)
    out = channel_permute(out, rng, p=0.2)
    return out


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


# ============================================================================
# BYOL wrapper
# ============================================================================


class BYOL(nn.Module):
    def __init__(self, encoder: InceptionEncoder, proj_dim: int = 256, pred_hidden: int = 512) -> None:
        super().__init__()
        self.online_encoder = encoder
        feat_dim = encoder.feature_dim
        self.online_projector = MLPHead(feat_dim, hidden_dim=pred_hidden, out_dim=proj_dim)
        self.online_predictor = MLPHead(proj_dim, hidden_dim=pred_hidden, out_dim=proj_dim)
        # Target = EMA copy of online encoder + projector
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)
        for online_p, target_p in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor):
        # Online branch
        h1 = self.online_encoder(v1)
        z1 = self.online_projector(h1)
        p1 = self.online_predictor(z1)
        h2 = self.online_encoder(v2)
        z2 = self.online_projector(h2)
        p2 = self.online_predictor(z2)
        # Target branch (no grad)
        with torch.no_grad():
            zt1 = self.target_projector(self.target_encoder(v1))
            zt2 = self.target_projector(self.target_encoder(v2))
        loss = 0.5 * (byol_loss(p1, zt2) + byol_loss(p2, zt1))
        return loss


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seq", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    p.add_argument("--test-seq", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--output", type=Path, default=Path("artifacts/inception_byol_v3/encoder.pt"))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.5e-6)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--n-filters", type=int, default=32)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--pred-hidden", type=int, default=512)
    p.add_argument("--ema-tau-base", type=float, default=0.996)
    p.add_argument("--ema-tau-final", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=50, help="Save intermediate checkpoint every N epochs.")
    p.add_argument("--log-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train = np.load(args.train_seq, allow_pickle=True)
    test = np.load(args.test_seq, allow_pickle=True)
    x_all = np.concatenate([train["x"], test["x"]], axis=0).astype(np.float32)
    print(f"Combined sequences: {x_all.shape}")

    dataset = TwoViewDataset(x_all, augment_seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    encoder = InceptionEncoder(
        in_channels=6,
        n_blocks=args.n_blocks,
        n_filters=args.n_filters,
        kernel_sizes=(9, 19, 39, 79),
    )
    print(f"Encoder feature dim: {encoder.feature_dim}")
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {n_params/1e6:.2f}M")

    model = BYOL(encoder, proj_dim=args.proj_dim, pred_hidden=args.pred_hidden).to(device)

    optim_params = list(model.online_encoder.parameters()) + list(model.online_projector.parameters()) + list(model.online_predictor.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        total_steps = args.epochs * len(loader)
        warmup_steps = args.warmup_epochs * len(loader)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    ctx_factory = (lambda: torch.amp.autocast(device.type, dtype=torch.bfloat16)) if use_amp else (lambda: torch.cuda.amp.autocast(enabled=False))

    best_loss = float("inf")
    global_step = 0
    total_steps = args.epochs * len(loader)
    metrics = []

    print(f"Starting BYOL pretrain: {args.epochs} epochs, batch {args.batch_size}, total steps {total_steps}")
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        loss_acc = 0.0
        n_seen = 0
        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with ctx_factory():
                loss = model(v1, v2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # EMA update with cosine-scheduled tau:
            # tau starts at ema_tau_base (e.g. 0.996, target updates faster early)
            # and increases to ema_tau_final (e.g. 1.0, target frozen at end).
            prog = global_step / max(1, total_steps)
            tau = args.ema_tau_final - (args.ema_tau_final - args.ema_tau_base) * 0.5 * (1.0 + np.cos(np.pi * prog))
            model.update_target(tau)
            global_step += 1
            loss_acc += float(loss.detach().item()) * v1.size(0)
            n_seen += v1.size(0)
        avg_loss = loss_acc / max(1, n_seen)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "encoder_state_dict": model.online_encoder.state_dict(),
                    "projector_state_dict": model.online_projector.state_dict(),
                    "predictor_state_dict": model.online_predictor.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "args": vars(args),
                },
                args.output,
            )
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{args.epochs}: loss={avg_loss:.4f} best={best_loss:.4f} tau={tau:.4f} {time.time()-t:.1f}s", flush=True)
        metrics.append({"epoch": epoch, "loss": avg_loss, "tau": float(tau), "lr": optimizer.param_groups[0]["lr"]})

        if epoch % args.save_every == 0:
            ckpt_path = args.output.parent / f"encoder_e{epoch:03d}.pt"
            torch.save({"encoder_state_dict": model.online_encoder.state_dict(), "epoch": epoch, "loss": avg_loss}, ckpt_path)
            print(f"  → checkpoint {ckpt_path}", flush=True)

    # Final save
    final_path = args.output.parent / "encoder_final.pt"
    torch.save(
        {
            "encoder_state_dict": model.online_encoder.state_dict(),
            "projector_state_dict": model.online_projector.state_dict(),
            "predictor_state_dict": model.online_predictor.state_dict(),
            "epoch": args.epochs,
            "loss": avg_loss,
            "args": vars(args),
        },
        final_path,
    )
    (args.output.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved final encoder to {final_path}")


if __name__ == "__main__":
    main()
