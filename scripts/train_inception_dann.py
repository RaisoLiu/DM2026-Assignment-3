#!/usr/bin/env python3
"""H2: Domain-Adversarial fine-tune of SSL InceptionTime (DANN).

Trains InceptionTime + user-classification head with gradient reversal,
forcing the encoder to learn user-invariant features. This is the most
theoretically targeted attack on the train/test user distribution shift —
the dominant unmodeled effect in our pipeline.

Outputs: OOF probas per fold/seed; test probas from full-trained model.
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
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_inceptiontime_oof import (
    InceptionTime,
    N_CLASSES,
    apply_zscore,
    build_test_sequence_cache,
    compute_per_user_zscore,
)


# ============== Gradient Reversal Layer ==============


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, alpha)


# ============== DANN model ==============


class InceptionDANN(nn.Module):
    """InceptionTime encoder + class head + user head (with GRL)."""

    def __init__(self, base_inception: InceptionTime, n_users: int, dropout: float = 0.3):
        super().__init__()
        self.blocks = base_inception.blocks
        self.shortcuts = base_inception.shortcuts
        self.gap = base_inception.gap
        # Use blocks' output channel count
        feat_dim = base_inception.head_main.in_features  # 128
        self.head_share = base_inception.head_share
        self.head_class = nn.Linear(feat_dim, N_CLASSES)
        # Copy weights from existing class head
        with torch.no_grad():
            self.head_class.weight.copy_(base_inception.head_main.weight)
            self.head_class.bias.copy_(base_inception.head_main.bias)
        # User head (predict which user) for adversarial training
        self.head_user = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_users),
        )

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = block(x)
            if shortcut is not None:
                x = x + shortcut(residual)
                residual = x
        return self.gap(x).squeeze(-1)  # (B, feat_dim)

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_forward(x)
        h = self.head_share(z)
        class_logits = self.head_class(h)
        # Adversarial: reverse gradients then predict user
        h_rev = grad_reverse(z, alpha)
        user_logits = self.head_user(h_rev)
        return class_logits, user_logits


# ============== Dataset ==============


class SeqDataset(Dataset):
    def __init__(self, x, y, user_idx):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.user_idx = torch.from_numpy(user_idx.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx]), int(self.user_idx[idx])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-checkpoint", type=Path, default=Path("artifacts/inception_ssl/encoder.pt"))
    p.add_argument("--mode", choices=["oof", "full"], default="full")
    p.add_argument("--fold-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026_train52.csv"))
    p.add_argument("--seq-cache", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    p.add_argument("--test-seq-cache", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/dann_full"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--alpha-max", type=float, default=0.5, help="Max DANN reversal strength")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all training sequences
    seq = np.load(args.seq_cache, allow_pickle=True)
    x_all = seq["x"].astype(np.float32)
    y_all = seq["y"].astype(np.int64)
    users_all = seq["users"].astype(str)
    file_ids_all = seq["file_ids"].astype(int)

    # Load test
    x_test, users_test, file_ids_test = build_test_sequence_cache(args.data_dir, args.test_seq_cache)
    print(f"Train: {len(x_all)} ({len(set(users_all))} users); Test: {len(x_test)} ({len(set(users_test))} users)")

    # Per-user z-score using train+test stats
    stats = compute_per_user_zscore(x_all, users_all, x_test, users_test)
    x_all = apply_zscore(x_all, users_all, stats)
    x_test = apply_zscore(x_test, users_test, stats)

    # User-to-idx map (TRAIN users only — DANN tries to make train features non-discriminative)
    unique_train_users = sorted(set(users_all))
    user_to_idx = {u: i for i, u in enumerate(unique_train_users)}
    user_idx_all = np.array([user_to_idx[u] for u in users_all], dtype=np.int64)
    n_users = len(unique_train_users)
    print(f"Train user count: {n_users}")

    # Build base InceptionTime (random init for heads; SSL init for backbone)
    base = InceptionTime(in_channels=6, n_blocks=6, n_filters=32, bottleneck=32, kernel_sizes=(9, 19, 39))
    if args.ssl_checkpoint.exists():
        ckpt = torch.load(args.ssl_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt)
        own_state = base.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
        print(f"SSL init loaded {loaded} keys from {args.ssl_checkpoint}")

    model = InceptionDANN(base, n_users=n_users).to(device)

    # Train on full data (for test predictions)
    if args.mode == "full":
        train_x = x_all
        train_y = y_all
        train_user_idx = user_idx_all
        train_loader = DataLoader(
            SeqDataset(train_x, train_y, train_user_idx),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        class_loss = nn.CrossEntropyLoss()
        user_loss = nn.CrossEntropyLoss()
        use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        for epoch in range(1, args.epochs + 1):
            t_e = time.time()
            model.train()
            # Schedule alpha: 0 → alpha_max over training
            p = epoch / args.epochs
            alpha = float(args.alpha_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0))
            cl_acc, ul_acc, n_seen = 0.0, 0.0, 0
            ctx_factory = (lambda: torch.amp.autocast(device.type, dtype=torch.bfloat16)) if use_amp else (lambda: torch.cuda.amp.autocast(enabled=False))
            for xb, yb, ub in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                ub = ub.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with ctx_factory():
                    class_logits, user_logits = model(xb, alpha=alpha)
                    cl = class_loss(class_logits, yb)
                    ul = user_loss(user_logits, ub)
                    loss = cl + ul  # Note: user_logits has GRL upstream, so user_loss flows reversed
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                cl_acc += float(cl.detach().item()) * xb.size(0)
                ul_acc += float(ul.detach().item()) * xb.size(0)
                n_seen += xb.size(0)
            scheduler.step()
            print(f"  epoch {epoch:02d}: alpha={alpha:.3f} cl={cl_acc/max(1,n_seen):.4f} ul={ul_acc/max(1,n_seen):.4f} {time.time()-t_e:.1f}s")
        # Predict test
        test_user_idx = np.zeros(len(x_test), dtype=np.int64)  # dummy
        test_loader = DataLoader(
            SeqDataset(x_test, np.zeros(len(x_test), dtype=np.int64), test_user_idx),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                with ctx_factory():
                    class_logits, _ = model(xb, alpha=0.0)
                probs.append(torch.softmax(class_logits.float(), dim=1).cpu().numpy())
        test_proba = np.concatenate(probs, axis=0)
        out = args.output_dir / f"test_proba_seed{args.seed}.npz"
        np.savez_compressed(
            out,
            proba=test_proba.astype(np.float32),
            classes=np.arange(N_CLASSES, dtype=np.int64),
            file_id=file_ids_test.astype(np.int64),
            user_id=users_test.astype(str),
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
