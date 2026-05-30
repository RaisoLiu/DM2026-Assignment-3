#!/usr/bin/env python3
"""Class-2 specialist on raw 6×300 sequences (no pooling) — targets the c2
F1 bottleneck (existing best 0.343, foundation-model variants ≤ 0.10).

Mechanism:
  - InceptionTime backbone trained as binary c2 vs not-c2.
  - REMOVES the final GAP — instead uses a learnable attention pool that
    decides which timesteps matter for the c2 decision. This preserves the
    transition-shape signal that GAP destroys.
  - Trained on the full 11,020 train rows with 5-fold StratifiedGroupKFold OOF.
  - Outputs binary c2 probability per row (test predictions too).

Use case: combine with the existing class-specialist artifacts in a c2-aware
hybrid recovery (refine row labels only where c2 specialist is highly confident).

Run: .venv/bin/python scripts/train_c2_seq_specialist.py
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
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


from pretrain_byol_v3 import InceptionBlock  # reuse the deeper InceptionBlock


class AttentionPool1d(nn.Module):
    """Learnable per-timestep attention pooling (replaces GAP).

    For input (B, C, T), computes per-timestep score (B, T) via a tiny MLP,
    softmaxes over T, then returns the attention-weighted average over C
    yielding (B, C). Self-attention WITHIN time, parameterized by a learnable
    query, NOT cross-attention.
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scores = self.score(x).squeeze(1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        out = (x * weights.unsqueeze(1)).sum(dim=-1)  # (B, C)
        return out


class InceptionC2(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        n_blocks: int = 6,
        n_filters: int = 32,
        bottleneck: int = 32,
        kernel_sizes=(9, 19, 39),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        block_out = n_filters * (len(kernel_sizes) + 1)
        prev_channels = in_channels
        residual_in = in_channels
        for i in range(n_blocks):
            self.blocks.append(
                InceptionBlock(prev_channels, n_filters, bottleneck, kernel_sizes)
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
        self.pool = AttentionPool1d(prev_channels)
        self.head = nn.Sequential(
            nn.Linear(prev_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = block(x)
            if shortcut is not None:
                x = x + shortcut(residual)
                residual = x
        feat = self.pool(x)
        return self.head(feat).squeeze(-1)


class SeqBinDataset(Dataset):
    def __init__(self, x, y_bin):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y_bin.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-seq", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    p.add_argument("--test-seq", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/c2_seq_specialist"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="If unset, uses n_neg/n_pos.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
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
    print(f"Device: {device}")

    train = np.load(args.train_seq, allow_pickle=True)
    test = np.load(args.test_seq, allow_pickle=True)
    x = train["x"]
    y_full = train["y"].astype(int)
    users = train["users"].astype(str)
    file_ids = train["file_ids"].astype(int)

    y_bin = (y_full == 2).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int((1 - y_bin).sum())
    print(f"Class-2 positives: {n_pos}/{len(y_bin)} ({100*n_pos/len(y_bin):.2f}%)")
    pos_weight = args.pos_weight if args.pos_weight is not None else float(n_neg / max(n_pos, 1))
    print(f"pos_weight: {pos_weight:.2f}")

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_proba = np.zeros(len(y_bin), dtype=np.float32)
    fold_metrics = []

    test_x = test["x"]
    test_fid = test["file_ids"].astype(int)
    test_users = test["users"].astype(str)
    test_proba_folds = np.zeros(len(test_x), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y_bin, users), start=1):
        print(f"\n=== Fold {fold} ===  train={len(tr_idx)} val={len(va_idx)}")
        model = InceptionC2().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

        tr_loader = DataLoader(
            SeqBinDataset(x[tr_idx], y_bin[tr_idx]),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )
        va_loader = DataLoader(
            SeqBinDataset(x[va_idx], y_bin[va_idx]),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )
        test_loader = DataLoader(
            SeqBinDataset(test_x, np.zeros(len(test_x), dtype=np.int64)),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )

        best_f1 = 0.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            t = time.time()
            model.train()
            loss_acc = 0.0
            n = 0
            for xb, yb in tr_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = bce(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                loss_acc += float(loss.detach().item()) * xb.size(0)
                n += xb.size(0)
            sched.step()
            avg_loss = loss_acc / max(1, n)

            # Validation
            model.eval()
            val_probas = []
            val_labels = []
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(device, non_blocking=True)
                    val_probas.append(torch.sigmoid(model(xb)).cpu().numpy())
                    val_labels.append(yb.numpy())
            val_probas = np.concatenate(val_probas)
            val_labels = np.concatenate(val_labels).astype(int)
            val_ap = average_precision_score(val_labels, val_probas)
            val_f1 = f1_score(val_labels, (val_probas >= 0.5).astype(int))

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"  epoch {epoch:02d}: loss={avg_loss:.4f} val_AP={val_ap:.4f} val_F1={val_f1:.4f} {time.time()-t:.1f}s")

        # Restore best and predict
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        val_probas = []
        with torch.no_grad():
            for xb, _ in va_loader:
                xb = xb.to(device, non_blocking=True)
                val_probas.append(torch.sigmoid(model(xb)).cpu().numpy())
        oof_proba[va_idx] = np.concatenate(val_probas)

        # Test predictions for this fold (averaged across folds)
        test_probas = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                test_probas.append(torch.sigmoid(model(xb)).cpu().numpy())
        test_proba_folds += np.concatenate(test_probas) / args.n_splits

        fold_ap = average_precision_score(y_bin[va_idx], oof_proba[va_idx])
        fold_f1 = f1_score(y_bin[va_idx], (oof_proba[va_idx] >= 0.5).astype(int))
        print(f"  fold {fold} BEST: AP={fold_ap:.4f} F1@0.5={fold_f1:.4f}")
        fold_metrics.append({"fold": fold, "ap": float(fold_ap), "f1_at_0.5": float(fold_f1)})

    overall_ap = average_precision_score(y_bin, oof_proba)
    overall_f1 = f1_score(y_bin, (oof_proba >= 0.5).astype(int))
    print(f"\n=== Overall ===")
    print(f"OOF AP: {overall_ap:.4f}")
    print(f"OOF F1@0.5: {overall_f1:.4f}")

    # Find best threshold for max F1
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.02, 0.98, 97):
        f1 = f1_score(y_bin, (oof_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    print(f"Best OOF F1: {best_f1:.4f} at threshold {best_thresh:.3f}")

    np.savez_compressed(
        args.output_dir / "oof_c2_seq.npz",
        proba=oof_proba,
        binary_label=y_bin,
        full_label=y_full,
        file_id=file_ids,
        user_id=users,
        target_class=2,
        best_threshold=best_thresh,
    )
    np.savez_compressed(
        args.output_dir / "test_c2_seq.npz",
        proba=test_proba_folds.astype(np.float32),
        file_id=test_fid,
        user_id=test_users,
        target_class=2,
        best_threshold=best_thresh,
    )

    summary = {
        "oof_ap": float(overall_ap),
        "oof_f1_at_0.5": float(overall_f1),
        "best_oof_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "fold_metrics": fold_metrics,
        "n_positives": int(n_pos),
        "n_negatives": int(n_neg),
        "pos_weight": float(pos_weight),
        "epochs": int(args.epochs),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
