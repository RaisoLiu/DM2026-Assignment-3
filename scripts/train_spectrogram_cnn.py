#!/usr/bin/env python3
"""H3: 2D CNN on STFT spectrograms.

Apply Short-Time FFT to each 300-row × 6-channel sequence → (6, F, T') tensor.
Train a small 2D CNN. Provides orthogonal signal vs 1D time-domain models.

Outputs:
  - OOF probas: artifacts/spectrogram_oof/oof_seed{seed}.npz
  - Test probas: artifacts/spectrogram_full/test_seed{seed}.npz
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
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


N_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D-CNN on spectrograms.")
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=Path("artifacts/folds/sgkf_seed2026_train52.csv"),
    )
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/spectrogram_oof"),
    )
    parser.add_argument("--mode", choices=["oof", "full"], default="oof")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=8)
    return parser.parse_args()


def compute_stft(x: np.ndarray, n_fft: int = 64, hop: int = 8) -> np.ndarray:
    """Compute |STFT|^2 for each channel. Input (C, T) → output (C, F, T').

    Uses simple windowed FFT without overlap-add complexity.
    """
    C, T = x.shape
    window = np.hanning(n_fft)
    n_frames = max(1, (T - n_fft) // hop + 1)
    spec = np.zeros((C, n_fft // 2 + 1, n_frames), dtype=np.float32)
    for c in range(C):
        for t in range(n_frames):
            start = t * hop
            end = start + n_fft
            if end > T:
                break
            segment = x[c, start:end] * window
            fft = np.fft.rfft(segment)
            spec[c, :, t] = np.log1p(np.abs(fft) ** 2)
    return spec


class SpectrogramDataset(Dataset):
    def __init__(self, specs: np.ndarray, y: np.ndarray):
        self.specs = torch.from_numpy(specs.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.specs[idx], int(self.y[idx])


class SpectrogramCNN(nn.Module):
    def __init__(self, in_channels: int = 6, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        z = self.conv(x)
        return self.head(z)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq = np.load(args.seq_cache, allow_pickle=True)
    x_all = seq["x"].astype(np.float32)
    y_all = seq["y"].astype(np.int64)
    users_all = seq["users"].astype(str)
    file_ids_all = seq["file_ids"].astype(int)

    fold_df = pd.read_csv(args.fold_file)
    keep_ids = set(fold_df["file_id"].astype(int).tolist())
    mask = np.array([int(f) in keep_ids for f in file_ids_all])
    x = x_all[mask]
    y = y_all[mask]
    users = users_all[mask]
    file_ids = file_ids_all[mask]

    # Reorder to match fold_df
    fid_to_idx = {int(f): i for i, f in enumerate(file_ids)}
    target = fold_df["file_id"].astype(int).to_numpy()
    order = np.array([fid_to_idx[int(f)] for f in target], dtype=int)
    x = x[order]; y = y[order]; users = users[order]; file_ids = file_ids[order]
    fold_ids = fold_df["fold"].astype(int).to_numpy()
    print(f"Loaded {len(x)} train rows; computing STFTs...")

    t_start = time.time()
    specs = np.stack([compute_stft(x[i], args.n_fft, args.hop_length) for i in range(len(x))], axis=0)
    print(f"Spectrograms: {specs.shape} (computed in {time.time() - t_start:.1f}s)")

    n_folds = int(fold_ids.max())
    classes = np.arange(N_CLASSES, dtype=np.int64)
    oof_proba = np.full((len(x), N_CLASSES), np.nan, dtype=np.float32)
    fold_metrics = {}

    for fold in range(1, n_folds + 1):
        print(f"\n=== Fold {fold} ===")
        tr = fold_ids != fold
        va = fold_ids == fold
        train_loader = DataLoader(
            SpectrogramDataset(specs[tr], y[tr]),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            SpectrogramDataset(specs[va], y[va]),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )

        model = SpectrogramCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        best_macro = -1.0
        best_proba = None
        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            model.train()
            loss_acc, n_seen = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_acc += float(loss.detach().item()) * xb.size(0)
                n_seen += xb.size(0)
            scheduler.step()
            # Eval
            model.eval()
            probs = []
            with torch.no_grad():
                for xb, _ in valid_loader:
                    xb = xb.to(device, non_blocking=True)
                    logits = model(xb)
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            pred = probs.argmax(axis=1)
            macro = f1_score(y[va], pred, average="macro")
            per_class = f1_score(y[va], pred, average=None, labels=list(range(N_CLASSES)))
            print(f"  epoch {epoch:02d}: loss={loss_acc/max(1,n_seen):.4f} macroF1={macro:.4f} perClass=[{','.join(f'{p:.2f}' for p in per_class)}] {time.time()-t_epoch:.1f}s")
            if macro > best_macro:
                best_macro = float(macro)
                best_proba = probs
        oof_proba[va] = best_proba
        fold_metrics[fold] = {"best_macro_f1": best_macro}

    oof_pred = oof_proba.argmax(axis=1)
    full_macro = f1_score(y, oof_pred, average="macro")
    print(f"\nFull OOF macroF1: {full_macro:.6f}")
    print(classification_report(y, oof_pred, digits=4))

    out_oof = args.output_dir / f"oof_spectrogram_seed{args.seed}.npz"
    np.savez_compressed(
        out_oof,
        proba=oof_proba.astype(np.float32),
        classes=classes,
        label=y.astype(np.int64),
        file_id=file_ids.astype(np.int64),
        user_id=users.astype(str),
    )
    print(f"Wrote {out_oof}")

    # Test predictions: train one final model on all data, then predict test
    if args.mode == "full":
        print("\n=== Full-train + test predict ===")
        test_seq = np.load(args.test_seq_cache, allow_pickle=True)
        x_test = test_seq["x"].astype(np.float32)
        test_fids = test_seq["file_ids"].astype(int)
        test_users = test_seq["users"].astype(str)
        print(f"Test sequences: {x_test.shape}, computing STFTs...")
        t0 = time.time()
        test_specs = np.stack([compute_stft(x_test[i], args.n_fft, args.hop_length) for i in range(len(x_test))], axis=0)
        print(f"Test spectrograms in {time.time() - t0:.1f}s")

        # Train on full
        full_loader = DataLoader(
            SpectrogramDataset(specs, y),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        )
        model = SpectrogramCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs + 1):
            t_e = time.time()
            model.train()
            n = 0; l = 0.0
            for xb, yb in full_loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                l += float(loss.detach().item()) * xb.size(0); n += xb.size(0)
            scheduler.step()
            print(f"  full epoch {epoch:02d}: loss={l/max(1,n):.4f} {time.time()-t_e:.1f}s")
        # Predict test
        test_loader = DataLoader(
            SpectrogramDataset(test_specs, np.zeros(len(test_specs), dtype=np.int64)),
            batch_size=args.batch_size, shuffle=False, num_workers=2,
        )
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        test_proba = np.concatenate(probs, axis=0)
        out_test = args.output_dir / f"test_spectrogram_seed{args.seed}.npz"
        np.savez_compressed(
            out_test,
            proba=test_proba.astype(np.float32),
            classes=classes,
            file_id=test_fids.astype(np.int64),
            user_id=test_users.astype(str),
        )
        print(f"Wrote {out_test}")


if __name__ == "__main__":
    main()
