#!/usr/bin/env python3
"""Inception fine-tune with aggressive class-2 augmentation.

Targets the class-2 F1 bottleneck (0.343 in current ensemble) by:
  1. Oversampling c2 train rows (×4 default) — sees each c2 example multiple times per epoch
  2. MixUp WITHIN c2 — interpolates random c2 pairs to create synthetic c2 samples
  3. Class weights in CE loss biased toward c2

Outputs: standard 6-class OOF + test proba, format-compatible with the
weighted-ensemble blender.

Initialized from existing SSL v2 checkpoint (artifacts/inception_ssl_v2/encoder.pt)
to keep training cost low. 5-fold StratifiedGroupKFold, multi-seed possible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


from train_inceptiontime_oof import (
    InceptionTime,
    N_CLASSES,
    apply_zscore,
    build_test_sequence_cache,
    compute_per_user_zscore,
)


class C2AugDataset(Dataset):
    """Returns either real samples or synthetic c2-mixup samples.

    With probability mixup_prob, returns:
      - x = lam * c2_a + (1 - lam) * c2_b for two random c2 samples
      - y = 2
    Otherwise returns a real sample (oversampled by sampler).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, mixup_prob: float = 0.3, mixup_alpha: float = 0.4):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.c2_indices = np.where(y == 2)[0]
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if len(self.c2_indices) >= 2 and np.random.random() < self.mixup_prob:
            i, j = np.random.choice(self.c2_indices, size=2, replace=False)
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1 - lam)  # bias toward heavier of the two — keep label clearly c2
            x_mix = lam * self.x[i] + (1 - lam) * self.x[j]
            return x_mix, 2
        return self.x[idx], int(self.y[idx])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-seq", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    p.add_argument("--test-seq", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--fold-file", type=Path, default=Path("artifacts/folds/sgkf_seed2026.csv"))
    p.add_argument("--ssl-checkpoint", type=Path, default=Path("artifacts/inception_ssl_v2/encoder.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/inception_c2_aug"))
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--c2-oversample", type=float, default=4.0)
    p.add_argument("--c2-class-weight", type=float, default=2.5)
    p.add_argument("--mixup-prob", type=float, default=0.3)
    p.add_argument("--mixup-alpha", type=float, default=0.4)
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

    train = np.load(args.train_seq, allow_pickle=True)
    test_x, test_users, test_fid = build_test_sequence_cache(args.data_dir, args.test_seq)
    x_all = train["x"].astype(np.float32)
    y_all = train["y"].astype(np.int64)
    users = train["users"].astype(str)
    file_ids = train["file_ids"].astype(int)
    print(f"Train: {len(x_all)} from {len(set(users))} users; Test: {len(test_x)} from {len(set(test_users))}")

    # Per-user z-score
    stats = compute_per_user_zscore(x_all, users, test_x, test_users)
    x_all = apply_zscore(x_all, users, stats)
    test_x = apply_zscore(test_x, test_users, stats)

    # Fold split
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_proba = np.zeros((len(y_all), N_CLASSES), dtype=np.float32)
    test_proba_folds = np.zeros((len(test_x), N_CLASSES), dtype=np.float32)

    class_weights = torch.ones(N_CLASSES, dtype=torch.float32, device=device)
    class_weights[2] = args.c2_class_weight
    print(f"Class weights: {class_weights.cpu().numpy().tolist()}")

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_all, y_all, users), start=1):
        print(f"\n=== Fold {fold} ===  train={len(tr_idx)}  val={len(va_idx)}")
        tr_y = y_all[tr_idx]
        tr_x = x_all[tr_idx]

        # WeightedRandomSampler: c2 samples weighted ×oversample, others ×1
        sample_w = np.ones(len(tr_y), dtype=np.float32)
        sample_w[tr_y == 2] = float(args.c2_oversample)
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(tr_y), replacement=True)

        tr_ds = C2AugDataset(tr_x, tr_y, mixup_prob=args.mixup_prob, mixup_alpha=args.mixup_alpha)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
        va_ds = C2AugDataset(x_all[va_idx], y_all[va_idx], mixup_prob=0.0)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = InceptionTime(in_channels=6, n_blocks=6, n_filters=32, bottleneck=32, kernel_sizes=(9, 19, 39)).to(device)
        if args.ssl_checkpoint.exists():
            ckpt = torch.load(args.ssl_checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("encoder_state_dict", ckpt)
            own = model.state_dict()
            loaded = 0
            for k, v in state.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v)
                    loaded += 1
            print(f"  Loaded {loaded} SSL keys from {args.ssl_checkpoint}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        ce = nn.CrossEntropyLoss(weight=class_weights)

        best_macro = 0.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            t = time.time()
            model.train()
            loss_acc, n_seen = 0.0, 0
            for xb, yb in tr_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = ce(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                loss_acc += float(loss.detach().item()) * xb.size(0)
                n_seen += xb.size(0)
            sched.step()

            # Eval
            model.eval()
            val_probas = []
            val_labels = []
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(device, non_blocking=True)
                    out = model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                    val_probas.append(torch.softmax(logits, dim=1).cpu().numpy())
                    val_labels.append(yb.numpy())
            val_probas = np.concatenate(val_probas)
            val_labels = np.concatenate(val_labels)
            val_pred = val_probas.argmax(axis=1)
            val_macro = f1_score(val_labels, val_pred, average="macro")
            val_c2 = f1_score(val_labels, val_pred, average=None)[2] if (val_labels == 2).any() else 0.0
            if val_macro > best_macro:
                best_macro = val_macro
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"  ep{epoch:02d} loss={loss_acc/max(n_seen,1):.4f} val_macro={val_macro:.4f} val_c2={val_c2:.4f} {time.time()-t:.1f}s")

        # Restore best, predict
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        val_probas = []
        with torch.no_grad():
            for xb, _ in va_loader:
                xb = xb.to(device, non_blocking=True)
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                val_probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        oof_proba[va_idx] = np.concatenate(val_probas)

        test_loader = DataLoader(
            C2AugDataset(test_x, np.zeros(len(test_x), dtype=np.int64), mixup_prob=0.0),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )
        test_probas = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                test_probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_proba_folds += np.concatenate(test_probas) / 5
        fold_macro = f1_score(y_all[va_idx], oof_proba[va_idx].argmax(axis=1), average="macro")
        fold_c2 = f1_score(y_all[va_idx], oof_proba[va_idx].argmax(axis=1), average=None)
        print(f"  Fold {fold} BEST: macro={fold_macro:.4f}  per-class={[f'{x:.3f}' for x in fold_c2]}")

    overall_macro = f1_score(y_all, oof_proba.argmax(axis=1), average="macro")
    overall_per_class = f1_score(y_all, oof_proba.argmax(axis=1), average=None)
    print(f"\n=== Overall OOF macro: {overall_macro:.4f} ===")
    print(f"Per-class: {[f'{x:.4f}' for x in overall_per_class]}")
    print(classification_report(y_all, oof_proba.argmax(axis=1), digits=4))

    np.savez_compressed(
        args.output_dir / "oof.npz",
        proba=oof_proba,
        label=y_all,
        file_id=file_ids,
        user_id=users,
        classes=np.arange(N_CLASSES, dtype=int),
    )
    np.savez_compressed(
        args.output_dir / "test_proba.npz",
        proba=test_proba_folds,
        file_id=test_fid,
        user_id=test_users,
        classes=np.arange(N_CLASSES, dtype=int),
    )
    (args.output_dir / "summary.json").write_text(json.dumps({
        "overall_oof_macro_f1": float(overall_macro),
        "per_class_oof_f1": [float(x) for x in overall_per_class],
        "args": vars(args),
    }, indent=2, default=str))
    print(f"\nWrote {args.output_dir}/")


if __name__ == "__main__":
    main()
