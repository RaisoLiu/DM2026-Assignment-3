#!/usr/bin/env python3
"""Train InceptionTime on a chosen subset of users and emit predictions for
specified rows.

Two intended uses:
  1. Train on the 52 train-CV users → predict on the 8 held-out users (clean HOS eval).
  2. Train on the full 60 users → predict on the 6849 test windows (max signal for submission).

Reuses the InceptionTime model & augmentation defined in train_inceptiontime_oof.py.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_inceptiontime_oof import (
    FocalCE,
    InceptionTime,
    N_CLASSES,
    SIGNAL_COLS,
    apply_zscore,
    build_test_sequence_cache,
    class_balanced_weights,
    compute_per_user_zscore,
    make_eval_loader,
    make_train_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one sequence model on a user subset; emit predictions for specified rows.")
    parser.add_argument(
        "--model",
        choices=["inception", "patchtst", "tcn", "resnet1d"],
        default="inception",
    )
    parser.add_argument(
        "--pseudo-labels",
        type=Path,
        default=None,
    )
    parser.add_argument("--pseudo-weight", type=float, default=0.4)
    parser.add_argument(
        "--train-users-file",
        type=Path,
        required=True,
        help="CSV with user_id column listing users to TRAIN on.",
    )
    parser.add_argument(
        "--predict-mode",
        choices=["holdout", "test", "both"],
        default="test",
    )
    parser.add_argument(
        "--holdout-file",
        type=Path,
        default=Path("artifacts/folds/holdout8_seed2026.csv"),
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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--n-filters", type=int, default=32)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--kernel-sizes", type=str, default="9,19,39")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument(
        "--sampler",
        choices=["shuffle", "soft", "balanced"],
        default="shuffle",
    )
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Path to a SimCLR pretraining checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    kernels = tuple(int(k) for k in args.kernel_sizes.split(","))

    # Load training data and select users
    train_users_df = pd.read_csv(args.train_users_file)
    train_users = set(train_users_df["user_id"].astype(str).tolist())

    seq = np.load(args.seq_cache, allow_pickle=True)
    x_full = seq["x"].astype(np.float32)
    y_full = seq["y"].astype(np.int64)
    users_full = seq["users"].astype(str)
    file_ids_full = seq["file_ids"].astype(int)

    # Mask train rows
    train_mask = np.array([u in train_users for u in users_full], dtype=bool)
    x_train = x_full[train_mask]
    y_train = y_full[train_mask]
    users_train = users_full[train_mask]
    file_ids_train = file_ids_full[train_mask]
    print(
        f"Training on {len(x_train)} rows from {len(set(users_train))} users",
        flush=True,
    )

    # Load test sequences for z-score stats and optional prediction
    x_test, users_test, file_ids_test = build_test_sequence_cache(args.data_dir, args.test_seq_cache)
    print(
        f"Test cache: {len(x_test)} rows from {len(set(users_test))} users",
        flush=True,
    )

    # Per-user z-score using train+test stats
    stats = compute_per_user_zscore(x_train, users_train, x_test, users_test)
    x_train_z = apply_zscore(x_train, users_train, stats)

    # Holdout users: load and prepare
    if args.predict_mode in ("holdout", "both"):
        hos_df = pd.read_csv(args.holdout_file)
        hos_users = set(hos_df["user_id"].astype(str).tolist())
        hos_mask = np.array([u in hos_users for u in users_full], dtype=bool)
        x_hos = x_full[hos_mask]
        y_hos = y_full[hos_mask]
        users_hos = users_full[hos_mask]
        file_ids_hos = file_ids_full[hos_mask]
        # Use stats computed on train+test (HOS users are in train+test? They are in train but excluded above. We need their stats.)
        # Recompute stats including HOS for normalization
        stats_full = compute_per_user_zscore(np.concatenate([x_train, x_hos], axis=0),
                                             np.concatenate([users_train, users_hos], axis=0),
                                             x_test, users_test)
        x_hos_z = apply_zscore(x_hos, users_hos, stats_full)
        # Re-z-score train with the full stats (more robust normalization)
        x_train_z = apply_zscore(x_train, users_train, stats_full)
        print(f"HOS: {len(x_hos)} rows from {len(hos_users)} users", flush=True)

    if args.predict_mode in ("test", "both"):
        x_test_z = apply_zscore(x_test, users_test, stats)
        print(f"Test (for prediction): {len(x_test_z)} rows", flush=True)

    # Optionally append pseudo-labels for training (test rows with consensus labels)
    train_sample_weights = np.ones(len(y_train), dtype=np.float32)
    if args.pseudo_labels is not None and args.pseudo_labels.exists():
        pseudo_df = pd.read_csv(args.pseudo_labels)
        pseudo_ids = pseudo_df["Id"].astype(int).to_numpy()
        pseudo_labels = pseudo_df["Label"].astype(int).to_numpy()
        pseudo_conf = (
            pseudo_df["confidence_score"].astype(float).to_numpy()
            if "confidence_score" in pseudo_df.columns
            else np.ones(len(pseudo_df), dtype=float)
        )
        test_idx_map = {int(fid): i for i, fid in enumerate(file_ids_test)}
        sel = np.array([test_idx_map[int(i)] for i in pseudo_ids if int(i) in test_idx_map], dtype=int)
        sel_ids = np.array([int(i) for i in pseudo_ids if int(i) in test_idx_map], dtype=int)
        extra_x = x_test_z[sel] if args.predict_mode in ("test", "both") else apply_zscore(x_test, users_test, stats)[sel]
        id_to_label = dict(zip(pseudo_ids.tolist(), pseudo_labels.tolist()))
        id_to_conf = dict(zip(pseudo_ids.tolist(), pseudo_conf.tolist()))
        extra_y = np.array([id_to_label[int(i)] for i in sel_ids], dtype=np.int64)
        extra_u = users_test[sel].astype(str)
        extra_w = (np.array([id_to_conf[int(i)] for i in sel_ids], dtype=np.float32) * args.pseudo_weight).astype(np.float32)
        x_train_z = np.concatenate([x_train_z, extra_x], axis=0)
        y_train = np.concatenate([y_train, extra_y], axis=0)
        users_train = np.concatenate([users_train, extra_u], axis=0)
        train_sample_weights = np.concatenate([train_sample_weights, extra_w], axis=0)
        print(
            f"Added {len(extra_x)} pseudo-labels (weight {args.pseudo_weight}), total train rows {len(y_train)}",
            flush=True,
        )

    # Class weights
    cls_w_np = class_balanced_weights(y_train, beta=0.999)
    cls_w_np = np.clip(cls_w_np, a_min=None, a_max=2.0).astype(np.float32)
    print(f"Class weights: {np.round(cls_w_np, 3).tolist()}", flush=True)

    # Build train loader
    train_loader = make_train_loader(
        x_train_z, y_train, users_train, args.batch_size, args.seed, args.mixup_alpha, args.num_workers,
        sampler_mode=args.sampler, sample_weights=train_sample_weights,
    )

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
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from har_models import build_model
        model = build_model(args.model, dropout=args.dropout).to(device)
    if args.init_from_checkpoint and args.init_from_checkpoint.exists():
        ckpt = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        encoder_state = ckpt.get("encoder_state_dict", ckpt)
        own_state = model.state_dict()
        loaded = sum(1 for k, v in encoder_state.items() if k in own_state and own_state[k].shape == v.shape)
        for k, v in encoder_state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
        print(f"SSL init loaded {loaded} keys from {args.init_from_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cls_w = torch.from_numpy(cls_w_np).to(device)
    main_loss = FocalCE(gamma=args.focal_gamma, weight=cls_w)
    aux_loss = nn.BCEWithLogitsLoss()
    use_amp = (not args.no_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        loss_acc = 0.0
        n_seen = 0
        import torch.nn.functional as _F
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
                aux_raw = _F.binary_cross_entropy_with_logits(aux_logits, target_aux, reduction="none")
                aux_loss_val = (aux_raw * wb).sum() / wb.sum().clamp(min=1e-6)
                loss = main_loss_val + args.aux_weight * aux_loss_val
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_acc += float(loss.detach().item()) * xb.size(0)
            n_seen += xb.size(0)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}: loss={loss_acc / max(1, n_seen):.4f} {time.time() - t:.1f}s",
            flush=True,
        )

    # Predict
    @torch.no_grad()
    def predict(x_arr: np.ndarray, y_arr: np.ndarray | None, users_arr: np.ndarray) -> np.ndarray:
        if y_arr is None:
            y_dummy = np.zeros(len(x_arr), dtype=np.int64)
        else:
            y_dummy = y_arr
        loader = make_eval_loader(x_arr, y_dummy, users_arr, args.batch_size, args.num_workers)
        model.eval()
        outs = []
        for xb, _, _, _ in loader:
            xb = xb.to(device, non_blocking=True)
            ctx = torch.amp.autocast(device.type, dtype=torch.bfloat16) if use_amp else _NullCtx()
            with ctx:
                logits, _ = model(xb)
            outs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    if args.predict_mode in ("holdout", "both"):
        proba_hos = predict(x_hos_z, y_hos, users_hos)
        out_hos = args.output_dir / f"holdout_proba_seed{args.seed}.npz"
        np.savez_compressed(
            out_hos,
            proba=proba_hos.astype(np.float32),
            classes=np.arange(N_CLASSES, dtype=np.int64),
            label=y_hos.astype(np.int64),
            file_id=file_ids_hos.astype(np.int64),
            user_id=users_hos.astype(str),
        )
        # Quick metric on HOS
        from sklearn.metrics import f1_score, classification_report
        pred_hos = proba_hos.argmax(axis=1)
        hos_macro = f1_score(y_hos, pred_hos, average="macro")
        print(f"\nHOS prediction macro-F1 (no Viterbi): {hos_macro:.6f}")
        print(classification_report(y_hos, pred_hos, digits=4))
        print(f"Wrote {out_hos}", flush=True)

    if args.predict_mode in ("test", "both"):
        proba_test = predict(x_test_z, None, users_test)
        out_test = args.output_dir / f"test_proba_seed{args.seed}.npz"
        np.savez_compressed(
            out_test,
            proba=proba_test.astype(np.float32),
            classes=np.arange(N_CLASSES, dtype=np.int64),
            file_id=file_ids_test.astype(np.int64),
            user_id=users_test.astype(str),
        )
        print(f"Wrote {out_test}", flush=True)


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *_a):
        return False


if __name__ == "__main__":
    main()
