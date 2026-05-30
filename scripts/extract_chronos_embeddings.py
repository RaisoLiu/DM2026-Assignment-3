#!/usr/bin/env python3
"""Extract Chronos-Bolt encoder embeddings per (window, channel).

For each test/train window of shape (6, 300), feed each of the 6 channels
through Chronos as an independent univariate series. The encoder returns
(num_patches, d_model) embeddings; we mean-pool over patches to get a
per-channel vector, then concatenate across the 6 channels.

Output: artifacts/external_pretrained/chronos_{train,test}_embeddings.npz
  - emb: (N, 6 * d_model) float32
  - file_ids, users (str), y (train only)
  - loc, scale: (N, 6) per-channel mean/std reported by Chronos
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from chronos import ChronosBoltPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seq", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    p.add_argument("--test-seq", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/external_pretrained"))
    p.add_argument("--model", default="amazon/chronos-bolt-small",
                   help="HF id. chronos-bolt-small ~46M, chronos-bolt-base ~205M.")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Number of (channel) series per forward pass.")
    p.add_argument("--pool", choices=["mean", "first", "max"], default="mean",
                   help="How to pool patch embeddings.")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def embed_array(pipe: ChronosBoltPipeline, x: np.ndarray, batch: int, pool: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x: (N, 6, 300). Returns (emb (N, 6*d_model), loc (N, 6), scale (N, 6))."""
    N, C, T = x.shape
    flat = torch.from_numpy(x.reshape(N * C, T).astype(np.float32))
    out_emb = []
    out_loc = []
    out_scale = []
    device = next(pipe.model.parameters()).device
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(flat), batch):
            chunk = flat[i:i + batch].to(device)
            emb, loc_scale = pipe.embed(chunk)
            # emb: (B, num_patches[+1], d_model)
            if pool == "mean":
                pooled = emb.mean(dim=1)  # (B, d_model)
            elif pool == "first":
                pooled = emb[:, 0, :]
            elif pool == "max":
                pooled = emb.max(dim=1).values
            out_emb.append(pooled.float().cpu().numpy())
            out_loc.append(loc_scale[0].float().cpu().numpy())
            out_scale.append(loc_scale[1].float().cpu().numpy())
            if (i // batch) % 50 == 0:
                done = i + len(chunk)
                eta = (len(flat) - done) * (time.time() - t0) / max(done, 1)
                print(f"  {done}/{len(flat)}  eta {eta:.0f}s", flush=True)
    emb_arr = np.concatenate(out_emb, axis=0).reshape(N, C, -1)
    loc_arr = np.concatenate(out_loc, axis=0).reshape(N, C)
    scale_arr = np.concatenate(out_scale, axis=0).reshape(N, C)
    # Concat 6 channels into single vector
    emb_flat = emb_arr.reshape(N, C * emb_arr.shape[-1])
    return emb_flat.astype(np.float32), loc_arr.astype(np.float32), scale_arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; loading {args.model}...")
    pipe = ChronosBoltPipeline.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float32,
    )
    pipe.model.eval()

    train = np.load(args.train_seq, allow_pickle=True)
    test = np.load(args.test_seq, allow_pickle=True)

    print(f"Train: {train['x'].shape}  Test: {test['x'].shape}")

    print("Embedding TRAIN...")
    t_emb, t_loc, t_scale = embed_array(pipe, train["x"], args.batch_size, args.pool)
    print(f"  done: train_emb shape {t_emb.shape}")

    out_train = args.output_dir / f"chronos_train_embeddings.npz"
    np.savez_compressed(
        out_train,
        emb=t_emb,
        loc=t_loc,
        scale=t_scale,
        file_ids=train["file_ids"],
        users=train["users"],
        y=train["y"],
        model=args.model,
        pool=args.pool,
    )
    print(f"  wrote {out_train}")

    print("Embedding TEST...")
    s_emb, s_loc, s_scale = embed_array(pipe, test["x"], args.batch_size, args.pool)
    print(f"  done: test_emb shape {s_emb.shape}")

    out_test = args.output_dir / f"chronos_test_embeddings.npz"
    np.savez_compressed(
        out_test,
        emb=s_emb,
        loc=s_loc,
        scale=s_scale,
        file_ids=test["file_ids"],
        users=test["users"],
        model=args.model,
        pool=args.pool,
    )
    print(f"  wrote {out_test}")


if __name__ == "__main__":
    main()
