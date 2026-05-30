#!/usr/bin/env python3
"""Extract BYOL-v3 encoder embeddings for downstream LGBM classification.

For each (6,300) window:
  - Forward through BYOL online encoder (12 blocks, 4 kernel scales).
  - Returns a 160-dim feature vector per window.

Output: artifacts/byol_v3/{train,test}_embeddings.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrain_byol_v3 import InceptionEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/inception_byol_v3/encoder.pt"))
    p.add_argument("--train-seq", type=Path, default=Path("artifacts/sequence/train_sequences.npz"))
    p.add_argument("--test-seq", type=Path, default=Path("artifacts/sequence/test_sequences.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/byol_v3"))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--n-filters", type=int, default=32)
    return p.parse_args()


def embed_array(model: InceptionEncoder, x: np.ndarray, batch: int, device) -> np.ndarray:
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            chunk = torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)
            feat = model(chunk).cpu().numpy()
            out.append(feat)
            if (i // batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(x)}  elapsed {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading encoder from {args.checkpoint}")

    encoder = InceptionEncoder(
        in_channels=6,
        n_blocks=args.n_blocks,
        n_filters=args.n_filters,
        kernel_sizes=(9, 19, 39, 79),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.4f}")

    train = np.load(args.train_seq, allow_pickle=True)
    test = np.load(args.test_seq, allow_pickle=True)
    print(f"Train: {train['x'].shape}  Test: {test['x'].shape}")

    print("Embedding TRAIN...")
    t_emb = embed_array(encoder, train["x"], args.batch_size, device)
    print(f"  shape: {t_emb.shape}")

    out_train = args.output_dir / "train_embeddings.npz"
    np.savez_compressed(
        out_train,
        emb=t_emb,
        y=train["y"],
        file_ids=train["file_ids"],
        users=train["users"],
    )
    print(f"  wrote {out_train}")

    print("Embedding TEST...")
    s_emb = embed_array(encoder, test["x"], args.batch_size, device)
    print(f"  shape: {s_emb.shape}")

    out_test = args.output_dir / "test_embeddings.npz"
    np.savez_compressed(
        out_test,
        emb=s_emb,
        file_ids=test["file_ids"],
        users=test["users"],
    )
    print(f"  wrote {out_test}")


if __name__ == "__main__":
    main()
