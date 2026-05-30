#!/bin/bash
# Orchestrator: after BYOL pretrain finishes, extract embeddings + train LGBM on remote.
# Run on local; SSH-orchestrates remote.

set -euo pipefail
REMOTE=raiso@192.168.195.43
REMOTE_DIR=/home/raiso/DM2026-Asg3-byol

# 1. Wait for BYOL training to finish
echo "=== Waiting for BYOL training to finish ==="
while ssh "$REMOTE" "pgrep -f pretrain_byol_v3 >/dev/null"; do
    sleep 30
    ssh "$REMOTE" "tail -1 $REMOTE_DIR/artifacts/inception_byol_v3/train.log 2>&1"
done

# 2. Extract embeddings (uses BYOL encoder, runs on remote GPU)
echo "=== Extract embeddings ==="
ssh "$REMOTE" "cd $REMOTE_DIR && python3 scripts/extract_byol_embeddings.py \
    --checkpoint artifacts/inception_byol_v3/encoder.pt \
    --train-seq artifacts/sequence/train_sequences.npz \
    --test-seq artifacts/sequence/test_sequences.npz \
    --output-dir artifacts/byol_v3 \
    --batch-size 512"

# 3. Train LGBM on BYOL features (5-fold OOF)
echo "=== Train LGBM on BYOL features ==="
ssh "$REMOTE" "cd $REMOTE_DIR && python3 scripts/train_lgbm_chronos.py \
    --train-emb artifacts/byol_v3/train_embeddings.npz \
    --test-emb artifacts/byol_v3/test_embeddings.npz \
    --fold-file artifacts/folds/sgkf_seed2026.csv \
    --output-dir artifacts/byol_lgbm"

# 4. Sync results back
echo "=== Sync results back ==="
mkdir -p artifacts/byol_lgbm artifacts/byol_v3 artifacts/inception_byol_v3
rsync -avz "$REMOTE:$REMOTE_DIR/artifacts/byol_lgbm/" artifacts/byol_lgbm/
rsync -avz "$REMOTE:$REMOTE_DIR/artifacts/byol_v3/" artifacts/byol_v3/
rsync -avz "$REMOTE:$REMOTE_DIR/artifacts/inception_byol_v3/encoder.pt" artifacts/inception_byol_v3/
rsync -avz "$REMOTE:$REMOTE_DIR/artifacts/inception_byol_v3/metrics.json" artifacts/inception_byol_v3/

echo "=== Pipeline done ==="
echo "Local artifacts:"
ls -la artifacts/byol_lgbm/
