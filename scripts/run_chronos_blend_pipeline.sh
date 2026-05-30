#!/bin/bash
# Day-16 slot-2: build Chronos-augmented 5-source weighted ensemble
# Run after train_lgbm_chronos.py finishes.

set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Add 'label' alias to Chronos OOF so build_weighted_ensemble can load it
.venv/bin/python - <<'PY'
import numpy as np
d = np.load("artifacts/chronos_lgbm/oof_proba.npz", allow_pickle=True)
out = {k: d[k] for k in d.files}
if "label" not in out:
    out["label"] = out["y"]
np.savez_compressed("artifacts/chronos_lgbm/oof_proba.npz", **out)
print("Added 'label' alias to chronos OOF")
PY

# 2. Search 5-source weighted ensemble (existing 4 + chronos)
echo "=== 5-source weighted ensemble search ==="
.venv/bin/python scripts/build_weighted_ensemble.py \
  --source-oofs \
    artifacts/inception_oof_ssl/oof_inception_avg3.npz \
    artifacts/blend_search/oof_blend_centered_meta_round2_best.npz \
    artifacts/catch22_oof/oof_catch22_raw_lgbm_c22.npz \
    artifacts/catch22_oof/oof_catch22_raw_xgb_c22.npz \
    artifacts/chronos_lgbm/oof_proba.npz \
  --source-tests \
    artifacts/inception_full_ssl/test_proba_avg3.npz \
    artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz \
    artifacts/catch22_full/test_proba_lgbm.npz \
    artifacts/catch22_full/test_proba_xgb.npz \
    artifacts/chronos_lgbm/test_proba.npz \
  --source-names inception_ssl centered_meta catch22_lgbm catch22_xgb chronos \
  --lnuo-folds \
    artifacts/folds/lnuo15_seed2026.csv \
    artifacts/folds/lnuo15_seed2027.csv \
    artifacts/folds/lnuo15_seed2028.csv \
    artifacts/folds/lnuo15_seed2029.csv \
    artifacts/folds/lnuo15_seed2030.csv \
    artifacts/folds/lnuo15_seed2031.csv \
  --fold-file artifacts/folds/sgkf_seed2026_train52.csv \
  --output-dir artifacts/weighted_ensemble_chronos \
  --n-trials 5000

# 3. Apply Viterbi smoothing to the new test probas
echo ""
echo "=== Viterbi smoothing ==="
.venv/bin/python scripts/evaluate_sequence_smoothing.py \
  --oof-npz artifacts/weighted_ensemble_chronos/oof.npz \
  --test-npz artifacts/weighted_ensemble_chronos/test.npz \
  --output-test artifacts/weighted_ensemble_chronos/test_viterbi.npz

# 4. Build submission CSV
echo ""
echo "=== Build submission ==="
.venv/bin/python - <<'PY'
import numpy as np, pandas as pd, hashlib
from pathlib import Path
d = np.load("artifacts/weighted_ensemble_chronos/test_viterbi.npz", allow_pickle=True)
fid = d["file_id"].astype(int)
pred = d["pred_viterbi"].astype(int)
df = pd.DataFrame({"Id": fid, "Label": pred})
sample = pd.read_csv("data/raw/sample_submission.csv")
df = df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
out = Path("submissions/submission_chronos_5src_viterbi.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out}")
print(f"SHA: {hashlib.sha256(out.read_bytes()).hexdigest()}")
print(f"Label counts: {df['Label'].value_counts().sort_index().to_dict()}")
PY

echo "=== Pipeline complete ==="
