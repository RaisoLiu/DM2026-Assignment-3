#!/bin/bash
# Day-16 slot-2: Build 5-source ensemble with BYOL LGBM added; emit submission.
# Assumes scripts/run_byol_pipeline_remote.sh has finished and the OOF/test probas
# are synced to artifacts/byol_lgbm/.

set -euo pipefail
cd "$(dirname "$0")/.."

# Sanity checks
test -f artifacts/byol_lgbm/oof_proba.npz || { echo "Missing BYOL OOF"; exit 1; }
test -f artifacts/byol_lgbm/test_proba.npz || { echo "Missing BYOL test proba"; exit 1; }

# 1. Add `label` alias to BYOL OOF (build_weighted_ensemble.py expects `label`).
.venv/bin/python - <<'PY'
import numpy as np
d = np.load("artifacts/byol_lgbm/oof_proba.npz", allow_pickle=True)
out = {k: d[k] for k in d.files}
if "label" not in out:
    out["label"] = out["y"]
np.savez_compressed("artifacts/byol_lgbm/oof_proba.npz", **out)
print("Added 'label' alias to BYOL OOF")
PY

# 2. Inspect per-class BYOL OOF F1 before committing to blend.
.venv/bin/python - <<'PY'
import numpy as np
from sklearn.metrics import classification_report, f1_score
d = np.load("artifacts/byol_lgbm/oof_proba.npz", allow_pickle=True)
y = d["label"]
proba = d["proba"]
pred = proba.argmax(axis=1)
print(f"BYOL LGBM OOF macro-F1: {f1_score(y, pred, average='macro'):.4f}")
print(classification_report(y, pred, digits=4))
PY

# 3. Build 5-source weighted ensemble (4 existing + BYOL).
echo "=== 5-source weighted ensemble search ==="
.venv/bin/python scripts/build_weighted_ensemble.py \
  --source-oofs \
    artifacts/inception_oof_ssl/oof_inception_avg3.npz \
    artifacts/blend_search/oof_blend_centered_meta_round2_best.npz \
    artifacts/catch22_oof/oof_catch22_raw_lgbm_c22.npz \
    artifacts/catch22_oof/oof_catch22_raw_xgb_c22.npz \
    artifacts/byol_lgbm/oof_proba.npz \
  --source-tests \
    artifacts/inception_full_ssl/test_proba_avg3.npz \
    artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz \
    artifacts/catch22_full/test_proba_lgbm.npz \
    artifacts/catch22_full/test_proba_xgb.npz \
    artifacts/byol_lgbm/test_proba.npz \
  --source-names inception_ssl centered_meta catch22_lgbm catch22_xgb byol_v3 \
  --lnuo-folds \
    artifacts/folds/lnuo15_seed2026.csv \
    artifacts/folds/lnuo15_seed2027.csv \
    artifacts/folds/lnuo15_seed2028.csv \
    artifacts/folds/lnuo15_seed2029.csv \
    artifacts/folds/lnuo15_seed2030.csv \
    artifacts/folds/lnuo15_seed2031.csv \
  --fold-file artifacts/folds/sgkf_seed2026_train52.csv \
  --output-dir artifacts/weighted_ensemble_byol \
  --n-trials 5000

# 4. Apply Viterbi smoothing per-user using OOF transitions.
.venv/bin/python - <<'PY'
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from evaluate_sequence_smoothing import estimate_transition_model, tune_viterbi_params, viterbi_predict_by_user

oof = np.load("artifacts/weighted_ensemble_byol/oof.npz", allow_pickle=True)
test = np.load("artifacts/weighted_ensemble_byol/test.npz", allow_pickle=True)

classes = np.arange(6, dtype=int)
oof_proba = oof["proba"].astype(np.float64)
oof_y = oof["label"].astype(int)
oof_fid = oof["file_id"].astype(int)
oof_uid = oof["user_id"].astype(str)

test_proba = test["proba"].astype(np.float64)
test_fid = test["file_id"].astype(int)
test_uid = test["user_id"].astype(str)

class_w = np.ones(6, dtype=np.float64)
best_params = tune_viterbi_params(
    proba=oof_proba,
    y=oof_y,
    classes=classes,
    file_ids=oof_fid,
    user_ids=oof_uid,
    class_weights=class_w,
    alpha_grid=[0.1, 0.3, 1.0, 3.0],
    beta_grid=[0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30],
    stay_grid=[0.0],
)
print(f"Best Viterbi α={best_params['alpha']} β={best_params['beta']} train_F1={best_params['train_macro_f1']:.4f}")

transition, start = estimate_transition_model(oof_y, classes, oof_fid, oof_uid, alpha=best_params["alpha"])
pred = viterbi_predict_by_user(
    test_proba,
    classes,
    test_fid,
    test_uid,
    class_w,
    transition,
    start,
    beta=best_params["beta"],
    stay_bonus=best_params.get("stay_bonus", 0.0),
)

np.savez_compressed(
    "artifacts/weighted_ensemble_byol/test_viterbi.npz",
    proba=test_proba.astype(np.float32),
    pred_viterbi=pred.astype(np.int64),
    classes=classes,
    file_id=test_fid,
    user_id=test_uid,
    viterbi_alpha=best_params["alpha"],
    viterbi_beta=best_params["beta"],
)
print(f"Wrote artifacts/weighted_ensemble_byol/test_viterbi.npz")

# Quick summary
import pandas as pd
counts = pd.Series(pred).value_counts().sort_index().to_dict()
print(f"Test pred counts: {counts}")
PY

# 5. Build submission CSV from Viterbi.
.venv/bin/python - <<'PY'
import numpy as np, pandas as pd, hashlib
from pathlib import Path
d = np.load("artifacts/weighted_ensemble_byol/test_viterbi.npz", allow_pickle=True)
fid = d["file_id"].astype(int)
pred = d["pred_viterbi"].astype(int)
df = pd.DataFrame({"Id": fid, "Label": pred})
sample = pd.read_csv("data/raw/sample_submission.csv")
df = df.set_index("Id").reindex(sample["Id"].astype(int)).reset_index()
out = Path("submissions/submission_byol_5src_viterbi.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out}")
print(f"SHA: {hashlib.sha256(out.read_bytes()).hexdigest()}")
print(f"Label counts: {df['Label'].value_counts().sort_index().to_dict()}")
PY

echo "=== Pipeline complete; submission at submissions/submission_byol_5src_viterbi.csv ==="
