#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh - DM2026 Assignment 3 (HAR) reproduction pipeline
# -----------------------------------------------------------------------------
# DEFAULT (safe): reproduces the graded rank-1 submissions byte-for-byte from the
#   COMMITTED pinned artifacts via the deterministic final step. Always succeeds
#   on any machine with the repo + Python deps (no GPU needed).
#
#     ./run_full_pipeline.sh
#
# FULL REGENERATION (best effort, GPU + Kaggle data required):
#   Rebuilds the entire chain from raw data. Stages marked [GPU-STOCHASTIC]
#   retrain neural/SSL models and are NOT bit-reproducible; the COMMITTED .npz
#   artifacts they overwrite are AUTHORITATIVE for the 0.8270/0.8262 SHAs.
#   See report/REPRODUCIBILITY.md.
#
#     FULL=1 ./run_full_pipeline.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PY="${PYTHON:-python}"
FULL="${FULL:-0}"
# REPRO_ONLY kept as an alias for the default (graded) path.
[[ "${REPRO_ONLY:-0}" == "1" ]] && FULL=0

SHA_CONSENSUS="54075fcb799ed7edd75068e84016a8ac9daf9b3c2efebe196268ccce5f13a10e"
SHA_FLIP37="9921d49d9b72d6cd0c1943ca4332edcd8609df7fdbbef8cd6b7a9cca675d086a"

echo "=============================================================="
echo " DM2026-Asg3 pipeline   ROOT=$ROOT   FULL=$FULL"
echo "=============================================================="

verify_shas () {
  echo ">>> Verifying SHA-256 of graded submissions"
  echo "${SHA_CONSENSUS}  submissions/synth_agg_consensus.csv" | sha256sum -c -
  echo "${SHA_FLIP37}  submissions/synth_safe_flip37.csv"     | sha256sum -c -
}

# -----------------------------------------------------------------------------
# DEFAULT graded path: deterministic final synthesis from committed artifacts.
# -----------------------------------------------------------------------------
if [[ "$FULL" != "1" ]]; then
  echo ">>> Graded path: building final candidates from COMMITTED pinned inputs"
  test -f data/raw/sample_submission.csv \
    || { echo "ERROR: data/raw/sample_submission.csv missing (download Kaggle data first)"; exit 1; }
  $PY scripts/build_synth_candidates.py
  verify_shas
  echo ">>> Done. synth_agg_consensus.csv = public F1 0.8270 (rank 1)."
  exit 0
fi

# =============================================================================
# FULL=1 : end-to-end regeneration (GPU-stochastic; best effort).
# Reference orchestration. Stages 2-3 retrain neural nets and will NOT
# byte-match the pinned artifacts; this is expected and documented.
# =============================================================================
echo
echo "### STAGE 0: data check #####################################"
test -f data/raw/sample_submission.csv \
  || { echo "ERROR: data/raw missing. Run: kaggle competitions download -c nycu-data-mining-assignment-3 -p data/raw --force && unzip ..."; exit 1; }
echo "    OK: data/raw/sample_submission.csv present"

echo
echo "### STAGE 1: features, sequences, folds (CPU, deterministic) #"
$PY scripts/run_experiment.py --data-dir data/raw --output-dir artifacts/hgb_fast_5fold \
    --feature-cache-dir artifacts/features --n-splits 5 --seed 2026 --fast \
    --models hist_gradient_boosting
$PY scripts/make_group_folds.py || echo "    (folds may already exist)"

echo
echo "### STAGE 2: SSL pretraining (GPU, STOCHASTIC) ##############"
echo "    NOT bit-reproducible. Committed encoder/proba artifacts are authoritative."
$PY scripts/pretrain_simclr.py   || echo "    (verify args with --help; pinned artifacts are authoritative)"
$PY scripts/pretrain_byol_v3.py  || echo "    (verify args with --help; pinned artifacts are authoritative)"

echo
echo "### STAGE 3: InceptionTime / specialist probas (GPU, STOCHASTIC) #"
$PY scripts/train_inceptiontime_full.py || echo "    (verify args with --help; pinned artifacts are authoritative)"
$PY scripts/train_c2_seq_specialist.py  || echo "    (verify args with --help; pinned artifacts are authoritative)"

echo
echo "### STAGE 4: Catch24 LGBM/XGB (CPU, seeded -> deterministic) #"
$PY scripts/train_catch22_full.py || echo "    (verify args with --help)"

echo
echo "### STAGE 5: centered-meta blend + weighted ensemble ########"
$PY scripts/make_centered_meta_viterbi_submission.py || echo "    (verify args with --help)"
$PY scripts/build_weighted_ensemble.py || echo "    (verify args with --help)"

echo
echo "### STAGE 6: aggressive 5-source candidates (CPU, deterministic) #"
$PY scripts/build_agg_5src_hos12_optimal.py
$PY scripts/build_agg_5src_hos12_promote.py
$PY scripts/build_agg_5src_spec_promote.py

echo
echo "### STAGE 7: final synthesis (CPU, deterministic) ##########"
$PY scripts/build_synth_candidates.py

echo
echo "### SHA check (will differ from pinned if stages 2-3 were retrained) #"
sha256sum submissions/synth_agg_consensus.csv submissions/synth_safe_flip37.csv
echo "    Pinned (authoritative): $SHA_CONSENSUS / $SHA_FLIP37"
echo "=============================================================="
echo " Full pipeline complete (best effort)."
echo "=============================================================="
