# Reproducibility: DM2026 Assignment 3 (HAR)

## 1. Overview

The final Kaggle submissions for this project are produced by a single,
deterministic, CPU-only script:

    python scripts/build_synth_candidates.py

It reads **7 committed input CSVs** (test-space predictions) plus
`data/raw/sample_submission.csv` (for `Id` ordering) and writes three candidate
CSVs. Two of them are the graded submissions:

| File | Public F1 | Rank | SHA-256 |
|------|-----------|------|---------|
| `submissions/synth_agg_consensus.csv` | **0.8270** | 1 | `54075fcb799ed7edd75068e84016a8ac9daf9b3c2efebe196268ccce5f13a10e` |
| `submissions/synth_safe_flip37.csv`   | 0.8262 |   | `9921d49d9b72d6cd0c1943ca4332edcd8609df7fdbbef8cd6b7a9cca675d086a` |

This step is **byte-for-byte reproducible**: a fresh clone with the committed
inputs regenerates the exact SHA-256 values above (verified).

### Reproducibility strategy: "committed artifacts"

The pipeline has three layers:

1. **Final aggregation (deterministic CPU).** `build_synth_candidates.py` and
   the three `build_agg_5src_*.py` blenders are pure NumPy/pandas plus a
   deterministic Viterbi smoother. Given the same inputs they are bit-exact.
2. **Representation learning (GPU / SSL — stochastic).** The InceptionTime SSL
   encoders (v1, v2), the class-2 sequence specialist, and the BYOL encoder are
   trained on a GPU. CUDA kernels, multi-seed averaging, and SSL pretraining are
   not bit-reproducible across runs/hardware — re-training yields
   *near-equivalent* (not identical) probabilities.
3. **Raw data.** Kaggle accelerometer windows under `data/raw/`.

To satisfy the grading rule "code can be executed AND Kaggle results consistent
with code", we **pin the stochastic outputs as committed artifacts**: the small
`.npz` probability files and the input CSVs are committed, so the deterministic
final step reproduces the submitted CSVs exactly from a fresh clone. The
upstream GPU scripts are committed too, so the full chain is auditable and
*re-runnable*, but the pinned artifacts are the authoritative source for
byte-identical results.

> Note on metadata: `build_synth_candidates.py` additionally loads
> `artifacts/agg_generator/oof_anchor_8248_full.npz`,
> `artifacts/agg_generator/oof_cand3_full.npz`, and
> `artifacts/folds/holdout12_seed2027.csv`. These are used **only** to print a
> held-out-12 (HOS-12) F1 *estimate* in the metadata JSON. They do **not** affect
> the submitted CSV labels.

## 2. DAG

```
                              data/raw/  (Kaggle: train/ test/ sample_submission.csv)
                                  |  [RAW]
            +---------------------+-------------------------------+
            v                     v                               v
   sequence/*.npz         features/*.csv                 catch22_oof/*.csv
   [CPU]                  [CPU]                           [CPU]
      |                      |                               |
      v                      v                               v
 train_inceptiontime_full.py   make_group_folds.py     train_catch22_full.py
 (+ SSL encoder, multi-seed)   build_fresh_holdout12.py [CPU]
 train_c2_seq_specialist.py    [CPU]                      |
 [GPU-STOCHASTIC -> PINNED]      |                        v
      |                          |              catch22_full/test_proba_{lgbm,xgb}.npz
      v                          |                        |
 inception_full_v2/test_avg.npz  |   make_centered_meta_viterbi_submission.py
 inception_full_ssl/test_proba_avg3.npz  [CPU->PINNED]    |
 class_specialist/test_c{2,3}.npz       blend_search/*centered_meta*.npz
      |                          |              |          |
      +----------+---------------+--------------+----------+
                 v
      build_weighted_ensemble.py  ->  weighted_ensemble/{test,oof}.npz  [CPU]
                 |
      +----------+---------------------------+
      v          v                           v
 build_agg_5src_hos12_optimal.py   build_agg_5src_hos12_promote.py   build_agg_5src_spec_promote.py
 [CPU]                              [CPU]                             [CPU]
      |  agg_5src_hos12_optimal.csv      | agg_5src_hos12_promote.csv      | agg_5src_spec_promote.csv
      +----------+-----------------------+----------------+-------------+
                 |                                         |
   PINNED anchors (no committed producer):                |
   submission_h8_v3_sslv2_weighted.csv  (0.8248)          |
   submission_h8_uncertain_refined.csv  (0.8244)          |
   submission_ssl_hybrid_recover.csv    (0.8240, lineage make_ssl_hybrid_submission.py)
   cons_top3_vote.csv                                     |
                 +--------------------+--------------------+
                                      v
                       build_synth_candidates.py   [DETERMINISTIC-CPU]
                                      |
                    +-----------------+------------------+
                    v                 v                  v
        synth_agg_consensus.csv  synth_safe_flip37.csv  synth_rare_lift.csv
            (0.8270)                 (0.8262)
```

## 3. Per-node table

| Node | Type | Producer script | Inputs | Pinned? |
|------|------|-----------------|--------|---------|
| `synth_*.csv` (final) | DET-CPU | `build_synth_candidates.py` | 7 input CSVs + sample_submission (+metadata: 2 oof npz, holdout12) | inputs pinned |
| `agg_5src_hos12_optimal.csv` | DET-CPU | `build_agg_5src_hos12_optimal.py` | weighted_ensemble + 5 source npz | inputs pinned |
| `agg_5src_hos12_promote.csv` | DET-CPU | `build_agg_5src_hos12_promote.py` | same + self-promote c2/c3 | inputs pinned |
| `agg_5src_spec_promote.csv` | DET-CPU | `build_agg_5src_spec_promote.py` | same + class_specialist/test_c{2,3}.npz | inputs pinned |
| `submission_h8_v3_sslv2_weighted.csv` (0.8248) | PINNED | — (final submission) | SSL v2 ensemble | yes |
| `submission_h8_uncertain_refined.csv` (0.8244) | PINNED | — | uncertain-row refinement | yes |
| `submission_ssl_hybrid_recover.csv` (0.8240) | PINNED | `make_ssl_hybrid_submission.py` (lineage) | SSL blend + consensus | yes |
| `cons_top3_vote.csv` | PINNED | — | top-3 anchor vote | yes |
| `weighted_ensemble/{test,oof}.npz` | DET-CPU | `build_weighted_ensemble.py` | 5 source OOF/test npz | inputs pinned |
| `inception_full_v2/test_avg.npz` | GPU-STOCH | `train_inceptiontime_full.py` | sequence npz, SSL v2 encoder | PINNED authoritative |
| `inception_full_ssl/test_proba_avg3.npz` | GPU-STOCH | `train_inceptiontime_full.py` | sequence npz, SSL v1 encoder | PINNED authoritative |
| `class_specialist/test_c{2,3}.npz` | GPU-STOCH | `train_c2_seq_specialist.py` | sequence npz, folds | PINNED authoritative |
| `catch22_full/test_proba_{lgbm,xgb}.npz` | DET-CPU | `train_catch22_full.py` | catch24 features, sequence npz | inputs pinned |
| `blend_search/*centered_meta*.npz` | CPU->PINNED | `make_centered_meta_viterbi_submission.py` | OOF stacker features | yes |
| `agg_generator/oof_anchor_8248_full.npz`, `oof_cand3_full.npz` | PINNED | none (metadata-only) | — | yes (metadata only) |
| `folds/holdout12_seed2027.csv` | DET-CPU | `build_fresh_holdout12.py` | sgkf folds, holdout8 | inputs pinned |
| `folds/sgkf_seed2026.csv` | DET-CPU | `make_group_folds.py` | train_features.csv | — |
| `sequence/*.npz`, `features/*.csv` | DET-CPU | loaders / feature builders | data/raw | — |

## 4. Quick reproduce the 0.8270 submission (graded path)

This is byte-for-byte reproducible.

```bash
# 1. Clone
git clone https://github.com/RaisoLiu/DM2026-Assignment-3.git
cd DM2026-Assignment-3

# 2. Virtualenv + dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Kaggle data (needed for data/raw/sample_submission.csv = Id ordering)
kaggle competitions download -c nycu-data-mining-assignment-3 -p data/raw --force
unzip -q -o data/raw/nycu-data-mining-assignment-3.zip -d data/raw
#    Ensure data/raw/sample_submission.csv is present.

# 4. Build the final candidates (CPU, ~seconds)
make final        # == python scripts/build_synth_candidates.py + sha256sum

# 5. Verify byte-identical reproduction
sha256sum submissions/synth_agg_consensus.csv submissions/synth_safe_flip37.csv
# Expected:
# 54075fcb799ed7edd75068e84016a8ac9daf9b3c2efebe196268ccce5f13a10e  submissions/synth_agg_consensus.csv
# 9921d49d9b72d6cd0c1943ca4332edcd8609df7fdbbef8cd6b7a9cca675d086a  submissions/synth_safe_flip37.csv
```

`synth_agg_consensus.csv` (SHA `54075f...`) is the rank-1 public-F1-0.8270 file.

## 5. Full regeneration from raw data (best effort, GPU-stochastic)

`run_full_pipeline.sh` rebuilds the *entire* chain from Kaggle data. It is **NOT
byte-reproducible**: re-running the InceptionTime/SSL/BYOL/specialist training
produces near-equivalent (not identical) probability files, which shift a handful
of test labels. The committed `.npz` artifacts are the **authoritative** source
for the 0.8270/0.8262 SHAs above.

Stages (see `run_full_pipeline.sh`):

1. **Features / sequences / folds (CPU, deterministic).**
2. **SSL pretraining (GPU, STOCHASTIC).** SimCLR v1/v2 + BYOL v3 encoders.
3. **Sequence models / probabilities (GPU, STOCHASTIC).** InceptionTime (SSL v1
   3 seeds, SSL v2 2 seeds) + class-2 specialist.
4. **Catch24 trees (CPU, deterministic).** LGBM/XGB.
5. **Centered-meta Viterbi blend (CPU, seeded -> pinned) + weighted ensemble (CPU).**
6. **Aggressive 5-source candidates (CPU, deterministic).**
7. **Final synthesis (CPU, deterministic).** `build_synth_candidates.py`.

### Honesty note on what is / isn't deterministic

- **Bit-exact:** stages 4, 5 (ensemble), 6, 7 (CPU NumPy/pandas/seeded trees + Viterbi).
- **Near-equivalent only:** stages 2, 3, and the centered-meta source (GPU and/or
  SSL-pretrained). Do not expect the regenerated `.npz` to match the committed ones
  byte-for-byte. Use the committed artifacts + the Quick-reproduce path (Section 4)
  for the graded, byte-identical result. For convenience,
  `REPRO_ONLY=1 ./run_full_pipeline.sh` runs only the deterministic final step.
