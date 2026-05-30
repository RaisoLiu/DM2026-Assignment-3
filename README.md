# DM2026 Assignment 3 — Human Activity Recognition (HAR)

Kaggle competition **nycu-data-mining-assignment-3**. Predict one activity label
(0–5) per 5-minute wrist-accelerometer window (300 rows of `mean_{x,y,z}` /
`std_{x,y,z}`). Metric: **macro-F1**.

**Result:** public leaderboard **rank 1, macro-F1 0.8270** (team `413551030`).
Report: [`report/DM_asg3_413551030.pdf`](report/DM_asg3_413551030.pdf).
Reproducibility / dependency DAG: [`report/REPRODUCIBILITY.md`](report/REPRODUCIBILITY.md).

## Quick start — reproduce the rank-1 submission

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Kaggle data (provides data/raw/sample_submission.csv used for Id ordering)
kaggle competitions download -c nycu-data-mining-assignment-3 -p data/raw --force
unzip -q -o data/raw/nycu-data-mining-assignment-3.zip -d data/raw

# Deterministic, CPU, ~seconds. Reproduces the graded submissions byte-for-byte.
make final
# 54075fcb...  submissions/synth_agg_consensus.csv   (public 0.8270, rank 1)
# 9921d49d...  submissions/synth_safe_flip37.csv      (public 0.8262)
```

## Method (summary)

1. **Feature engineering** — each 300×6 window → 399 statistical / gravity-magnitude
   / temporal-derivative / segment / FFT / cross-axis features (+6 user-sequence
   position features). Gravity/magnitude is the most load-bearing family.
2. **Class-bias calibration** — per-class multiplicative weights fit on OOF
   predictions to maximise macro-F1 (rare class 2 up-weighted ×5.86).
3. **Sequence modelling** — self-supervised InceptionTime (SimCLR/BYOL) + ROCKET /
   MiniRocket + Catch24 trees on the raw 6×300 sequences.
4. **Viterbi temporal smoothing** — each user's chronological file sequence is an
   HMM chain; a fold-fair transition prior corrects temporally implausible labels.
5. **Audited consensus aggregation** (`scripts/build_synth_candidates.py`) — a small,
   gated set of label flips over high-scoring anchors, validated on a held-out
   12-user split (HOS-12) to avoid public-leaderboard overfitting.

All validation uses **StratifiedGroupKFold grouped by user** (seed 2026), because
train and test users are disjoint.

## Reproducibility

`make final` is **byte-for-byte reproducible** from committed inputs. The upstream
GPU/SSL training steps are not bit-reproducible across hardware, so their trained
probability artifacts are committed as pinned inputs; the deterministic final step
always reproduces the submitted CSVs. Full audit chain + a best-effort end-to-end
GPU regeneration:

```bash
FULL=1 ./run_full_pipeline.sh      # rebuild everything from raw data (GPU)
```

See [`report/REPRODUCIBILITY.md`](report/REPRODUCIBILITY.md) for the complete DAG,
per-node determinism classification, and SHA-256 verification.

## Reproducible safety baseline (LightGBM-leaves63, public 0.8106)

```bash
python scripts/run_experiment.py --data-dir data/raw \
  --output-dir artifacts/hgb_fast_5fold --feature-cache-dir artifacts/features \
  --n-splits 5 --seed 2026 --fast --models hist_gradient_boosting
python scripts/explore_models.py --context position --models lgbm_leaves63 --seed 2026
python scripts/make_boosting_submission.py \
  --results-csv artifacts/model_search_position_proba/results_position.csv \
  --model lgbm_leaves63 --context position \
  --output submissions/submission_lgbm_leaves63_calibrated.csv --seed 2026
```

## Data layout

```text
data/raw/
  train/User_xxx/*.csv   # 60 users, 11,020 windows
  test/User_xxx/*.csv    # 40 users, 6,849 windows
  sample_submission.csv  # Id,Label
```
Each per-window CSV has `index, mean_x, mean_y, mean_z, std_x, std_y, std_z`
(+ `label` for train), one label per `file_id`.

## Repository map

- `scripts/build_synth_candidates.py` — final deterministic aggregation (graded).
- `scripts/` — feature build, SSL/InceptionTime, ROCKET, Catch24, Viterbi, blends.
- `src/dm2026_asg3/` — feature/data/model library.
- `run_full_pipeline.sh`, `Makefile` — orchestration (`make final`).
- `report/` — PDF report, `REPRODUCIBILITY.md`, figures.
- `SUBMISSION_LOG.md`, `experiments.md` — full experiment trail.
