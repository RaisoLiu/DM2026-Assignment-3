# Change Summary

## Data

- Downloaded `nycu-data-mining-assignment-3.zip` from Kaggle.
- Extracted and normalized the folder layout to `data/raw/train/User_*` and
  `data/raw/test/User_*`.
- Verified 11,020 train windows, 6,849 test windows, 60 train users, 40 test
  users, no duplicate file IDs, and exact alignment between test `file_id` and
  `sample_submission.csv`.

## Code

- Added a reproducible file-level HAR pipeline under `src/dm2026_asg3/`.
- Implemented deterministic feature extraction with gravity/magnitude,
  temporal, rolling/segment, frequency, and cross-axis feature families.
- Added grouped cross-validation by user, class-balanced training, OOF
  probability calibration for macro-F1, feature-family ablations, feature
  caching, and selected-feature replay for submission.
- Added Kaggle submission generation with ID-order validation.
- Added `scripts/make_boosting_submission.py` to reproduce the final
  LightGBM/XGBoost submission variants from saved model-search calibration
  weights.
- Added `SUBMISSION_LOG.md`; future Kaggle uploads must update that file with
  score, status, SHA-256, and reproduction notes.

## Experiments

- Final selected model: `LGBMClassifier` (`lgbm_leaves63`) with position
  context features, seed 2026.
- HGB baseline grouped OOF macro-F1: 0.73823 after calibration.
- XGBoost position calibrated public score: 0.7933.
- LightGBM leaves63 position calibrated public score: 0.8106.
- May 6 blend stress tests:
  - OOF-best five-model blend public score: 0.8010.
  - Conservative LGBM/XGB blend public score: 0.7983.
  - Re-submitted selected LightGBM CSV public score: 0.8106.
- Round-2 single-model hyperparameter sweep and temporal-shape feature
  expansion did not improve grouped OOF macro-F1; these were kept as negative
  controls rather than promoted to the selected submission.
- Selected local CSV: `submissions/submission_lgbm_leaves63_calibrated.csv`.
- Selected CSV SHA-256:
  `f78034a7e72d25ab4d52baebed585e130ad810e1f5cfae7469205008d5d5035e`.

## Report

- Rewrote `report/DM_asg3_report.md` in English with actual dataset counts,
  class imbalance analysis, preprocessing rationale, label-alignment logic,
  measured ablation results, selected-model justification, and residual risks.
- Added LaTeX source `report/report.tex` and compiled the final PDF
  `report/DM_asg3_413551030.pdf` with `pdflatex`.
- Added `report/PI_REVIEW.md` as a review checklist for final submission.
- Strengthened the final report with a post-selection stress-test table
  explaining why higher-OOF blends were rejected after public-score and test
  prior checks.
- Published the reproducible code/report repository at
  `https://github.com/RaisoLiu/DM2026-Assignment-3`.

## Final Verification

- Recompiled `report/report.tex` twice and copied the result to
  `report/DM_asg3_413551030.pdf`.
- Verified with `pdfinfo` that the final PDF is 8 pages after adding the
  post-selection stress-test discussion.
- Final PDF SHA-256:
  `2f0640a8d1876a2e7142c878f1f180ad6d629c11ab5eee93c9c0890ca2c0f926`.
- Scanned `report.log` for unresolved references, rerun warnings, LaTeX
  warnings, and overfull/underfull boxes; no matches remain.
