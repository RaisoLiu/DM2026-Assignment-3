# DM2026 Assignment 3 Experiment Log

Goal: improve DM2026 Assignment 3 performance while keeping the validation and
submission trail reproducible. The current best Kaggle public score is `0.8106`.

Evaluation protocol:

- Primary split: 5-fold `StratifiedGroupKFold`, grouped by `user_id`, seed 2026.
- Primary metric: macro-F1 on file-level predictions.
- Kaggle submissions are tracked in `SUBMISSION_LOG.md`; every future upload must update that log after the submission status/score is available.
- Calibration reporting:
  - `base`: direct argmax of OOF probabilities.
  - `global calibrated`: class-bias calibration tuned on all OOF predictions; useful for submission selection but optimistic.
  - `fold-fair calibrated`: each fold's decision weights are tuned using only the other folds' OOF predictions; preferred for rigorous model comparison.
- All promoted experiments must save OOF probabilities and per-class reports.

Current empirical status:

| family | validation signal | Kaggle public | notes |
|---|---:|---:|---|
| HGB engineered features | 0.73823 global calibrated OOF | 0.7922 | Stable first submission baseline. |
| XGBoost + user position features | 0.74977 global calibrated OOF | 0.7933 | Slight public gain over HGB. |
| LightGBM + user position features | 0.74771 global calibrated OOF | **0.8106** | Current selected submission. |
| LGBM/XGB/CatBoost ensemble | 0.75147 global calibrated OOF | not submitted | Small validation gain; daily quota used. |
| Viterbi sequence smoothing | 0.74964 fold-trained diagnostic on XGB | not submitted | Train-fold-only version gives limited gain. |
| MiniRocket raw sequence | 0.68618 OOF | not submitted | Worse than engineered features. |
| CNN raw sequence smoke | 0.41961 on first fold | not submitted | Not promising under current setup. |
| Rare-class one-vs-rest override | 0.72891 fixed-fold macro-F1 | not submitted | Class-2 specialist barely helps. |

Next rigorous steps:

1. Keep `SUBMISSION_LOG.md` current after every Kaggle upload.
2. Do not spend more daily quota without a clear hypothesis that can beat the
   `0.8106` LightGBM submission.
3. Prioritize class-2 feature engineering and calibration; it remains the main
   macro-F1 bottleneck.
4. Treat public score gains cautiously because the final grade may depend on the
   private leaderboard.
