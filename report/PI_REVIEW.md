# PI-Style Review Checklist

Use this checklist before the final PDF submission. The standard is not "does
it run once"; the standard is whether a skeptical reviewer can reproduce the
Kaggle file and understand why the design should generalize to the private
leaderboard.

## Critical Risks

1. Data leakage: validation must split by user whenever possible. Random
   window-level splits can overestimate private leaderboard performance because
   windows from the same wrist-wearing user may share calibration, movement
   amplitude, and orientation statistics.
2. Label alignment: each 300-row sequence maps to exactly one label. Do not
   train row-level labels unless the label is deliberately repeated only as an
   implementation detail and evaluation is aggregated back to one file-level
   prediction.
3. Public leaderboard overfitting: daily submissions are limited and public
   score uses only half of the test set. Model selection should be driven by
   cross-validation, not small public leaderboard changes.
4. Reproducibility: the report, code seed, feature set, and final submission
   must agree. Keep `metrics.json`, `cv_metrics.csv`, and the submitted CSV.

## Report Bar

1. Preliminary analysis should include class counts, user counts, sequence
   length sanity checks, signal magnitude summaries, and at least one naive
   baseline.
2. Preprocessing claims need numbers. Each claimed improvement should have a
   before/after macro-F1 in `ablation_results.csv`.
3. Temporal modeling must explicitly explain the conversion from 300 one-second
   rows to one 5-minute prediction, including segment, derivative, rolling, and
   frequency-domain features.
4. Ablation should be about design choices, not just a model leaderboard. The
   table should isolate feature family, validation strategy, class weighting,
   and ensemble effects where possible.
5. Any AI assistance should be acknowledged according to the course policy, and
   no copied code or prose should appear in the final submission.

## Final PI-Style Audit Notes

- The final report now explains why the selected model is not the highest-OOF
  model: XGBoost and blend probes had higher or competitive OOF scores but lower
  public scores, so the final choice is justified through grouped CV, public
  evidence, and test-prior sanity checks together.
- Negative controls are explicitly reported: round-2 LGBM/XGBoost
  hyperparameter variants, temporal-shape feature expansion, and two submitted
  blend probes did not beat the selected LightGBM model.
- The label-2 weakness is not hidden. The report names label 2 as the dominant
  residual risk, reports its OOF F1, and describes why aggressive class-2 priors
  were rejected after public-score probes.
- The reproducibility chain is complete: final CSV path, SHA-256, seed, feature
  context, calibration source, submission log, and PDF build procedure are all
  documented.
