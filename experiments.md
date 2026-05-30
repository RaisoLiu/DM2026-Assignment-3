# DM2026 Assignment 3 Experiment Log

Goal: improve DM2026 Assignment 3 performance while keeping the validation and
submission trail reproducible. The current best Kaggle public score is `0.8130`.

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
| Viterbi sequence smoothing | 0.74964 fold-trained diagnostic on XGB | 0.7996 | Submitted May 7; below the selected LightGBM score. |
| Rocket/meta blend + Viterbi | 0.76931 fold-fair Viterbi OOF | **0.8130** | Current best public submission: `submissions/submission_centered_meta_viterbi_oof07693.csv`. Artifact: `artifacts/sequence_smoothing_centered_meta_round2_best/summary.csv`. Unsmooth fine-calibrated blend is 0.76488; full-train global Viterbi translation used alpha 1.0 and beta 0.12. |
| Rocket/meta blend + fixed Viterbi/run/proba relabel screen | 0.80278 local OOF-screen | 0.8068 | Best local OOF-screen artifact: `artifacts/local_best_oof_0802784/summary.json`. Builds on fixed fold-param Viterbi, run-level relabeling, and an aggressive probability-threshold relabel screen over all class transitions. Public score shown is only for the earlier 0.77491 translation, which fell below the simpler centered-meta Viterbi submission. Do not assume the 0.80278 OOF-screen will translate to public/private LB without a separate test-time rule implementation and upload. |
| MiniRocket raw sequence | 0.68618 OOF | not submitted | Worse than engineered features. |
| CNN raw sequence smoke | 0.41961 on first fold | not submitted | Not promising under current setup. |
| Rare-class one-vs-rest override | 0.72891 fixed-fold macro-F1 | not submitted | Class-2 specialist barely helps. |

Next rigorous steps:

1. Keep `SUBMISSION_LOG.md` current after every Kaggle upload.
2. Do not spend more daily quota without a clear hypothesis that can beat the
   `0.8130` centered-meta Viterbi submission.
3. Prioritize class-2 feature engineering and calibration; it remains the main
   macro-F1 bottleneck.
4. Treat public score gains cautiously because the final grade may depend on the
   private leaderboard.
5. Do not rerun weak probes without a new reason: Hydra/MultiRocketHydra,
   Arsenal, RIST, augmented CNN, CatBoost depth8, and feature+proba stackers did
   not beat the current Rocket/meta blend path.

## 2026-05-08 daily log

Active goal: keep improving strict OOF validation macro-F1 toward `0.8000`;
do not upload to Kaggle unless explicitly requested.

Current best local validation remains:

- Strict fold-fair OOF macro-F1: `0.769308432188306`
- Artifact: `artifacts/sequence_smoothing_centered_meta_round2_best/summary.csv`
- Prediction artifact: `artifacts/sequence_smoothing_centered_meta_round2_best/centered_meta_viterbi_predictions.npz`
- Underlying unsmoothed/fine-calibrated blend: `0.7648817726531704`
- Main bottleneck: class `2` F1 around `0.3370`; class `5` recall remains a secondary issue.

Kaggle/submission status:

- Kaggle username found locally: `raisoliutw`
- Uploaded on request: `submissions/submission_blend_rocket_event_oof07610.csv`
- That file corresponds to OOF `0.761016` and public score `0.8062`
- No upload was made after that; daily goal work stayed local.

Environment work:

- GPU PyTorch is installed and verified in `.venv`:
  - `torch 2.11.0+cu130`
  - CUDA `13.0`
  - `torch.cuda.is_available() == True`
  - GPU: `NVIDIA GeForce RTX 3070`
- `tabpfn 7.1.1` is installed, but it cannot be used in this non-interactive
  environment without a TabPFN token/license for model weights.
- Attempted TabPFN downgrade was stopped; torch was rechecked afterward and
  remained healthy.

Ideas researched online and mapped to experiments:

- ROCKET/MiniRocket/MultiRocket-style convolution kernels
- HIVE-COTE style components: interval, dictionary, shapelet, and Arsenal-like classifiers
- InceptionTime/deep CNN sequence models
- TabPFN/tabular ensembling as a possible probability-level sidecar

Experiments completed or probed today:

- Class-2 hard override with proba-only LGBM: `0.7683247046665064`, lower than best.
- Class-2 hard override with proba + position context LGBM: `0.769042219552496`, still lower.
- Class-2 soft mix: best Viterbi around `0.7689310854054613`, lower.
- Class-5 hard override: held best exactly at `0.769308432188306`, no useful gain.
- Class-5 soft mix: best around `0.766840`, lower.
- Additive sidecar blend search:
  - Global calibrated unsmoothed reached `0.7652316954708785`
  - Fold-fair Viterbi dropped to `0.764044`, so not adopted.
- Viterbi class-weight tuning overfit training folds and dropped to `0.765835`.
- Aeon/shapelet/dictionary probes:
  - RocketClassifier fold 1 about `0.6024`, weak.
  - RDST fold 1 about `0.6215`, weak.
  - DrCIF/QUANT were too slow for practical iteration.
  - MrSQM/MrSEQL require system `fftw3.h`; no sudo available.
- GPU sequence CNN probes:
  - weighted sampler/no class weights fold 1 best `0.651856`
  - shuffle/balanced class weights fold 1 best `0.645988`
  - both are below MiniRocket fold 1 and were abandoned.
- GPU XGBoost probes:
  - position-shift depth4 calibrated `0.734536`
  - position depth5 calibrated `0.738871`
  - class-weighted class 2/5 depth5 calibrated `0.739620`
  - all below the current blend path.
- CRF sequence layer:
  - no-standardize components fold 1 `0.671343`
  - CE weight 1.0 fold 1 `0.679255`
  - below Viterbi fold 1 `0.7153`.
- Logistic stacker over component log-probas:
  - best calibrated `0.756658`
  - Viterbi `0.746208`
- Catch22 existing features were weak: LGBM `0.7028`, XGB `0.6969`.

New files/scripts from today's work:

- Added `scripts/build_spectral_wavelet_features.py`
- Built `artifacts/features_spectral_wavelet/train_features.csv` with shape `(11020, 1759)`
- Generated spectral/wavelet LGBM OOF:
  - `artifacts/model_search_spectral_wavelet_position/oof_position_lgbm_leaves63.npz`
  - base macro-F1 `0.7147525333861974`
  - calibrated macro-F1 `0.7417690533401137`
  - not useful as a primary model.

Stopped/unfinished before shutdown:

- Spectral/wavelet `lgbm_leaves47` and `xgb_gpu_depth5` run was stopped after
  the first spectral LGBM proved weak.
- Spectral/wavelet class-2 override was stopped after the first two outer folds
  produced no improvement over base Viterbi.
- A fixed-parameter Viterbi blend-weight search was started, then stopped for
  shutdown/logging; no result file was produced.

Recommended restart point for 2026-05-09:

1. Do not rerun the spectral/wavelet full model path unless using it only as a
   very small sidecar; its standalone OOF is too weak.
2. Continue from class-2-specific raw/window features rather than posthoc
   thresholds; thresholding has repeatedly failed fold-fair validation.
3. Consider a narrower targeted feature set for class 2:
   short-run burst shape, local peak timing, adjacent-window deltas, and
   user-normalized transition features.
4. If doing blend search, optimize the actual fold-fair Viterbi objective from
   the start, but keep candidate count modest and save progress incrementally.
5. Do not upload anything unless explicitly requested.

## 2026-05-09 daily log

Active goal: improve local OOF macro-F1 toward `0.8000`; do not upload to
Kaggle unless explicitly requested.

Work completed:

- Added `scripts/search_fixed_viterbi_blends.py` to search and materialize OOF
  blends under fixed fold-specific Viterbi parameters from an existing
  fold-fair smoothing run.
- Checked a class-2 per-user count postprocessor. It reduced some false
  positives but dropped macro-F1 to about `0.7627`, so it was not adopted.
- Ran a global Viterbi blend probe. The best global fixed candidate reached
  about `0.7731`, but strict `scripts/evaluate_sequence_smoothing.py`
  fold-fair retuning dropped candidates to `0.7636` or lower; do not treat that
  path as a promoted retuned smoother.
- Ran a fixed fold-parameter Viterbi blend search seeded from the previous
  centered-meta blend. The best materialized candidate is:
  - OOF macro-F1: `0.7739126451501365`
  - Artifact: `artifacts/fixed_viterbi_blend_local_best/summary.json`
  - Predictions: `artifacts/fixed_viterbi_blend_local_best/best_predictions.npz`
  - Report: `artifacts/fixed_viterbi_blend_local_best/best_report.csv`
  - Class `2` F1 improved from `0.3370` to `0.3521`
  - Class `5` F1 improved from `0.7395` to `0.7437`
  - Weights: `xgb=0.131057`, `cat=0.152170`, `xgb_d6=0`,
    `mini10=0.436832`, `mini20=0.029231`, `miniraw=0.119146`,
    `multi=0.024272`, `event_lgbm=0.019342`, `meta_lgbm=0.064478`,
    `meta_xgb=0.021916`, `lgbm47=0.001556`
  - Postprocess: first `10` windows of each user sequence locked to class `0`
- Checked the fixed candidate with the class-2 specialist override against
  event + position-shift features. The fold-fair override selected no effective
  changes and stayed at `0.7739126451501365`.
- Tested simple fold-fair threshold postprocessing:
  - Class-2 add threshold overfit: global `0.77433`, fold-fair `0.77270`; not adopted.
  - Class-5 add threshold overfit: global `0.77422`, fold-fair `0.77138`; not adopted.
  - Class-3 demotion by log-margin gave a small fold-fair improvement and was adopted:
    - OOF macro-F1: `0.7741720079574431`
    - Artifact: `artifacts/fixed_viterbi_blend_local_best_script_c3demote/summary.json`
    - Predictions: `artifacts/fixed_viterbi_blend_local_best_script_c3demote/best_predictions.npz`
    - Report: `artifacts/fixed_viterbi_blend_local_best_script_c3demote/best_report.csv`
    - Net changes from the fixed candidate: `4` windows
- Tested two sequence-template/duration ideas:
  - Linear stretched training-user label templates were very weak; even
    conservative probability-ratio mixing stayed below `0.767`.
  - A bounded duration-aware semi-Markov decoder also underperformed; the best
    quick setting observed was about `0.7678`, below the current fixed Viterbi
    screen. Do not spend more time on this without a faster implementation and
    a new hypothesis.
- Tested row-level LGBM as an additional sidecar. Its standalone calibrated OOF
  was only `0.67068`, and a fixed-Viterbi blend search assigned it weight `0`.
  It was not adopted.
- Started a class-2 ExtraTrees specialist using landmark/raw-shape features,
  but stopped it after outer fold 1 selected no override and the run was too
  slow for the weak inner-fold signal. Not adopted.
- Completed a lighter class-2 linear/raw-shape specialist:
  - Best variant: `logreg_c003_k300`
  - OOF macro-F1: `0.7743618566203608`
  - Artifact: `artifacts/class2_linear_shape_probe/logreg_c003_k300.npz`
  - It changed only two rows relative to the current fixed-Viterbi candidate.
- Added a fold-fair CatBoost class-3 high-confidence add rule:
  - OOF macro-F1: `0.7747143448705351`
  - Artifact: `artifacts/foldfair_component_threshold_probe/best_predictions.npz`
  - Rule: predicted label in `{1, 2, 5}` and CatBoost `p3` above the fold-tuned threshold.
- Combined the CatBoost class-3 rule with the class-2 linear override:
  - OOF macro-F1: `0.7749068514497729`
  - Artifact: `artifacts/local_best_oof_0774907/best_predictions.npz`
  - Report: `artifacts/local_best_oof_0774907/best_report.csv`
  - Net changes from the previous fixed-Viterbi candidate: `8` rows.
- Added two additional local sequence-edit rules on top of `0.7749068514497729`:
  - MiniRocket-10k class-2 long-run low-margin demotion:
    `0.7749068514497729 -> 0.7752419488386373`
  - XGBoost class-5 boundary extension from class-2 rows:
    `0.7752419488386373 -> 0.775950597872075`
  - Best current local OOF artifact: `artifacts/local_best_oof_0775951/best_predictions.npz`
  - Best report: `artifacts/local_best_oof_0775951/best_report.csv`
- Tested a class-5 linear/raw-shape specialist on the `0.77595` artifact:
  - Artifact: `artifacts/class5_linear_shape_probe_current/summary.json`
  - It selected no override in all five folds and left OOF unchanged at
    `0.775950597872075`; not adopted.
- Started user-level class2-to-class5 sequence threshold probes using per-user
  predicted class-2 counts, run length, position, and XGB/Cat/XGB-depth6
  class-5 scores. Both broad and narrowed versions were stopped because the
  threshold-heavy fold-fair scans were too slow for the likely marginal payoff;
  no promoted result was produced.
- Built and uploaded a test translation of the `0.7749068514497729` OOF
  candidate on user request:
  - CSV: `submissions/submission_local_best_oof0774907.csv`
  - Metadata: `artifacts/blend_search/test_blend_local_best_oof0774907.json`
  - Public score: `0.8068`
  - It did not beat `submissions/submission_centered_meta_viterbi_oof07693.csv`,
    which scored `0.8130` and is now the current public best.

Commands run:

- `make smoke`
- `python scripts/search_fixed_viterbi_blends.py --help`
- `python scripts/search_fixed_viterbi_blends.py --output-dir artifacts/fixed_viterbi_blend_local_best --trials 1 --first-lock 10 --center ...`
- `.venv/bin/python scripts/search_fixed_viterbi_blends.py --output-dir artifacts/fixed_viterbi_blend_local_best_script_c3demote --trials 1 --first-lock 10 --fold-fair-demote-class 3 --center ...`
- `.venv/bin/python scripts/evaluate_class2_override.py --blend-npz artifacts/blend_search/oof_blend_viterbi_fixedfold_top1.npz --viterbi-npz artifacts/fixed_viterbi_blend_local_best/best_predictions.npz --feature-cache artifacts/features_event/train_features_event.csv --context position_shift --output-dir artifacts/class2_override_fixed_viterbi_event_lgbm --model lgbm --feature-set proba_context --target-class 2 --positive-weight-factor 1.0 --n-jobs 6`
- `.venv/bin/python scripts/build_landmark_features.py --data-dir data/raw --split test --base-cache artifacts/features/test_features.csv --output artifacts/features/test_features_landmark.csv`
- `.venv/bin/kaggle competitions submit -c nycu-data-mining-assignment-3 -f submissions/submission_local_best_oof0774907.csv -m local_best_oof0774907`

Kaggle uploads/status:

- `submissions/submission_local_best_oof0774907.csv`: public `0.8068`.
- `submissions/submission_centered_meta_viterbi_oof07693.csv`: public `0.8130`;
  this is the current public best and was added to `SUBMISSION_LOG.md` after
  confirming it in the Kaggle submissions list.

Recommended next steps:

1. Do not prefer the `0.77491` local OOF test translation for public scoring;
   it underperformed the simpler centered-meta Viterbi submission.
2. Continue from the fixed-smoother search, but add progress checkpointing if
   running more than a few thousand trials.
3. The gap to `0.8000` remains large; the next high-risk/high-reward path is
   still better class-2 signal, not another broad model blend.

## 2026-05-09 continuation log

Current best local validation:

- Local OOF-screen macro-F1: `0.7895017478606237`
- Artifact: `artifacts/local_best_oof_0789502/summary.json`
- Prediction artifact: `artifacts/local_best_oof_0789502/best_predictions.npz`
- Report: `artifacts/local_best_oof_0789502/best_report.csv`
- This is still below the target `0.8000` OOF and has not been translated to a
  Kaggle submission.

Experiments completed:

- Added `scripts/probe_run_level_relabel.py`, a reproducible probe for
  fold-aware run-level relabeling of current predicted class-2 runs.
- The first implementation selected thresholds on only run rows and overfit;
  it was corrected to select thresholds on the full training folds' macro-F1.
- No-cache sequence/probability run features beat landmark-cache variants:
  the cache version reached only `0.7791666735718681`, while no-cache reached
  `0.77938784258011` before C/k sweeps.
- Logistic target-1 relabel sweep on predicted class-2 runs:
  - `C=0.35, k=80`: `0.77938784258011`
  - `C=0.10, k=80`: `0.7795928640072348`
  - `C=0.10, k=500`: `0.7801058454328823`
  - `C=0.05, k=500`: `0.7810215857553963`
  - `C=0.045, k=500`: `0.7813514765762658`
- Two additional fold-aware run-level passes on top of `C=0.045, k=500`:
  - Round 2 HGB multiclass relabel: `0.7813514765762658 -> 0.782438002483171`
  - Round 3 logistic multiclass relabel: `0.782438002483171 -> 0.7826865883289297`
  - Round 4 selected no changes and left OOF unchanged.
- Net change from `artifacts/local_best_oof_0777285/best_predictions.npz`:
  `74` rows. Class-2 F1 improved from `0.3634085213` to `0.3950276243`;
  macro-F1 improved by `0.0054013403`.
- Generalized `scripts/probe_run_level_relabel.py` with `--candidate-labels`
  and `--binary-targets`, then probed non-2 predicted runs for class-2 adds:
  - Round 1, candidates `{1, 3, 5}`, HGB multiclass:
    `0.7826865883289297 -> 0.7834643665691`
  - Round 2, candidates `{1, 3, 5}`, logistic multiclass:
    `0.7834643665691 -> 0.7843407629591118`
  - Round 3, candidates `{1, 3, 5}`, logistic binary target-2:
    `0.7843407629591118 -> 0.7845512651686737`
  - Round 4 selected no changes and left OOF unchanged.
- Final net change from `artifacts/local_best_oof_0777285/best_predictions.npz`:
  `107` rows. Class-2 F1 improved from `0.3634085213` to `0.3966942149`,
  class-5 F1 improved from `0.7475409836` to `0.7569892473`, and macro-F1
  improved by `0.0072660171`.
- Class-3 follow-up on top of `0.7845512651686737`:
  - HGB k80 class-3 add from candidates `{1, 2, 5}`:
    `0.7845512651686737 -> 0.7848026664603464`
  - HGB k80 predicted-class-3 cleanup:
    `0.7848026664603464 -> 0.7851241717532372 -> 0.7852998133927165`
  - All-activity k80 broad relabel selected no changes and stayed at
    `0.7852998133927165`.
- Latest net change from `artifacts/local_best_oof_0777285/best_predictions.npz`:
  `122` rows. Class-2 F1 improved from `0.3634085213` to `0.3972413793`,
  class-3 F1 improved from `0.7324364723` to `0.7382962394`, class-5 F1
  improved from `0.7475409836` to `0.7569892473`, and macro-F1 improved by
  `0.0080145654`.
- Added configurable run-level tuning controls to
  `scripts/probe_run_level_relabel.py`: `--binary-thresholds`,
  `--multiclass-thresholds`, `--multiclass-margins`,
  `--multiclass-reference-label`, and `--multiclass-target-labels`.
- Fine predicted-class-2 HGB multiclass cleanup on top of `0.7852998133927165`
  selected one fold-4 row (`file_id=9191`, `User_051`) from class `2` to class
  `1`, matching the OOF label:
  `0.7852998133927165 -> 0.7854101934817392`.
- Landmark-aggregate predicted-class-2 HGB target-1 cleanup on top of
  `0.7854101934817392` selected 18 rows from class `2` to class `1`:
  `0.7854101934817392 -> 0.786828068117277`. Of those changes, 14 were true
  class `1`, while 1/2/1 were true classes `2`/`3`/`5`.
- Landmark-aggregate predicted-class-3 HGB multiclass cleanup selected two
  additional positive rounds:
  `0.786828068117277 -> 0.7869641279465189 -> 0.7871920937380522`.
  A third round selected no improvement.
- Added `--tree-n-jobs`, HGB hyperparameter CLI controls, and optional
  `lgbm`/`cat` model support to `scripts/probe_run_level_relabel.py` for
  bounded run-level model sweeps.
- ExtraTrees predicted-class-3 landmark-aggregate cleanup selected two more
  changes: `0.7871920937380522 -> 0.7873855550270575`.
- HGB predicted-class-0 target-1 cleanup with landmark aggregates selected 67
  rows from class `0` to class `1`:
  `0.7873855550270575 -> 0.7894709553800107`.
- HGB predicted-class-3 target-1 cleanup after the class-0 pass selected 6 more
  rows to class `1`: `0.7894709553800107 -> 0.7895017478606237`.
- Latest net change from `artifacts/local_best_oof_0777285/best_predictions.npz`:
  `223` rows. Class-0 F1 improved from `0.9827456864` to `0.9887737478`,
  class-1 F1 improved from `0.9234828496` to `0.9308688388`, class-2 F1
  improved from `0.3634085213` to `0.4050991501`, class-3 F1 improved from
  `0.7324364723` to `0.7422360248`, and macro-F1 improved by `0.0122164998`.
- Negative follow-up probes:
  - Class-5 add from candidates `{1, 2, 3}` selected no changes and stayed at
    `0.7845512651686737`.
  - Final predicted-class-2 cleanup on top of `0.7845512651686737` selected no
    changes and stayed at `0.7845512651686737`.
  - Landmark-cache class-2 add from candidates `{1, 3, 5}` reached only
    `0.7833914425119705`, below the no-cache path.
  - Single-feature threshold search for `pred=1 -> 5` overfit training folds
    and dropped OOF to `0.7759889279256043`; not adopted.
  - Narrow `pred=1 -> 5` probes with no-cache, landmark/event cache, and
    ExtraTrees selected no changes or hurt validation; not adopted.
  - Latest event-cache class-2 add on top of `0.7852998133927165` selected no
    changes and stayed unchanged.
  - Fine low-threshold logreg sweeps for candidates `{1}`, `{2}`, and `{3}`
    selected no changes or hurt validation.
  - Fine HGB sweep for predicted class-3 cleanup with reference label `3`
    selected no improvement; class-3/class-4 exchange probes selected no
    changes.
  - ExtraTrees follow-ups for predicted class-1 and predicted class-2 cleanup
    selected no further improvement.
  - Event-cache predicted-class-2 cleanup on top of `0.7854101934817392`
    selected no changes; landmark-cache class-2 add-back selected hurtful
    changes only.
  - Landmark-aggregate HGB k-best sweep around the new target-1 cleanup showed
    `k=160` remains best: `k=80` reached `0.7857428318`, `k=120` reached
    `0.7858853122`, `k=180` reached `0.7864506202`, `k=200` reached
    `0.7864857412`, `k=240` reached `0.7866490829`, and `k=500` selected no
    improvement.
  - Fine binary threshold sweep around `0.70` reproduced `0.7868280681` but did
    not improve it.
  - Landmark-aggregate HGB `pred=1 -> 5` probe after `0.7868280681` selected
    no changes.
  - After `0.7871920937`, another landmark-aggregate predicted-class-2 cleanup
    selected no changes; target-3/5 variants hurt validation.
  - ExtraTrees/RF class-1 cleanup with landmark aggregates selected no changes.
  - ExtraTrees predicted-class-2 cleanup after `0.7873855550` gave only a
    negligible `+0.0000003`, so it was not promoted.
  - HGB predicted-class-0 target-1 second round selected no changes.
  - HGB k-best sweep for predicted-class-0 cleanup did not beat `k=160`; nearby
    variants reached `0.7888156472`, `0.7889936713`, `0.7890340859`, and
    `0.7890580292`.
  - HGB leaf/l2/iteration variants reproduced the same `0.7894709554` best but
    did not improve it.
  - After `0.7895017479`, predicted-class-5 cleanup and class-4-to-3 cleanup
    selected no changes.
  - LightGBM and CatBoost landmark-aggregate run-level probes for candidates
    `{1}`, `{2}`, and `{3}` selected no changes or hurt validation.
  - HGB class-1 cleanup to targets `{0, 3, 4}` selected no improvement.
  - HGB class-3 cleanup to targets `{0, 4}` selected no improvement.

Commands run:

- `.venv/bin/python -m py_compile scripts/probe_run_level_relabel.py`
- `make smoke`
- `.venv/bin/python scripts/probe_run_level_relabel.py --no-feature-cache --output-dir artifacts/run_level_rich_relabel_probe_nocache --k-best 80 --models logreg,hgb`
- `.venv/bin/python scripts/probe_run_level_relabel.py --output-dir artifacts/run_level_rich_relabel_probe --k-best 60 --models logreg,hgb`
- `.venv/bin/python scripts/probe_run_level_relabel.py --no-feature-cache --output-dir artifacts/run_level_target1_sweep/c0045_k500 --k-best 500 --models logreg --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_target1_sweep/c0045_k500/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --no-feature-cache --output-dir artifacts/run_level_target1_sweep/c0045_k500_round2 --k-best 500 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_target1_sweep/c0045_k500_round2/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --no-feature-cache --output-dir artifacts/run_level_target1_sweep/c0045_k500_round3 --k-best 500 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0782687/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 1,3,5 --binary-targets 2 --no-feature-cache --output-dir artifacts/run_level_add_class2_probe_nocache --k-best 500 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_add_class2_probe_nocache/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 1,3,5 --binary-targets 2 --no-feature-cache --output-dir artifacts/run_level_add_class2_probe_round2_nocache --k-best 500 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_add_class2_probe_round2_nocache/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 1,3,5 --binary-targets 2 --no-feature-cache --output-dir artifacts/run_level_add_class2_probe_round3_nocache --k-best 500 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0784551/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 1,2,3 --binary-targets 5 --no-feature-cache --output-dir artifacts/run_level_add_class5_probe_nocache --k-best 500 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0784551/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 1,2,5 --binary-targets 3 --no-feature-cache --output-dir artifacts/run_level_add_class3_probe_hgb_k80 --k-best 80 --models hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_add_class3_probe_hgb_k80/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 3 --binary-targets 1,2,5 --no-feature-cache --output-dir artifacts/run_level_pred3_cleanup_probe_hgb_k80 --k-best 80 --models hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred3_cleanup_probe_hgb_k80/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 3 --binary-targets 1,2,5 --no-feature-cache --output-dir artifacts/run_level_pred3_cleanup_probe_hgb_k80_round2 --k-best 80 --models hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0785300/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 1 --binary-targets 5 --no-feature-cache --output-dir artifacts/run_level_pred1_to5_probe_k80 --k-best 80 --models logreg,hgb --logreg-c 0.045`
- `.venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0785300/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --feature-cache artifacts/features_event/train_features_event.csv --candidate-labels 1,3,5 --binary-targets 2 --output-dir artifacts/run_level_add_class2_probe_event_k80_latest --k-best 80 --models logreg,hgb --logreg-c 0.045`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0785300/best_predictions.npz --proba-npz artifacts/local_best_oof_0775951/best_predictions.npz --component-npz artifacts/blend_search/oof_blend_centered_meta_round2_best.npz --candidate-labels 2 --binary-targets 1,3,5 --no-feature-cache --output-dir artifacts/run_level_pred2_cleanup_fine_hgb_current --k-best 500 --models hgb --logreg-c 0.045 --binary-thresholds ... --multiclass-thresholds ... --multiclass-margins=...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred2_cleanup_fine_hgb_current/best_predictions.npz --candidate-labels 1 --binary-targets 2,5 --no-feature-cache --output-dir artifacts/run_level_pred1_cleanup_fine_hgb_after_pred2 --k-best 500 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred2_cleanup_fine_hgb_current/best_predictions.npz --candidate-labels 4 --binary-targets 3 --no-feature-cache --output-dir artifacts/run_level_pred4_to3_hgb_after_pred2 --k-best 80 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred2_cleanup_fine_hgb_current/best_predictions.npz --candidate-labels 3 --binary-targets 4 --no-feature-cache --output-dir artifacts/run_level_pred3_to4_hgb_after_pred2 --k-best 80 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred2_cleanup_fine_hgb_current/best_predictions.npz --candidate-labels 1 --binary-targets 2,5 --no-feature-cache --output-dir artifacts/run_level_pred1_cleanup_fine_extra_after_pred2 --k-best 500 --models extra ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred2_cleanup_fine_hgb_current/best_predictions.npz --candidate-labels 2 --binary-targets 1,3,5 --no-feature-cache --output-dir artifacts/run_level_pred2_cleanup_fine_extra_after_pred2 --k-best 500 --models extra ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0785410/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 2 --binary-targets 1,3,5 --output-dir artifacts/run_level_pred2_cleanup_landmark_aggs_hgb_after_0785410 --k-best 160 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred2_cleanup_landmark_aggs_hgb_after_0785410/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 2 --binary-targets 1,3,5 --output-dir artifacts/run_level_pred2_cleanup_landmark_aggs_hgb_round2_after_0786828 --k-best 160 --models hgb ...`
- `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0785410/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 2 --binary-targets 1 --output-dir artifacts/run_level_pred2_cleanup_landmark_aggs_hgb_k{80,120,180,200,240,500}_after_0785410 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0786828/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 3 --binary-targets 1,2,4,5 --output-dir artifacts/run_level_pred3_cleanup_landmark_aggs_hgb_after_0786828 --k-best 160 --models hgb --multiclass-reference-label 3 --multiclass-target-labels 1,2,5 ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred3_cleanup_landmark_aggs_hgb_after_0786828/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 3 --binary-targets 1,2,4,5 --output-dir artifacts/run_level_pred3_cleanup_landmark_aggs_hgb_round2_after_0786964 --k-best 160 --models hgb --multiclass-reference-label 3 --multiclass-target-labels 1,2,5 ...`
- `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0787192/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 3 --binary-targets 1,2,4,5 --output-dir artifacts/run_level_pred3_cleanup_landmark_aggs_extra_after_0787192 --k-best 160 --models extra --tree-n-jobs 4 --multiclass-reference-label 3 --multiclass-target-labels 1,2,5 ...`
- `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred3_cleanup_landmark_aggs_extra_after_0787192/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 0 --binary-targets 1 --output-dir artifacts/run_level_pred0_to1_landmark_aggs_hgb_after_0787386 --k-best 160 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/run_level_pred0_to1_landmark_aggs_hgb_after_0787386/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 3 --binary-targets 1,2,4,5 --output-dir artifacts/run_level_pred3_cleanup_landmark_aggs_hgb_after_0789471 --k-best 160 --models hgb --multiclass-reference-label 3 --multiclass-target-labels 1,2,5 ...`
- `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0789502/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels {1,2,3} --models lgbm ...`
- `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0789502/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels {1,2,3} --models cat ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0789502/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 1 --binary-targets 0,3,4 --output-dir artifacts/run_level_pred1_cleanup_0_3_4_landmark_aggs_hgb_after_0789502 --k-best 160 --models hgb ...`
- `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/probe_run_level_relabel.py --base-npz artifacts/local_best_oof_0789502/best_predictions.npz --feature-cache artifacts/features/train_features_landmark.csv --cache-aggs mean,std,min,max,first,last --candidate-labels 3 --binary-targets 0,4 --output-dir artifacts/run_level_pred3_to0_4_landmark_aggs_hgb_after_0789502 --k-best 160 --models hgb ...`

## 2026-05-09 proba-threshold continuation

Current best local validation:

- Local OOF-screen macro-F1: `0.8027842809283494`
- Artifact: `artifacts/local_best_oof_0802784/summary.json`
- Prediction artifact: `artifacts/local_best_oof_0802784/best_predictions.npz`
- Source run: `artifacts/proba_threshold_all_pairs_all_sources_after_0797693/summary.json`
- This reaches the `0.8000` local OOF target, but it is an aggressive
  OOF-screen over many probability sources and all class transitions. Treat it
  as a hypothesis generator until a matching test-time implementation is built
  and checked on Kaggle.

Work completed:

- Added `scripts/probe_proba_threshold_relabel.py`, a fold-aware threshold
  relabel probe for aligned OOF probability sources.
- Row-level binary logreg relabeling improved the previous `0.7895017479`
  artifact to `0.7903553930`, then a one-row HGB correction improved it to
  `0.7906131020`.
- Curated probability-threshold passes promoted local OOF:
  `0.7906131020 -> 0.7946590413 -> 0.7954557637 -> 0.7970847129 -> 0.7972256718`.
- An all-source pass added two small rules and promoted:
  `0.7972256718 -> 0.7976934416`.
- A second all-source pass over the curated transition list selected no further
  improvement.
- Expanding to all 30 class transitions found a large local OOF-screen gain:
  `0.7976934416 -> 0.8027842809` with `60` changed rows.
- Final selected threshold rules:
  - `blend_search/oof_blend_centered_meta_round2_best.npz:mini10_proba:p_to:4->0`
  - `blend_search/oof_blend_centered_meta_additive_sidecars_best.npz:proba:margin_other:2->0`
  - `model_search_position_shift/oof_position_shift_lgbm_base.npz:proba:margin_other:2->0`
  - `model_search_position_cw_2x5/oof_position_lgbm_leaves63.npz:proba:margin_other:2->0`
  - `model_search_position_cat_depth8/oof_position_catboost_depth8.npz:proba:margin_other:4->2`
  - `blend_search/oof_blend_centered_meta_with_row_lgbm_source.npz:row_lgbm_proba:margin_from:3->0`
  - `model_search_temporal_v2_position/oof_position_xgb_base.npz:proba:p_to:2->0`
  - `catch22_oof/oof_catch22_raw_lgbm_c22.npz:proba:p_to:2->0`
  - `class5_softmix_position_lgbm/oof_softmix_lam0.075.npz:proba:margin_other:2->0`
  - `model_search_position_proba/oof_position_lgbm_leaves63.npz:proba:log_margin_from:2->0`
- Final local class F1s: class `0` `0.9825`, class `1` `0.9319`,
  class `2` `0.4372`, class `3` `0.7677`, class `4` `0.9254`, class `5`
  `0.7721`.
- A user-requested test translation was uploaded after this local screen:
  - CSV: `submissions/submission_local_oof0802784_threshold_probe.csv`
  - Metadata: `artifacts/blend_search/test_local_oof0802784_threshold_probe.json`
  - Translation caveat: this was a leaderboard probe, not an exact test
    implementation of all OOF-screen rules. It used median fold thresholds,
    full-train/approximate test counterparts for available sources, and skipped
    row-LGBM, temporal-v2 XGB, and Catch22 rules.
  - Applied `7` rules, changed `45` rows versus
    `submissions/submission_local_best_oof0774907.csv`, and produced label
    counts `{0: 2872, 1: 3003, 2: 218, 3: 449, 4: 70, 5: 237}`.
  - Public score: `0.8067`, below both the prior local upload `0.8068` and the
    current best centered-meta Viterbi public score `0.8130`.

Commands run:

- `.venv/bin/python -m py_compile scripts/probe_proba_threshold_relabel.py scripts/probe_run_level_relabel.py`
- `make smoke`
- `.venv/bin/python scripts/probe_proba_threshold_relabel.py --base-npz artifacts/local_best_oof_0797226/best_predictions.npz --output-dir artifacts/proba_threshold_all_sources_after_0797226 --all-sources`
- `.venv/bin/python scripts/probe_proba_threshold_relabel.py --base-npz artifacts/local_best_oof_0797693/best_predictions.npz --output-dir artifacts/proba_threshold_all_sources_after_0797693 --all-sources`
- `.venv/bin/python scripts/probe_proba_threshold_relabel.py --base-npz artifacts/local_best_oof_0797693/best_predictions.npz --output-dir artifacts/proba_threshold_all_pairs_all_sources_after_0797693 --all-sources --transitions 0:1,0:2,0:3,0:4,0:5,1:0,1:2,1:3,1:4,1:5,2:0,2:1,2:3,2:4,2:5,3:0,3:1,3:2,3:4,3:5,4:0,4:1,4:2,4:3,4:5,5:0,5:1,5:2,5:3,5:4`
- `.venv/bin/python scripts/make_threshold_relabel_submission.py --skip-row-lgbm --output submissions/submission_local_oof0802784_threshold_probe.csv --metadata artifacts/blend_search/test_local_oof0802784_threshold_probe.json --save-npz artifacts/blend_search/test_local_oof0802784_threshold_probe.npz`
- `.venv/bin/kaggle competitions submit -c nycu-data-mining-assignment-3 -f submissions/submission_local_oof0802784_threshold_probe.csv -m local_oof0802784_threshold_probe_skiprow`

## 2026-05-09 May 10 upload plan

Objective: prepare three candidates for the next daily submission window, with
automatic upload scheduled for `2026-05-10 08:10 CST`.

Reasoning update:

- The best public submission remains `submission_centered_meta_viterbi_oof07693.csv`
  at public `0.8130`.
- Broad OOF relabel screens are now considered unreliable for public scoring:
  the local `0.802784` screen translated to public `0.8067`.
- Consensus analysis on public-strong models showed that naive majority voting
  among centered-meta Viterbi, LGBM, and MiniRocket hurts OOF.
- A narrow rare-class add-back rule was the only positive OOF signal:
  when LGBM and MiniRocket agree on a rare-class target and the centered-meta
  Viterbi base disagrees, apply only selected target/source subsets.
- Re-ran a restricted threshold screen starting from the current public best
  instead of the broader `0.802784` local screen. The selected five-rule probe
  improved centered-meta Viterbi OOF by about `+0.005`, with only `20` test
  changes after median-threshold translation. Because threshold screens have
  overfit before, these are kept as adaptive high-upside branches rather than
  the first upload.
- A finer OOF search over only the from-to groups where LGBM and MiniRocket
  agree found a cleaner candidate than the earlier rare-target bundles:
  groups `{5->3, 2->5, 1->2, 3->1, 4->5, 1->5, 3->4}` give OOF
  delta `+0.0053380` with all test changes supported by both public-strong
  side models. This is now the first scheduled upload.

Prepared candidate pool:

| candidate | CSV | OOF delta vs centered-meta Viterbi | test changes vs base | label counts | SHA256 |
|---|---|---:|---:|---|---|
| consensus rare B | `submissions/submission_consensus_rare_b.csv` | +0.0023497 | 18 | `{0: 2807, 1: 3026, 2: 222, 3: 463, 4: 79, 5: 252}` | `213b4b5e81411e8ee4bb89c3ce10c49df02b17716f386db8f2d57d2e0bafed85` |
| consensus rare B+C | `submissions/submission_consensus_rare_bplusc.csv` | +0.0034055 | 19 | `{0: 2807, 1: 3026, 2: 221, 3: 463, 4: 79, 5: 253}` | `05c441ee4f42d62f9c33435e87238508074807502026736bc3380dbbc2ba9bc4` |
| consensus rare C | `submissions/submission_consensus_rare_c.csv` | +0.0010687 | 1 | `{0: 2810, 1: 3034, 2: 221, 3: 449, 4: 82, 5: 253}` | `481693245b7dab468cd5df10ef91a4d56925fcd7c3d4160229e192267216fc0c` |
| consensus rare F | `submissions/submission_consensus_rare_f.csv` | +0.0016365 | 14 | `{0: 2807, 1: 3029, 2: 222, 3: 463, 4: 80, 5: 248}` | `2f2c5a15242b16f3d284787e12a67a049c425b5fe704d79b5617183fef1c290c` |
| consensus rare G | `submissions/submission_consensus_rare_g.csv` | +0.0007266 | 4 | `{0: 2810, 1: 3031, 2: 222, 3: 449, 4: 81, 5: 256}` | `c5a999790f328640c8563bea551b46faac9137f7338d8a103c8fc051eda80dd3` |
| consensus rare A+C | `submissions/submission_consensus_rare_aplusc.csv` | +0.0035471 | 40 | `{0: 2807, 1: 3013, 2: 239, 3: 461, 4: 82, 5: 247}` | `3153b8d1ea4718ba04cce61fd3f0954bcd908d8db4f97a683b6d5192808ad1e4` |
| consensus rare A | `submissions/submission_consensus_rare_a.csv` | +0.0024973 | 39 | `{0: 2807, 1: 3013, 2: 240, 3: 461, 4: 82, 5: 246}` | `225905ce8d5b98df3a719137b138a7587b1e9266f465e7515b3752cac856b474` |
| consensus OOF-greedy | `submissions/submission_consensus_oof_greedy.csv` | +0.0053380 | 24 | `{0: 2810, 1: 3020, 2: 234, 3: 451, 4: 81, 5: 253}` | `aee96eebedc4dfbfb2813bf17b313cf2e430cf9b2a6335e80838e1824aa959bf` |
| consensus OOF-greedy + threshold3 | `submissions/submission_consensus_oof_greedy_threshold3.csv` | +0.0077004 | 36 | `{0: 2810, 1: 3020, 2: 247, 3: 440, 4: 80, 5: 252}` | `a5a0c765d6c6e69f737b1158750f1773b4a640d5cf8b6954da6b0056f95ca22f` |
| consensus OOF-alt + threshold4 | `submissions/submission_consensus_oof_alt_threshold4.csv` | +0.0074513 | 32 | `{0: 2810, 1: 3020, 2: 247, 3: 436, 4: 80, 5: 256}` | `2ac3ec94049013801a3460222518189c8a35a0c7944092488c66c4ed47ad5e45` |
| public-best threshold | `submissions/submission_public_best_threshold.csv` | +0.0050261 | 20 | `{0: 2810, 1: 3028, 2: 242, 3: 437, 4: 80, 5: 252}` | `623845e43a17a35b32f690778e34a420668d168f778f9296ea7e6de98047174c` |
| consensus rare C + threshold | `submissions/submission_consensus_rare_c_threshold.csv` | +0.0060955 | 21 | `{0: 2810, 1: 3028, 2: 241, 3: 437, 4: 80, 5: 253}` | `cb77f3c1e25dd7ceece2f4a7fb56781b521eb8812c7f63d1ba003c8555404e88` |
| consensus rare B+C + threshold | `submissions/submission_consensus_rare_bplusc_threshold.csv` | +0.0054358 | 38 | `{0: 2807, 1: 3020, 2: 241, 3: 450, 4: 78, 5: 253}` | `cd8a486b30ff695c69b774848c3e9f8b73c05d8ec377d997c6d5696b5468a4ce` |

Generated files/scripts:

- `scripts/make_consensus_candidates.py`
- `scripts/upload_may10_0810_candidates.sh`
- `scripts/audit_may10_upload.py`
- `scripts/audit_may10_0840_status.sh`
- `artifacts/blend_search/consensus_candidates_may10.json`

Scheduled adaptive upload:

- Current crontab entry:
  `10 8 10 5 * /bin/bash /home/raiso/playground/DM2026-Assignment-3/scripts/upload_may10_0810_candidates.sh # DM2026-May10-0810`
- Independent status audit crontab entry:
  `40 8 10 5 * /bin/bash /home/raiso/playground/DM2026-Assignment-3/scripts/audit_may10_0840_status.sh # DM2026-May10-0840-Audit`
- Timing note: the upload is scheduled for `2026-05-10 08:10 CST`, which is
  `2026-05-10 00:10 UTC`, shortly after UTC date rollover.
- Upload decision tree:
  1. Submit `submission_consensus_oof_greedy.csv` first.
  2. Always submit `submission_consensus_oof_greedy_threshold3.csv` second
     because it has the highest controlled OOF delta (`+0.0077004`) and is the
     most plausible path to public `>0.82`.
  3. If the threshold branch matches/beats `0.8130`, submit
     `submission_consensus_oof_alt_threshold4.csv`, an adjacent high-OOF
     threshold-family candidate with fewer test changes and more class-5 mass.
  4. If the first greedy consensus branch matches/beats `0.8130` but threshold3
     does not, submit `submission_consensus_rare_b.csv`.
  5. If both remain below current best, still use
     `submission_consensus_oof_alt_threshold4.csv` when threshold3 strictly
     beats the first candidate; otherwise use
     `submission_consensus_rare_b.csv`.
- Upload log will be written to
  `artifacts/blend_search/upload_may10_0810.log`.
- Upload-script reliability fixes:
  - Candidate CSVs are validated against `sample_submission.csv` before any
    upload attempt.
  - `HOME` and `KAGGLE_CONFIG_DIR` are set explicitly for cron so Kaggle
    credentials resolve the same way as in the interactive shell.
  - Each Kaggle submit command retries up to three times before failing.
  - Kaggle status polling now accepts `SubmissionStatus.COMPLETE` as well as
    plain `complete`.
  - Status polling now matches both submitted filename and Kaggle message, so
    an older same-filename submission cannot drive the adaptive branch choice.
  - The final status listing is written to files before `head`, avoiding
    `pipefail`/broken-pipe failures.
  - An `EXIT` trap removes the one-shot cron entry and logs the exit status.
  - `DM2026_DRY_RUN=1` simulates branch choices without submitting, writes to
    `upload_may10_0810_dryrun_submitted.tsv`, and leaves the real manifest
    untouched. Dry-run also skips cron cleanup automatically, even if
    `DM2026_SKIP_CRON_CLEANUP=1` is not set.
  - `scripts/audit_may10_upload.py` writes
    `artifacts/blend_search/upload_may10_0810_audit.json`; the audit requires
    exactly three completed scheduled submissions and best public score `>0.82`.
  - On non-zero exit after at least one manifest row is written, cleanup tries
    to snapshot Kaggle status and run the audit before removing the cron entry.
  - A `flock` lock at `artifacts/blend_search/upload_may10_0810.lock` prevents
    simultaneous duplicate executions from consuming extra Kaggle quota.
    The submitted manifest is truncated only after the lock is acquired, so a
    duplicate process cannot erase the active run's manifest.
  - In real mode, the upload script now refuses to rerun if
    `artifacts/blend_search/upload_may10_0810_submitted.tsv` is already
    non-empty, preventing a manual recovery attempt from clearing partial-run
    evidence and re-submitting duplicates.
  - The fallback branch now keeps the third upload high-upside when threshold3
    strictly beats the first greedy candidate, using
    `submission_consensus_oof_alt_threshold4.csv`; exact ties below the current
    best use the safer `submission_consensus_rare_b.csv`.
  - A separate 08:40 status-only cron refreshes Kaggle status and runs the
    audit verifier without consuming submission quota.
    The 08:40 audit script also has an `EXIT` trap so its one-shot cron entry is
    removed even if the status refresh fails; `DM2026_SKIP_CRON_CLEANUP=1`
    disables that cleanup for manual testing.

Verification:

- `.venv/bin/python scripts/make_consensus_candidates.py`
- `bash -n scripts/upload_may10_0810_candidates.sh`
- `.venv/bin/python -m py_compile scripts/make_consensus_candidates.py`
- `.venv/bin/python -m py_compile scripts/audit_may10_upload.py`
- `make smoke`
- Manual CSV validation against `data/raw/sample_submission.csv`
- Dry-run branch checks:
  - threshold3 >= current best -> third upload `submission_consensus_oof_alt_threshold4.csv`
  - greedy >= current best but threshold3 < current best -> third upload `submission_consensus_rare_b.csv`
  - both below current best and threshold3 strictly > greedy -> third upload `submission_consensus_oof_alt_threshold4.csv`
  - both below current best and threshold3 <= greedy -> third upload `submission_consensus_rare_b.csv`
- `scripts/audit_may10_upload.py` fake-status checks:
  - target-met case with best public `0.8210` exits successfully.
  - target-missed case with best public `0.8199` exits non-zero.
- Lock contention check: a second dry-run exits while preserving the existing
  dry-run manifest.
- `bash -n scripts/audit_may10_0840_status.sh`
- Dry-run without `DM2026_SKIP_CRON_CLEANUP=1` was checked to leave crontab
  unchanged.

The goal is not complete until the scheduled uploads run and Kaggle reports the
scores.
