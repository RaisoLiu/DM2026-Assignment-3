# DM2026 Assignment 3 Report: Human Activity Recognition

Student ID: 413551030

Public GitHub repository: https://github.com/RaisoLiu/DM2026-Assignment-3

Kaggle public score: 0.8106

## How to Run the Code

Install dependencies and place the Kaggle data under `data/raw/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
kaggle competitions download -c nycu-data-mining-assignment-3 -p data/raw --force
unzip -q -o data/raw/nycu-data-mining-assignment-3.zip -d data/raw
```

The downloaded archive originally contains `train/train/User_*` and
`test/test/User_*`. I moved these one level up so the final layout is:

```text
data/raw/train/User_001/*.csv ... data/raw/train/User_060/*.csv
data/raw/test/User_061/*.csv  ... data/raw/test/User_100/*.csv
data/raw/sample_submission.csv
```

Reproduce the final selected Kaggle CSV:

```bash
python scripts/make_boosting_submission.py \
  --results-csv artifacts/model_search_position_proba/results_position.csv \
  --model lgbm_leaves63 \
  --context position \
  --output submissions/submission_lgbm_leaves63_calibrated.csv \
  --seed 2026
```

The selected submission file is `submissions/submission_lgbm_leaves63_calibrated.csv`.

## 1. Preliminary Analysis and Design Motivation

The training set contains 11,020 five-minute windows from 60 users. The test set
contains 6,849 windows from 40 disjoint users. Each file has 300 one-second rows
with `mean_x`, `mean_y`, `mean_z`, `std_x`, `std_y`, `std_z`, `index`, and
`file_id`; training files also contain one repeated file-level `label`.

The task is strongly imbalanced:

| label | count | fraction |
|---:|---:|---:|
| 0 | 4,643 | 42.13% |
| 1 | 4,695 | 42.60% |
| 2 | 358 | 3.25% |
| 3 | 656 | 5.95% |
| 4 | 142 | 1.29% |
| 5 | 526 | 4.77% |

Because the official metric is macro-F1, accuracy is not sufficient. A model
can score high accuracy by predicting labels 0 and 1, but it will lose macro-F1
on labels 2, 3, 4, and 5. I therefore used user-grouped cross-validation and
class-balanced training weights. The validation split is grouped by `User_*`
because the public/private test users are not present in training; a random
window split would leak user-specific orientation and wearing-style information.

The accelerometer retains the gravity component. This makes vector magnitude
and orientation informative: activities can differ by both motion intensity and
wrist posture. The preliminary feature check showed that the average mean-vector
magnitude is close to 1g (`mean_vm_mean` median 0.997), while dynamic variability
(`std_vm_mean`) ranges widely. This motivated a file-level feature representation
combining static gravity-aware features and temporal variability features.

## 2. Preprocessing Techniques and Measured Effects

Each CSV is sorted by `index`, interpolated within the six signal channels, and
converted into one feature vector. I do not train on individual seconds as
independent samples because the label belongs to the whole five-minute window.

The base feature table has 399 engineered features per window. The final
LightGBM submission adds six compact user-sequence position features, giving
405 features:

| feature family | purpose |
|---|---|
| Global channel statistics | Quantiles, RMS, range, skewness, and kurtosis for all six channels. |
| Gravity/magnitude | Mean-vector magnitude, std-vector magnitude, deviation from 1g, and dynamic/static energy ratio. |
| Temporal derivatives | First-difference volatility, energy, absolute changes, and zero-crossing counts. |
| Segment/rolling | Five one-minute segment summaries plus 5/15/30-second rolling variability. |
| Frequency | FFT power, dominant frequency, entropy, centroid, and band ratios at 1 Hz. |
| Cross-axis/orientation | Axis-normalized orientation summaries, correlations, and covariance. |
| User sequence position | Normalized order, reverse order, edge distance, sine/cosine phase, and sequence length. |

The most important preprocessing choice was keeping the unit of prediction at
the file level. Feature extraction and final submission both align by `file_id`;
the submission is merged back to `sample_submission.csv` by `Id`.

Measured 5-fold grouped CV effects:

| design | macro-F1 | interpretation |
|---|---:|---|
| HGB, all features | 0.72921 | Main nonlinear model before decision calibration. |
| HGB + OOF class-bias calibration | 0.73823 | Stable baseline for ablation. |
| Remove gravity/magnitude features | 0.71548 | Largest feature-family loss; gravity-aware magnitude is useful. |
| Remove temporal derivatives | 0.72808 | Small loss; derivatives add modest but consistent information. |
| Remove segment/rolling features | 0.73131 | Slight gain before calibration, likely due reduced fold-specific noise. |
| Remove frequency features | 0.72923 | Approximately neutral at 1 Hz resolution. |
| Remove orientation/cross-axis features | 0.73168 | Slight gain before calibration, suggesting mild fold-specific noise. |

The full 450-iteration HGB had a calibrated OOF score of 0.74751, but its
calibration pushed the test prediction fraction for label 2 to 15.56%, compared
with 3.25% in the training set. I kept this as an alternative artifact but did
not choose it as the final submission because the predicted prior shift was too
large for a private-leaderboard setting. The final LightGBM submission keeps the
test label distribution closer to the training prior: label 2 is 2.76%, label 4
is 1.26%, and labels 0/1 remain dominant.

## 3. Label Alignment and Temporal Dependency Modeling

Each 300-row file corresponds to exactly one activity label. My pipeline
therefore uses this alignment:

1. Read one CSV file and sort rows by `index`.
2. Infer the single `file_id` and, for training, the single label.
3. Extract a single window-level feature vector from all 300 seconds.
4. Train and validate on one row per file.
5. Predict one label per test `file_id`.
6. Merge predictions to `sample_submission.csv` in the original `Id` order.

Temporal dependencies are represented through deterministic sequence summaries.
First differences capture second-to-second motion changes. Segment statistics
preserve coarse order over the five-minute interval. Rolling statistics capture
local bursts or sustained variability. FFT summaries capture periodicity at the
available 1 Hz resolution. This design is simpler and more reproducible than a
deep sequence model, while still respecting the temporal nature of the data.

## 4. Core Model, Ablation, and Final Choice

The final selected model is `LGBMClassifier` with `num_leaves=63`, 900
estimators, balanced sample weights, and the 405-feature position-augmented
window representation. The final prediction applies out-of-fold class-bias
calibration to optimize macro-F1:

| class | calibration weight |
|---:|---:|
| 0 | 0.41647 |
| 1 | 0.45553 |
| 2 | 5.85992 |
| 3 | 0.89605 |
| 4 | 1.31184 |
| 5 | 0.76524 |

This calibration increased grouped OOF macro-F1 from 0.72509 to 0.74771. The
main gain was improved recall for label 2, a rare class that otherwise tends to
be absorbed into labels 1 and 3.

Calibrated 5-fold grouped CV report:

| label | precision | recall | F1 |
|---:|---:|---:|---:|
| 0 | 0.97537 | 0.98083 | 0.97809 |
| 1 | 0.90484 | 0.91949 | 0.91211 |
| 2 | 0.28536 | 0.32123 | 0.30223 |
| 3 | 0.68714 | 0.73323 | 0.70944 |
| 4 | 0.92537 | 0.87324 | 0.89855 |
| 5 | 0.86881 | 0.56654 | 0.68585 |
| macro avg | 0.77448 | 0.73243 | 0.74771 |

The main residual error is label 2, which is often confused with labels 1 and
3. I treat this as the highest-priority future improvement because the class is
rare but macro-F1 weights it equally. The selected model intentionally uses a
calibrated LightGBM decision rule because it gave the best public score while
keeping rare-class test priors plausible.

## Reproducibility Notes

All reported numbers were generated with seed 2026 and grouped CV by user. The
feature cache files are written to `artifacts/features/` to avoid repeated CSV
parsing. The selected submission checksum is:

```text
f78034a7e72d25ab4d52baebed585e130ad810e1f5cfae7469205008d5d5035e  submissions/submission_lgbm_leaves63_calibrated.csv
```

Kaggle submissions and scores are tracked in `SUBMISSION_LOG.md`. The current
best public score is 0.8106 from the LightGBM leaves63 submission.
