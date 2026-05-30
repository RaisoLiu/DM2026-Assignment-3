# Strategy and Improvement Report: 0.85+ Attempt #2

**Author**: DM2026 Assignment 3 pipeline (PI-style review draft)
**Date**: 2026-05-12 (planning) → 2026-05-13 08:10 CST (upload)
**Public best to beat**: 0.8240 (`submission_ssl_hybrid_recover.csv`)
**Today's failure**: 0.7963 (`submission_ensemble_v2.csv`) — a −0.028 regression that wasted the 08:10 slot
**Stretch goal**: > 0.85 public macro-F1

---

## 1. Failure post-mortem (2026-05-12 08:10 upload, public 0.7963)

The v2 pipeline that I optimistically projected as the "0.85 path" landed −0.028 below yesterday's 0.8240 anchor. Three root causes, each separately verifiable:

1. **Meta-learner overfit OOF**. The LGBM stacker over 4 architectures × 6 classes + agreement features hit **TRAIN macro-F1 = 1.0** — it memorized the 52-user OOF prediction structure perfectly. On test (40 disjoint users), those patterns vanished.
2. **Pseudo-labeling at 60% volume injected majority-class bias**. The 6,690 consensus pseudo-labels (vs 11,020 real) carried yesterday's CSV-implied class-2 deficit forward; new models systematically under-predicted class-2 on test (final count 149 vs anchor 222).
3. **Hybrid recovery 2.0 over-tuned**. Learned switching thresholds on the OOF disagreement distribution that doesn't generalize to test-user disagreement patterns.

The pre-registered HOS gate (which I disabled per user's "all-in" instruction) would have likely flagged the regression — fresh HOS evaluation is the only metric that can detect OOF→public distribution shift. **Skipping the gate cost us 0.028 public.**

## 2. What worked yesterday (0.8240) — surgical, not sophisticated

The 0.8156→0.8240 lift came from a 30-row manual hybrid: take the new SSL pipeline's predictions, then for each row where the baseline predicted class-2 or class-5 but the new pipeline predicted otherwise, restore the baseline's rare-class label only if the new pipeline's *soft* proba for that class was still ≥ 0.27. Confidence-gated, low-volume, no learned parameters. It generalized because it didn't fit any test-user-specific structure.

The takeaway is uncomfortable: at 0.82+, sophisticated end-to-end pipelines overfit, while surgical row-level corrections compound. Tomorrow's plan returns to this philosophy with risk in different axes (new feature space, zero-parameter test-time adaptation), avoiding the specific failure modes of today.

## 3. Selected pipeline for 2026-05-13

**4-source weighted ensemble + Viterbi smoothing + extended hybrid recovery 3.0, on a fresh 12-user HOS, no test-time adaptation (centering disabled after HOS diagnostic showed it hurt).**

### Sources (with measured 52-user OOF macro-F1)

| Source | OOF (52-user fold) | Weight (Dirichlet search) |
|---|---:|---:|
| Inception SSL (3-seed avg) | 0.7495 | 29.5% |
| Centered-meta 11-component blend | 0.7715 (with Viterbi) | 68.8% |
| Catch24-LGBM (NEW orthogonal) | 0.7010 | 0.9% |
| Catch24-XGB (NEW orthogonal) | 0.6994 | 0.9% |

The weight search assigned Catch22 near-zero weights — empirically Catch22 features carry orthogonal information but at lower fidelity than convolutional models for this dataset. The ensemble is essentially **70% centered-meta + 30% Inception SSL**, with Viterbi smoothing applied globally and hybrid v3 recovery layered on top.

### Validation

| Tier | Purpose |
|---|---|
| LNUO×6 (Leave-15-Users-Out, 6 seeds) | Weighted-ensemble Dirichlet search (5000 trials) |
| Fresh 12-user HOS (users `004,008,014,018,023,026,037,049,051,054,058,060`) | Diagnostic ONLY — never gates |

The fresh 12-user HOS is **different from yesterday's 8-user HOS** (which had been tuned against twice). Baseline reference on this HOS = centered-meta Viterbi pred = 0.7849.

### Hybrid recovery 3.0

Extends yesterday's proven script with class-3 in addition to class-2 and class-5:

| Target class | Threshold | Cap | Mechanism |
|---|---:|---:|---|
| Class 2 (transient bursts) | ≥ 0.20 soft proba | 35 rows | Restore anchor's prediction if new pipeline's soft proba ≥ threshold |
| Class 3 (rare activity) | ≥ 0.25 | 25 rows | New for v3 — same mechanism, different class |
| Class 5 (rare burst) | ≥ 0.25 | 10 rows | Same as yesterday |

Anchor at test = yesterday's `submission_ssl_hybrid_recover.csv` (0.8240).

### Per-user test-time centering (DISABLED in final pipeline)

I built and tested a zero-parameter per-user softmax centering: for each test user, subtract their per-class softmax mean, clip, re-normalize. On the fresh HOS this **hurt** macro-F1 by −0.007 (0.7752 → 0.7687). The pipeline ships without centering. The script is preserved at `scripts/apply_user_centering.py` for future analysis.

## 4. HOS diagnostic (informational; not a gate)

| Variant | HOS macro-F1 | Δ vs baseline |
|---|---:|---:|
| Centered-meta Viterbi alone (baseline) | 0.7849 | — |
| Ensemble + Viterbi (no hybrid) | 0.7701 | −0.015 |
| Ensemble + Viterbi + hybrid v3 (c2+c3+c5) | **0.7830** | **−0.002** |
| Ensemble + centering(α=0.5) + hybrid v3 | 0.7771 | −0.008 |

The chosen pipeline (ensemble + Viterbi + hybrid v3) lands within −0.002 of baseline HOS — essentially a wash. Per-class on fresh HOS: c0=0.984, c1=0.930, c2=0.357, c3=0.707, c4=0.891, c5=0.829 (vs baseline c2=0.376, c5=0.822 — slight class-2 trade for slight class-5 gain).

## 5. Expected public lift decomposition

Empirical OOF→public ratio from the last week: ~0.5× to 1.0× for genuine model lift, ~−1.0× for OOF-overfit (today's lesson). HOS Δ is a better predictor than OOF Δ for the public direction.

| Scenario | Public | Probability |
|---|---:|---:|
| Catastrophic OOF-overfit-style regression (like today) | 0.79–0.81 | 10% |
| Moderate regression | 0.81–0.82 | 20% |
| Wash with 0.8240 (HOS Δ ≈ 0 translates to public ≈ 0) | 0.82–0.83 | 35% |
| Mild lift (hybrid v3 c2/c3/c5 contributions add) | 0.83–0.84 | 20% |
| Strong lift (test users align with our class-2/3 corrections) | 0.84–0.85 | 10% |
| **Crossing 0.85** | ≥ 0.85 | **5%** |

The realistic expected public is **0.820–0.830**. The 0.85 stretch is now unlikely (~5%) because today's diagnostic revealed the ensemble doesn't add the kind of orthogonal signal that the 0.7849 → 0.85 jump would require. The 0.85 ceiling appears to need a fundamentally different signal source (large pretrained models, external data, or test-time meta-learning across users) — none available in this competition's constraints.

## 6. Risk acknowledgment

The user authorized "all-in, no gate" a second day in a row after seeing today's −0.028 regression. The choice is informed:
- Gain ceiling: ~5% chance of >0.85, ~10% chance of >0.83.
- Loss floor: ~30% chance of regression vs 0.8240; ~10% chance of catastrophic regression.

**The pre-registered diagnostic was respected** — I did not skip the HOS evaluation, even though it doesn't gate the upload. The HOS Δ = −0.002 was used to choose between variants (e.g., centering off, hybrid v3 on), not to override the user's all-in decision.

## 7. Reproduction commands

```bash
# Build fresh HOS + LNUO15×6
.venv/bin/python scripts/build_fresh_holdout12.py
.venv/bin/python scripts/build_extra_folds.py --sgkf-seeds "" --lnuo-seeds 2026,2027,2028,2029,2030,2031 --lnuo-n-users 15

# Catch22-LGBM/XGB full-train + test predictions
.venv/bin/python scripts/train_catch22_full.py

# Weighted ensemble (Dirichlet search on LNUO×6)
.venv/bin/python scripts/build_weighted_ensemble.py \
    --source-oofs artifacts/inception_oof_ssl/oof_inception_avg3.npz \
                  artifacts/blend_search/oof_blend_centered_meta_round2_best.npz \
                  artifacts/catch22_oof/oof_catch22_raw_lgbm_c22.npz \
                  artifacts/catch22_oof/oof_catch22_raw_xgb_c22.npz \
    --source-tests artifacts/inception_full_ssl/test_proba_avg3.npz \
                   artifacts/blend_search/test_blend_centered_meta_viterbi_oof07693.npz \
                   artifacts/catch22_full/test_proba_lgbm.npz \
                   artifacts/catch22_full/test_proba_xgb.npz \
    --source-names inception_ssl centered_meta catch22_lgbm catch22_xgb \
    --lnuo-folds artifacts/folds/lnuo15_seed*.csv

# Viterbi smoothing (tuned on weighted ensemble OOF)
# Inlined in the diagnostic script — alpha=0.1, beta=0.05

# Hybrid v3 + test CSV
.venv/bin/python scripts/make_hybrid_v3_submission.py \
    --new-proba-npz artifacts/weighted_ensemble/test_viterbi.npz \
    --anchor-csv submissions/submission_ssl_hybrid_recover.csv \
    --output submissions/submission_hybrid_v3.csv \
    --c2-threshold 0.20 --max-c2-recovered 35 \
    --c3-threshold 0.25 --max-c3-recovered 25 \
    --c5-threshold 0.25 --max-c5-recovered 10

# Upload at 08:10 (cron + manual backup)
# Locked CSV: submissions/submission_hybrid_v3.csv (SHA 8ac61613...)
```

## 8. Honest closure

Yesterday's 0.7963 regression taught one clear lesson: **at the 0.82+ ceiling, sophistication is a tax**. Tomorrow's plan trades sophistication for surgical additivity. The 0.85 stretch goal is now unlikely (~5%), and the realistic outcome is **0.820–0.830** — competitive with the public best but not the breakthrough. The user accepts that risk; this report documents the choice.

If 0.85 is the firm target, the path forward beyond tomorrow's attempt requires sources of signal that this competition's constraints don't permit: external pretrained models, additional unlabeled data from the same sensor population, or test-time meta-learning across users. The current iteration has reached the natural ceiling of train-data-only learning with the given features.
