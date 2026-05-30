# Strategy and Improvement Report: 0.85+ in One Shot

**Author**: DM2026 Assignment 3 pipeline (PI-style review draft)
**Date**: 2026-05-11 (planning) → 2026-05-12 08:10 CST (upload)
**Public best to beat**: 0.8240 (`submissions/submission_ssl_hybrid_recover.csv`, uploaded today 2026-05-11 03:01 UTC)
**Stretch goal**: > 0.85 macro-F1 (a +0.026 lift, ≈3× today's day's lift)

---

## 1. Where today's 0.8240 came from

Today (2026-05-11) the pipeline jumped from yesterday's 0.8156 to 0.8240 via two compounding moves:

1. **SimCLR contrastive pretraining** on 17,869 combined train+test sequences (150 epochs, NT-Xent loss, custom HAR augmentations). This gave InceptionTime a representation that no convolutional component in the existing 11-component blend had — bumping fine-tune OOF base from 0.660 to 0.707 (+0.047 stand-alone) and getting the InceptionTime + centered-meta blend to 0.7871 fold-fair Viterbi OOF (+0.012 over centered-meta-only). Public: 0.8175.
2. **Hybrid recovery** that restored 25 baseline-class-2 and 5 baseline-class-5 predictions where the new SSL pipeline's own *soft* probability for those rare classes was still plausibly high (≥0.27 for class 2, ≥0.25 for class 5). The 30-row tweak produced +0.0065 public (0.8175 → 0.8240) — vastly outsized vs. the +0.0001 OOF impact, indicating that test-set-specific row-level decisions matter far more than aggregate OOF.

Two empirical regularities now drive the 0.85 plan:

- **Architectural diversity matters**: SSL InceptionTime was orthogonal to the 11 existing components (XGB/CatBoost/MiniRocket/MultiRocket/event-LGBM/meta-LGBM/meta-XGB). Adding more *similar* models (e.g., another CNN seed) yields diminishing returns; adding *different* model families compounds.
- **OOF→public mapping is weak (~0.15×) for aggregate metrics but strong for row-level decisions**: when the same OOF improvement was realized through a learned per-row rule (the hybrid), the public lift was 3–5× larger than the macro-F1 lift would predict.

## 2. Why 0.85 is hard

The +0.026 gap requires combining several genuinely orthogonal signals. No single technique today gives that lift on this dataset. Class-2 macro-F1 on held-out users sits at 0.21 even after SSL+hybrid; only ~50% of class-2 windows are correctly identified. The 11,020 train labels are saturated for convolutional models — convolutional architectures with different kernel families (Inception, MiniRocket, MultiRocket) all converge near 0.78 fold-fair Viterbi OOF.

New lift must come from:
1. **Fundamentally different inductive bias** — attention-based or different convolutional shapes.
2. **The unlabeled test data** — 6,849 windows from 40 disjoint users contain the user-distribution-shift information that train alone cannot teach.
3. **Better row-level model selection** — learned rules over multiple disagreeing models.

## 3. Selected approach

**Multi-architecture ensemble (4 model families) + consensus-based pseudo-labeling on test + cross-validated hybrid recovery 2.0 + repeated 5-fold × 5 seeds validation**. All four signals combined; all-in upload regardless of diagnostic outcome.

### Pillar 1 — Three new architectures
- **PatchTST Transformer**: 30 patches of 10 timesteps, 4-layer self-attention (d_model=128, n_heads=8). Captures long-range dependencies via attention rather than convolution. Different inductive bias from any current ensemble member.
- **TCN**: 6 dilated TemporalBlocks (dilations 1, 2, 4, 8, 16, 32), residual. Exponentially growing receptive field via dilation — covers temporal scales that InceptionTime's parallel kernels do not.
- **ResNet1D-50**: deep residual CNN with channel attention, ~5M params. Captures fine-grained features that InceptionTime's bottleneck design (32-dim bottleneck) may compress away.

All three are initialized from the same SimCLR-pretrained encoder where shape-compatible (PatchTST/TCN/ResNet share early conv weights with InceptionTime's blocks).

### Pillar 2 — Consensus pseudo-labeling
The three best submissions of today (0.8156, 0.8175, 0.8240) agree on **6,690 of 6,849 test rows (97.7%)** — much higher than the 80% I estimated. These 6,690 rows are *very likely* correct (they passed three independent classifiers). Using them as additional training labels:
- Adds 6,690 weakly-labeled samples (60% more than the original 11,020) — significant data augmentation.
- Specifically adds 153 class-2 pseudo-labels (vs. 358 in real train) — a 43% increase in class-2 supervision.
- Comes from test-distribution users — directly addresses the disjoint-user generalization bottleneck.

We weight pseudo-labels at 0.4 (vs 1.0 for real) to limit error amplification.

### Pillar 3 — Meta-learner stacker
Per-row LGBM stacker over the 4 architectures' soft probabilities:
- 4 models × 6 class softmax = 24 features
- 4 per-model max-margin (best − 2nd best class) = 4 features
- Inter-model agreement count = 1
- Per-user position context = 1
- Per-user training-prior class distribution = 6
- Total = 36 features

This learns *when* each model is trustworthy — going beyond today's manual hybrid rule.

### Pillar 4 — Hybrid recovery 2.0
Today's rule (restore baseline class-2 where SSL class-2 proba ≥ 0.27, top-25 rows) was hand-tuned. Tomorrow we train a learned binary classifier per `(meta-learner-prediction, anchor-prediction)` transition. For each test row, the classifier decides whether to switch from the meta-learner's argmax to today's 0.8240 hybrid argmax. The anchor at test time is **today's `submission_ssl_hybrid_recover.csv` predictions** (the 0.8240 file).

### Pillar 5 — Repeated 5-fold × 5 seeds validation
The same 8-user HOS is preserved for direct comparability to today's 0.8240 baseline_HOS (0.6670). On the 52-user training partition, 5 seeds × 5 folds = 25 fold evaluations. This reduces fold-variance σ from ~0.005 (2 seeds) to ~0.002 — meaning the diagnostic D1 threshold (OOF ≥ 0.8071) is statistically meaningful.

## 4. Expected lift decomposition

| Component | Expected OOF lift | Expected public lift |
|---|---:|---:|
| Add PatchTST as 5th architecture | +0.005 | +0.001 to +0.003 |
| Add TCN as 6th architecture | +0.003 | +0.000 to +0.002 |
| Add ResNet1D as 7th architecture | +0.002 | +0.000 to +0.002 |
| Consensus pseudo-labeling (6,690 rows) | +0.008 | +0.003 to +0.010 |
| Meta-learner stacker | +0.005 | +0.005 to +0.015 |
| Hybrid recovery 2.0 (learned) | +0.002 (OOF unaffected) | +0.005 to +0.015 |
| **Subtotal (additive, median)** | **+0.025 OOF** | **+0.014 to +0.047 public** |

Three scenarios for the final public macro-F1:

- **Pessimistic (only 1-2 components help, weak transfer)**: +0.008 public → **0.832**
- **Median (most components help, average transfer)**: +0.020 public → **0.844**
- **Optimistic (all components help, strong transfer like today's hybrid)**: +0.045 public → **0.869**
- **Realistic probability of crossing 0.85**: ~25–40%

## 5. Risk analysis (and what we explicitly accept)

1. **Pseudo-label error amplification**. The 6,690 pseudo-labels include some incorrect ones (maybe 5–10%). At weight 0.4, the influence is bounded. Validation: train one architecture (TCN, fastest) with and without; if OOF drops, drop pseudo-labels entirely.
2. **Architecture failures**. PatchTST/TCN/ResNet1D may fail to converge in the 30-epoch budget. Time-pressure abort rule: drop any architecture that doesn't hit ≥0.65 fold-fair OOF base by T-14h; proceed with the survivors.
3. **OOF-overfitting**. The meta-learner and hybrid recovery 2.0 both train on OOF; both could overfit. Mitigation: cross-validate the meta-learner with the same user grouping; cross-validate recovery thresholds.
4. **All-in upload risk** (per user choice). If the new ensemble lands at 0.81x, we lose the 0.8240 floor. Documented but accepted.
5. **Cron failure** like yesterday. Mitigation: manual `kaggle submit` at 08:10, no cron dependency.

## 6. Honest probability assessment

- Most likely public score range: **0.82 – 0.85** (60% probability mass)
- Probability of >0.85: **25–40%**
- Probability of <0.8240 (regression vs today): **10–15%**
- Worst-case catastrophic (CSV malformed): <1% (automatic format check)

The 0.85 target is ambitious but plausible given that two of today's components (SSL pretraining, hybrid recovery) each delivered surprising public lifts (+0.0019 and +0.0065 respectively from small OOF moves). Four more orthogonal sources could plausibly compound.

## 7. Reproduction commands

```bash
# T-21h
.venv/bin/python scripts/build_extra_folds.py
.venv/bin/python scripts/build_pseudo_labels.py

# T-20h: smoke each new architecture
.venv/bin/python scripts/train_patchtst_oof.py --epochs 3 --folds 1 --output-dir artifacts/patchtst_oof/smoke
.venv/bin/python scripts/train_tcn_oof.py --epochs 3 --folds 1 --output-dir artifacts/tcn_oof/smoke
.venv/bin/python scripts/train_resnet1d_oof.py --epochs 3 --folds 1 --output-dir artifacts/resnet1d_oof/smoke

# T-18h to T-14h: full OOF training
for arch in patchtst tcn resnet1d; do
    for seed in 2026 2027 2028; do
        .venv/bin/python scripts/train_${arch}_oof.py \
            --fold-file artifacts/folds/sgkf_seed${seed}_train52.csv \
            --output-dir artifacts/${arch}_oof/seed${seed} --epochs 30
    done
done

# T-14h: pseudo-label validation
.venv/bin/python scripts/train_tcn_oof.py --output-dir artifacts/tcn_oof/seed2026_pseudo \
    --pseudo-labels artifacts/pseudo/pseudo_labels_consensus3.csv

# T-11h: meta-learner + hybrid recovery
.venv/bin/python scripts/train_meta_stacker.py --output artifacts/meta_stacker/
.venv/bin/python scripts/train_hybrid_rules.py \
    --meta-oof artifacts/meta_stacker/oof_meta.npz \
    --anchor-csv submissions/submission_ssl_hybrid_recover.csv \
    --output artifacts/hybrid_rules/

# T-6h: full-train + test predictions + final CSV
for arch in patchtst tcn resnet1d; do
    .venv/bin/python scripts/train_${arch}_full.py --output-dir artifacts/${arch}_full --epochs 30 --seeds 2026,2027
done
.venv/bin/python scripts/emit_final_submission_v2.py --output submissions/submission_ensemble_v2.csv

# T-0h: manual upload
.venv/bin/kaggle competitions submit -c nycu-data-mining-assignment-3 \
    -f submissions/submission_ensemble_v2.csv -m "may12_ensemble_v2_85_attempt"
```

## 8. Decision context (why all-in)

Yesterday the pre-registered gate prevented an OOF-overfit regression; today the gate enabled the soft signal of the hybrid recovery to be discovered. Tomorrow, the user has explicitly chosen *all-in* with no quality fallback — the goal is to maximize the chance of crossing 0.85, accepting the increased variance. The pre-registered diagnostic metrics (D1–D4) are recorded for documentation but are not gates.

If the new pipeline lands above 0.8240, today's discipline is vindicated. If it lands below, the lesson is that 0.85 requires not just more compute and architectures but a structurally different signal source (e.g., much larger pretrained models or richer external data) that this competition's constraints don't allow.

---

## 9. Post-execution outcome (2026-05-12 08:10 CST)

**Actual public score: 0.7963 — a major regression of −0.028 versus yesterday's 0.8240.**

This is the all-in failure mode. Every pre-registered diagnostic looked excellent:
- OOF macro-F1: 0.8118 (+0.027 over yesterday)
- Bootstrap pct(new > baseline): **100%**
- Per-class OOF F1 improved across the board: c2 0.38 (was 0.32), c5 0.88 (was 0.86)

Yet the public score *inverted*. Three compounding root causes:

1. **Meta-learner over-fit the training fold structure**. The LGBM meta-learner, trained on all 9,589 OOF rows with no held-out CV at the meta level, hit 1.000 TRAIN macro-F1. It memorized the OOF prediction patterns of the 52 training users — patterns that don't reproduce on the 40 disjoint test users.

2. **Pseudo-labels at scale injected majority-class bias**. The 6,690 pseudo-labels (60% extra data, weighted 0.4) carried the prior distribution of yesterday's CSVs, which class-2 was already under-represented in. The new models trained on pseudo-augmented data systematically under-predicted class 2 on test (final count 149 vs anchor's 222).

3. **Hybrid recovery 2.0 was tuned to OOF disagreement patterns**. The classifier learned where meta_pred disagreed with the centered-meta-Viterbi anchor *on TRAIN OOF*, but the test users' disagreement patterns are different. The 90 row switches on test were largely uncorrected errors.

**What the pre-registered gate would have done**: if applied (which the user disabled for the all-in attempt), the gate's HOS check would have likely flagged the regression. The OOF-only signal (+0.027) plus 100% bootstrap was overwhelmingly confident — but bootstrap on OOF cannot detect OOF→public distribution shift. This is the failure mode the held-out 8-user set was supposed to guard against.

### What we learned

1. **At 0.82+, "more is less"**. Stacking multiple weakly-orthogonal architectures, pseudo-labeling at high volume, and learned recovery rules all compound the same fold-structure overfitting. Yesterday's success (0.8156 → 0.8240) came from a *tiny* (30 row) targeted correction, not a complex pipeline.
2. **Pseudo-labeling beyond ~10% of training data is dangerous on disjoint-user setups**. The 60% pseudo-augmentation here destroyed test-distribution generalization.
3. **Meta-learner overfitting is harder to detect than base-learner overfitting**. The meta-learner sits on top of OOF predictions that are themselves fold-fair; the meta-LGBM has no "natural" held-out signal unless explicitly constructed.
4. **HOS-based gates work, OOF bootstrap doesn't**. The 100% OOF bootstrap was misleading; an HOS check would likely have shown a regression and triggered fallback.

### Remaining options for the next 24h

The 0.8240 hybrid (yesterday) is still the public best. Today's remaining quota (2/3 left after this upload) could be used to:

- Re-upload `submission_ssl_hybrid_recover.csv` to anchor the public best
- Try a *narrower* targeted variant of v2 (e.g., meta-learner ONLY without hybrid recovery, or hybrid recovery on Viterbi-smoothed proba)
- Stop and accept 0.8240 as the public best

The 0.85 stretch goal was not met. The all-in commit cost us 0.028 public.
