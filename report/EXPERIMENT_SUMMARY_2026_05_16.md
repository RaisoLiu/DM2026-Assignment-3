# 2026-05-16 Experiment Summary

## Status snapshot

- **Public best**: `0.8248` (`submissions/submission_h8_v3_sslv2_weighted.csv`, May 14 08:10 CST cron)
- **Gap to 0.85**: `+0.0252` (≈3× the largest single-day public lift we've ever produced)
- **Competition rules** (re-read from `ref/DM_114_Assignment 3.pptx.pdf`):
  - 30% of grade = public LB, 70% = private LB; deadline June 10
  - External pretrained models are NOT forbidden by the spec
  - Code+results must be reproducible from public GitHub repo (page 20)
- **Compute**: local RTX 3070 / 8 GB + remote `raiso@192.168.195.43` RTX 4090 / 24 GB
- **Today's quota**: 2/3 used (manual_rules, byol_low5_blend), 1 left

---

## Today's submissions

| Slot | Time CST | CSV | Public | Δ vs anchor | Verdict |
|---|---|---|---:|---:|---|
| 1 | 12:13 | `submission_manual_rules.csv` | **0.8186** | −0.0062 | regression |
| 2 | 12:55 | `submission_byol_low5_blend.csv` | **0.8155** | −0.0093 | regression |
| 3 | — | (skipped, saved quota) | — | — | — |

Both submissions regressed. Public best unchanged at 0.8248.

### Why slot 1 (manual rules) failed

Visualization agent inspected the 144 uncertain test rows (≤3/5 anchor agreement),
proposed 5 hand-crafted rules from per-class signal statistics. Applied 4 rules
(14 row changes total) on top of the 0.8248 anchor:
- Rule B (3→4 on saturated activity): 5 changes, mostly User_065 cluster
- Rule C (3→2 on intermittent activity): 3 changes
- Rule D (1→2 on mid-energy + level shifts): 6 changes

TRAIN-data validation of Rule B threshold showed only ~40% of "frac_high_std≥0.95"
rows were actually class-4 (the other 60% were class-3) — so the User_065 cluster
flip was unlikely to all be right. The 14 changes net regressed the public score
by −0.0062.

**Lesson**: even visual-pattern hand rules with tiny footprint don't generalize.
The H8 model's per-user calibration outranks ad-hoc overrides.

### Why slot 2 (BYOL 5% blend) failed

Trained a 12-block InceptionTime encoder with BYOL self-supervision on combined
train+test (17,869 sequences) on remote RTX 4090. Best loss 0.2549 at epoch 130,
killed at epoch 180 due to plateau + slight upward drift.

Extracted 160-dim per-window embeddings, trained 5-fold LGBM OOF:
- **OOF macro 0.6473** (vs existing ensemble 0.7830)
- Per-class F1: `[0.946, 0.867, 0.064, 0.668, 0.815, 0.523]` — c2 = 0.064 catastrophic

Blended at fixed 5% weight on top of existing 4-source weighted ensemble,
Viterbi-smoothed. Test c2 count dropped 184→82 (lost 102 c2 predictions to other
classes). Public 0.8155.

**Lesson**: pooled-vector encoder representations (Chronos, BYOL) destroy the
time-localized signal that distinguishes class-2 transitions. Even at 5% weight,
a weak c2 signal pulls down the ensemble's c2 calibration.

---

## Per-source OOF audit (just computed)

| Source | OOF macro | c0 | c1 | **c2** | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| inception_ssl_v1 (current ensemble) | 0.7181 | 0.954 | 0.879 | 0.298 | 0.731 | 0.772 | 0.675 |
| centered_meta (current ensemble) | 0.7547 | 0.979 | 0.920 | 0.300 | 0.711 | 0.906 | 0.712 |
| catch22_lgbm (current ensemble) | 0.6833 | 0.959 | 0.889 | 0.152 | 0.686 | 0.835 | 0.578 |
| catch22_xgb (current ensemble) | 0.6813 | 0.957 | 0.862 | 0.235 | 0.663 | 0.845 | 0.525 |
| **inception_ssl_v2** (NOT in ensemble) | 0.7495 | 0.959 | 0.892 | **0.344** | 0.753 | 0.814 | 0.734 |
| **inception_ssl_v3** (NOT in ensemble) | **0.7584** | 0.962 | 0.896 | **0.350** | 0.746 | 0.857 | 0.740 |
| c2_aug (today's attempt, REJECT) | 0.6870 | 0.960 | 0.871 | 0.253 | 0.694 | 0.779 | 0.566 |
| Chronos LGBM (today's attempt, REJECT) | 0.6605 | 0.963 | 0.889 | 0.031 | 0.690 | 0.808 | 0.582 |
| BYOL v3 LGBM (today's attempt, REJECT) | 0.6473 | 0.946 | 0.867 | 0.064 | 0.668 | 0.815 | 0.523 |
| **Existing 4-source weighted ensemble** | **0.7830** | 0.977 | 0.920 | **0.343** | 0.753 | 0.902 | 0.803 |

### Key insight (newly discovered today)

**SSL v3 individually has c2 F1 = 0.350, HIGHER than the 4-source ensemble (0.343).**
SSL v2 has c2 = 0.344, also higher than the ensemble. The current 4-source ensemble
INCLUDES inception_ssl_v1 (c2 = 0.298) but not v2 or v3.

This suggests two things:
1. The ensemble's c2 = 0.343 is being held DOWN by catch22 sources (c2 = 0.15-0.24)
   that contribute non-c2 weight.
2. We never deployed SSL v3 or v2 test predictions into the public-best pipeline.
   The H8 0.8248 was built on SSL v1 (v2 came later as "v3 sslv2 weighted" via H8 family
   hybrid recovery, but the underlying ensemble still uses v1).

**If we could swap SSL v1 → SSL v3 in the ensemble, expected c2 lift might be
+0.05 absolute F1**. That alone could move macro by ~+0.008-0.012.

Caveat: SSL v3 test predictions DO NOT EXIST locally. Only OOF (`artifacts/inception_oof_ssl_v3/`)
exists; full-train SSL v3 test_proba was never produced. Need to either:
- Re-run finetune with SSL v3 init on full 60 users + 3 seeds → ~30 min remote
- OR train m60 inference scripts from existing SSL v3 OOF checkpoints if they're preserved
  (need to check `artifacts/inception_oof_ssl_v3/seed{2026,2027,2028}/`)

---

## Failed strategy catalog (with reasons)

| Strategy | OOF c2 F1 | Public Δ | Why it failed |
|---|---:|---:|---|
| Chronos-bolt-small + LGBM (5 May 16 earlier exp) | 0.031 | (not uploaded) | Pooled foundation embedding destroys c2 transition shape |
| BYOL v3 (12-block, 500-ep target) + LGBM | 0.064 | (in slot 2) | Same as Chronos — pooled representation problem |
| BYOL v3 at 5% blend weight | -- | −0.0093 | Even tiny BYOL weight collapses c2 predictions |
| Visual hand rules (14 changes) | -- | −0.0062 | Doesn't generalize beyond visualized sample |
| C2 oversample×4 + within-c2 mixup finetune | 0.253 | (not uploaded) | 358 c2 samples too few; mixup destroys transition shape |
| C2 binary specialist + attention pool | 0.221 (AP 0.145) | (not uploaded) | Attention pool doesn't beat existing methods |
| Catch22 LGBM/XGB specialists for c2 | 0.15-0.24 | (already in ensemble) | Handcrafted features insufficient for c2 |

---

## What's left to try (ranked by EV)

### Tier A — high-EV, low compute (next session)

1. **Add SSL v3 to the weighted ensemble** (highest EV)
   - Run full-60-user finetune with SSL v3 init × 3 seeds on remote (~30 min)
   - Add as 5th source to `build_weighted_ensemble.py`
   - Expected: c2 lift +0.03-0.05, macro lift +0.005-0.012
   - Public expectation: 0.825-0.832 if c2 lift transfers
   - Risk: SSL v3 OOF is on 11,020-row full train, not 9,589-row 52-user subset.
     Need to align fold structure first.

2. **Multi-anchor vote of top-3 H8 anchors**
   - 0.8248, 0.8244, 0.8240 — same family, votes mostly agree (>95%)
   - Disagreements get resolved by 2/3 majority
   - Bounded downside (±0.002), upside marginal
   - Low risk hedge

### Tier B — medium-EV, medium compute

3. **C2-precision-promote ensemble**
   - Take BYOL c2 + Chronos c2 + ensemble c2 probabilities
   - PROMOTE a row to c2 only if ALL THREE rank c2 in top-2 AND ensemble c2 proba > 0.30
   - Should be high-precision low-recall; potentially +5-10 correct c2 promotions
   - Cost: ~30 min to build + verify on HOS

4. **Transformer encoder (PatchTST or vanilla) replacing InceptionTime**
   - Different inductive bias (cross-timestep attention)
   - 1-2 hours train on RTX 4090
   - May or may not beat InceptionTime; high variance outcome

### Tier C — low-EV but maybe surprising

5. **Cross-user augmentation** — synthesize new c2 examples by averaging time-shifted
   pairs from the SAME class-2 across DIFFERENT users
6. **Per-user nearest-neighbor calibration** at test time (Bayesian variant; the
   previous attempt zeroed c2, must redo more carefully)
7. **HIVE-COTE components** (BOSS dictionary, Shapelet Transform via aeon) — different
   feature space than what we've tried

### Tier D — defensive (private LB)

8. **Re-submit 0.8248 anchor untouched** + a slightly more conservative variant
   (fewer hybrid corrections) for private-LB hedge

---

## Compute infrastructure status

- Local: 8 GB RTX 3070 (fine for small models), `.venv` shared with sibling project
- Remote: 24 GB RTX 4090 (`raiso@192.168.195.43`), Python 3.10.12, 39 GB disk free
- Project on remote: `/home/raiso/DM2026-Asg3-byol/` — has artifacts/sequence/, scripts/, artifacts/inception_byol_v3/encoder.pt, artifacts/byol_v3/embeddings, artifacts/byol_lgbm/, artifacts/inception_c2_aug/, artifacts/c2_seq_specialist/
- All BYOL/c2-aug artifacts synced back to local

---

## Lessons learned (for record)

1. **Class-2 is structurally hard, not solvable by adding sources**
   - 358 train samples, high intra-class variance, transition-shape signal that
     pooled representations destroy.
   - Foundation models (Chronos, BYOL) have c2 F1 ≤ 0.07 — orders of magnitude worse
     than InceptionTime+SSL (0.30-0.35).
2. **Surgical row-level corrections (≤30 rows) are the only mechanism that gave
   public lifts (>0.005)**. Larger changes regressed every time.
3. **Multi-anchor voting at high diversity (5 anchors) regressed**; small-set voting
   (top-3 within H8 family) hasn't been tried but likely marginal.
4. **The H8 hybrid-recovery layer (specialist-aware c2/c3/c4/c5 restore) is the most
   value-add mechanism** — adding it on top of any sane base gives +0.005-0.008.
   Newer base submissions that skip this layer (like today's slot 2) lose this lift.
