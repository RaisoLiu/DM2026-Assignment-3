# Upload Runbook: 2026-05-12 08:10 CST single submission

**Target CSV** (auto-locked by `scripts/emit_final_submission_v2.py` ≤ T-3h): `submissions/submission_ensemble_v2.csv`
**Catastrophic fallback only** (CSV malformed / NaN / missing): `submissions/submission_ssl_hybrid_recover.csv` (verified public 0.8240)
**Decision discipline**: per user instruction, **upload the new ensemble regardless of diagnostic D1–D4 outcomes**. Quality fallback is only triggered by CSV-format failure, never by metric thresholds.

---

## T-1h (07:10 CST) — pre-flight

Run every line by hand. Each must print `OK` or be fixed before 08:00.

```bash
# 1. CSV exists, sample-format-valid
SELECTED=submissions/submission_ensemble_v2.csv
[ -f "$SELECTED" ] && echo "OK csv-exists"
.venv/bin/python -c "
import pandas as pd
sub = pd.read_csv('$SELECTED'); ref = pd.read_csv('data/raw/sample_submission.csv')
assert list(sub.columns) == ['Id', 'Label']
assert (sub['Id'].values == ref['Id'].values).all()
assert sub['Label'].isin([0,1,2,3,4,5]).all()
print('OK csv-format')
print('Label counts:', sub['Label'].value_counts().sort_index().to_dict())
"

# 2. SHA recorded
sha256sum "$SELECTED"

# 3. Diagnostic metrics readable (informational, not gating)
.venv/bin/python -c "
import json
r = json.load(open('artifacts/decision/decision_report_v2.json'))
m = r['metrics']
print(f'D1 OOF={m[\"new_oof_viterbi\"]:.6f}, target {r[\"targets\"][\"d1\"]:.6f}')
print(f'D2 HOS={m[\"new_hos_macro_f1\"]:.6f}, target {r[\"targets\"][\"d2\"]:.6f}')
print(f'D3 bootstrap={m[\"bootstrap_pct\"]:.3f}, target 0.60')
print(f'D4 c2={m[\"new_hos_per_class_f1\"][\"2\"]:.4f}, c5={m[\"new_hos_per_class_f1\"][\"5\"]:.4f}')
"

# 4. Kaggle credentials live
.venv/bin/kaggle competitions list -c nycu-data-mining-assignment-3 >/dev/null && echo "OK kaggle-auth"

# 5. Today's submission quota status (each new day resets to 3)
.venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 | head -8
```

If `submission_ensemble_v2.csv` is missing or invalid format at T-1h, fall back:
```bash
SELECTED=submissions/submission_ssl_hybrid_recover.csv
```

## T-0h (08:10 CST) — single upload

```bash
SELECTED=submissions/submission_ensemble_v2.csv  # or fallback if pre-flight failed
DESC="may12_ensemble_v2_85_attempt"  # adjust tag based on actual artifacts

.venv/bin/kaggle competitions submit \
    -c nycu-data-mining-assignment-3 \
    -f "$SELECTED" \
    -m "$DESC" \
    | tee artifacts/decision/upload_may12_0810.log
```

## T+5min (08:15 CST) — poll for score

```bash
for i in 1 2 3 4 5 6 7 8; do
    .venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 | head -6
    STATUS=$(.venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 --csv \
        | head -2 | tail -1 | awk -F, '{print $5}')
    if [[ "$STATUS" == *"complete"* ]] || [[ "$STATUS" == *"COMPLETE"* ]]; then
        break
    fi
    sleep 30
done
```

## T+10min (08:20 CST) — update SUBMISSION_LOG.md

Append one row to the top of the table:

```markdown
| 2026-05-12 00:10:??.??????? | `<SELECTED>` | `<DESC>` | COMPLETE | <PUBLIC> |  | `<SHA256>` | <NOTES> |
```

Notes template (copy & edit):

> 2026-05-12 08:10 CST single upload. Pipeline: 4-architecture ensemble (PatchTST + TCN + ResNet1D + SSL-InceptionTime v2) + consensus pseudo-labeling (6,690 rows weight 0.4) + LGBM meta-learner stacker + learned hybrid recovery 2.0 anchored on 0.8240 hybrid. OOF fold-fair Viterbi `<D1>`, HOS `<D2>` (baseline 0.6670), bootstrap pct `<D3>`, class-2 F1 `<c2>`, class-5 F1 `<c5>`. Pre-registered targets: D1≥0.8071, D2≥0.6820, D3≥0.60. **All-in upload per user choice** — no quality fallback; only catastrophic CSV-format fallback to 0.8240. Public result: `<PUBLIC>`.

## Post-upload action items

- If `public ≥ 0.85`: **goal achieved**. Update report with final numbers. Stop.
- If `0.8240 < public < 0.85`: partial success. Document the OOF→public conversion ratio for the four pillars. Update strategy report's Section 6 with actuals.
- If `public ≤ 0.8240`: regression. The 0.8240 hybrid remains the public best. Document failure mode for future work. Today's remaining quota (2 left after this one) can be used to manually re-submit 0.8240 if the leaderboard tracks "latest" not "best."

## Catastrophic fallback (CSV malformed / pipeline crashed)

```bash
.venv/bin/kaggle competitions submit \
    -c nycu-data-mining-assignment-3 \
    -f submissions/submission_ssl_hybrid_recover.csv \
    -m "may12_emergency_fallback_0_8240"
```

Do **not** trigger this for D1/D2/D3 failures — those are diagnostic-only by user instruction.

## Selection invariants

1. **The selected CSV path is locked at T-3h** by `scripts/emit_final_submission_v2.py`.
2. **No public-leaderboard probing before 08:10** — each probe burns a quota.
3. **Only the catastrophic fallback can override the locked CSV** — no metric-based switching.
