# Upload Runbook: 2026-05-13 08:10 CST single submission

**Locked CSV**: `submissions/submission_hybrid_v3.csv`
**SHA256**: `8ac616130354d891bb09f933b912b6fc3e621b05e6c05e8592cda3bbded690a7`
**Tag**: `may13_hybrid_v3_85_attempt_all_in`
**Pipeline**: 4-source weighted ensemble (cm 69%, inc 29%, c22 ~1%) + Viterbi (α=0.1, β=0.05) + hybrid v3 (c2×35, c3×1, c5×8) anchored on yesterday's 0.8240 CSV.
**Catastrophic fallback only**: `submissions/submission_ssl_hybrid_recover.csv` (verified 0.8240) — fires only on CSV format error, never on metric quality (per user's "no gate, all-in" choice).

---

## T-1h (07:10 CST) — pre-flight

```bash
SELECTED=submissions/submission_hybrid_v3.csv
EXPECTED_SHA=8ac616130354d891bb09f933b912b6fc3e621b05e6c05e8592cda3bbded690a7

# 1. CSV format check
.venv/bin/python -c "
import pandas as pd
sub = pd.read_csv('$SELECTED'); ref = pd.read_csv('data/raw/sample_submission.csv')
assert list(sub.columns) == ['Id', 'Label']
assert (sub['Id'].values == ref['Id'].values).all()
assert sub['Label'].isin([0,1,2,3,4,5]).all()
print(f'OK csv-format: {len(sub)} rows')
print(f'Label counts: {sub[\"Label\"].value_counts().sort_index().to_dict()}')
"

# 2. SHA matches expected
sha256sum "$SELECTED" | head -c 64 | { read got; [ "$got" = "$EXPECTED_SHA" ] && echo "OK sha-match" || echo "WARN sha-mismatch: $got"; }

# 3. Diagnostic JSON readable (informational)
.venv/bin/python -c "
import json
d = json.load(open('artifacts/decision/diagnostic_v3_final.json'))
print(f'HOS macro-F1: {d[\"hos_diagnostic_macro_f1\"]:.4f} (baseline {d[\"hos_baseline_macro_f1\"]:.4f}, Δ {d[\"hos_delta\"]:+.4f})')
print(f'Recoveries: c2={d[\"recoveries\"][\"c2\"]}, c3={d[\"recoveries\"][\"c3\"]}, c5={d[\"recoveries\"][\"c5\"]}')
"

# 4. Kaggle auth + quota
.venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 | head -5
```

If CSV format check fails → manually set `SELECTED=submissions/submission_ssl_hybrid_recover.csv`.

## T-0h (08:10 CST) — single upload

```bash
SELECTED=submissions/submission_hybrid_v3.csv
.venv/bin/kaggle competitions submit \
    -c nycu-data-mining-assignment-3 \
    -f "$SELECTED" \
    -m "may13_hybrid_v3_85_attempt_all_in" \
    | tee artifacts/decision/upload_may13_0810.log
```

## T+5min (08:15) — poll for status

```bash
for i in 1 2 3 4 5 6 7 8; do
    .venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 | head -6
    STATUS=$(.venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 --csv \
        | head -2 | tail -1 | awk -F, '{print $5}')
    if [[ "$STATUS" == *"complete"* ]] || [[ "$STATUS" == *"COMPLETE"* ]]; then break; fi
    sleep 30
done
```

## T+10min (08:20) — update SUBMISSION_LOG.md

Append one row:

```markdown
| 2026-05-13 00:10:??.??????? | `submissions/submission_hybrid_v3.csv` | `may13_hybrid_v3_85_attempt_all_in` | COMPLETE | <PUBLIC> |  | `8ac616130354d891bb09f933b912b6fc3e621b05e6c05e8592cda3bbded690a7` | <NOTES> |
```

Notes template:

> 2026-05-13 08:10 CST: 4-source weighted ensemble (cm 0.69, inc 0.29, c22 ~0.01 each) + Viterbi (α=0.1, β=0.05) + hybrid v3 (c2×35, c3×1, c5×8) anchored on submission_ssl_hybrid_recover.csv. HOS macro-F1 (fresh 12 users) `0.7830` vs baseline `0.7849` (Δ −0.002). Per-user centering tested and DISABLED (hurt HOS by 0.007). Public lift over 0.8240: `<DELTA>`. Per user's "no gate, all-in" choice, no quality fallback was applied.

## Catastrophic-only fallback

If primary CSV is malformed (NaN/wrong rows/etc.):

```bash
.venv/bin/kaggle competitions submit \
    -c nycu-data-mining-assignment-3 \
    -f submissions/submission_ssl_hybrid_recover.csv \
    -m "may13_emergency_fallback_0_8240"
```

## Post-upload review

| Public | Action |
|---|---|
| ≥ 0.85 | **Goal achieved.** Stop. Document in strategy report Section 8. |
| 0.83–0.85 | Strong success. Document the OOF→public conversion. |
| 0.824–0.83 | Mild lift. Acceptable. |
| 0.815–0.824 | ≈ 0.8240 wash. No regression. |
| < 0.815 | Regression repeat. Document failure modes. Yesterday's 0.8240 remains public best. |
