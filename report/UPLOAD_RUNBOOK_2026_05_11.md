# Upload Runbook: 2026-05-11 08:10 CST single submission

**Locked CSV (decided 2026-05-10 ~14:00 CST)**: `submissions/submission_consensus_oof_greedy.csv`
**Verified public score**: 0.8156 (SHA `aee96eeb…`)
**Decision source**: `artifacts/decision/holdout_decision_report.json` (4-condition gate failed on rules 1, 2, 3; new pipeline regressed by −0.011 macro-F1 on the 8-user holdout)
**Stretch goal of >0.82 was not achieved**; the gate fell through to the verified-best fallback to avoid an OOF-overfitting regression.

---

## Pre-flight (T-1h, 07:00 CST 2026-05-11)

Run this checklist by hand. Each line must print `OK` or be fixed before 08:00.

```bash
# 1. Decision report exists and is current
[ -f artifacts/decision/holdout_decision_report.json ] && echo "OK decision-json"
.venv/bin/python -c "import json,sys; r=json.load(open('artifacts/decision/holdout_decision_report.json')); print('OK upload_decision='+r['upload_decision']); assert r['upload_decision'] in ('new_pipeline','fallback_0.8156'); print('OK valid_decision')"

# 2. Selected CSV exists, sample-format-valid, SHA recorded
SELECTED=$(.venv/bin/python -c "import json; r=json.load(open('artifacts/decision/holdout_decision_report.json')); print(r['selected_csv'])")
[ -f "$SELECTED" ] && echo "OK csv-exists"
.venv/bin/python -c "
import pandas as pd, sys
sub=pd.read_csv('$SELECTED'); ref=pd.read_csv('data/raw/sample_submission.csv')
assert list(sub.columns)==['Id','Label']
assert (sub['Id'].values==ref['Id'].values).all()
assert sub['Label'].isin([0,1,2,3,4,5]).all()
print('OK csv-format')
"
sha256sum "$SELECTED"

# 3. Kaggle credentials live
.venv/bin/kaggle competitions list -c nycu-data-mining-assignment-3 >/dev/null && echo "OK kaggle-auth"

# 4. Today's submission quota not exhausted
.venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 | head -5
```

If decision JSON is missing → fall through: `SELECTED=submissions/submission_consensus_oof_greedy.csv`, log the manual override.

## 08:10 CST: single Kaggle upload

```bash
SELECTED=$(.venv/bin/python -c "import json; r=json.load(open('artifacts/decision/holdout_decision_report.json')); print(r['selected_csv'])")
TAG=$(.venv/bin/python -c "import json; r=json.load(open('artifacts/decision/holdout_decision_report.json')); print(r['upload_decision'])")

.venv/bin/kaggle competitions submit \
  -c nycu-data-mining-assignment-3 \
  -f "$SELECTED" \
  -m "may11_${TAG}" \
  | tee artifacts/decision/upload_may11_0810.log
```

Time: should complete in <30 s. The script emits `Successfully submitted to ...`.

## 08:15 CST: poll for status and score

```bash
for i in 1 2 3 4 5; do
  .venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 | head -10
  STATUS=$(.venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 --csv | head -2 | tail -1 | awk -F, '{print $5}')
  if [ "$STATUS" = "complete" ] || [ "$STATUS" = "SubmissionStatus.COMPLETE" ]; then break; fi
  sleep 60
done
```

## 08:20 CST: write SUBMISSION_LOG.md row

Append one row to the top of the table in `SUBMISSION_LOG.md` using this template:

```markdown
| 2026-05-11 00:10:??.??????? | `<SELECTED>` | `may11_<TAG>` | COMPLETE | <PUBLIC> |  | `<SHA256>` | <NOTES> |
```

Notes template (copy-paste then edit):

> 2026-05-11 08:10 CST single upload from `artifacts/decision/holdout_decision_report.json` decision = `<TAG>`. Pre-registered 4-condition gate; criterion SHA `<CRIT_SHA>`. Held-out 8-user macro-F1 baseline `<BASE_HOS>`, new `<NEW_HOS>` (Δ `<DELTA_HOS>`). Fold-fair Viterbi OOF baseline 0.7693, new `<NEW_OOF>` (Δ `<DELTA_OOF>`). Bootstrap CI on Δ_OOF: `<BOOT_PCT>`%. Class-2 HOS F1 `<C2>`. Class-5 HOS F1 `<C5>`.

## Rollback if upload fails

If `kaggle competitions submit` fails (network/auth/quota) at 08:10:

```bash
sleep 120
.venv/bin/kaggle competitions submit -c nycu-data-mining-assignment-3 \
  -f submissions/submission_consensus_oof_greedy.csv \
  -m may11_rollback_consensus_oof_greedy_resubmit
```

If two retries fail, escalate manually. Do NOT submit untested CSVs from the candidate pool.

## Selection invariants (do not violate)

1. **Selected CSV is the one written into `holdout_decision_report.json` at T-2h.** Never override at 08:10.
2. **One submission only.** The competition allows three per day, but the committed strategy is one shot. Reserving the remaining quota allows manual recovery if the score is anomalous.
3. **No `kaggle competitions submit` calls before 08:00 CST.** Probing the leaderboard breaks the pre-registration discipline.

## Final check

After step 4 reports a public score:

* If `public ≥ 0.82` → submit succeeded. Update `experiments.md` with a brief note. Stop.
* If `0.8156 < public < 0.82` → marginal success. Keep the submission as the new public best, document in `SUBMISSION_LOG.md`, do not retry.
* If `public ≤ 0.8156` → the new pipeline regressed. Use the next day's quota to resubmit `submission_consensus_oof_greedy.csv` to anchor the public best (one-shot insurance).
