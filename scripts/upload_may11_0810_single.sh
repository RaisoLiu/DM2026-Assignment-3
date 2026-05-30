#!/bin/bash
# 2026-05-11 08:10 CST single Kaggle upload.
# Reads artifacts/decision/holdout_decision_report.json to choose the CSV.
# On gate fail (which is the locked decision as of 2026-05-10) it submits
# submissions/submission_consensus_oof_greedy.csv (verified public 0.8156).
set -euo pipefail

ROOT="/home/raiso/playground/DM2026-Assignment-3-claude"
LOCK_FILE="$ROOT/artifacts/decision/upload_may11_0810.lock"
LOG_FILE="$ROOT/artifacts/decision/upload_may11_0810.log"
DECISION_FILE="$ROOT/artifacts/decision/holdout_decision_report.json"
SUBMISSION_LOG="$ROOT/SUBMISSION_LOG.md"

mkdir -p "$ROOT/artifacts/decision"

cleanup_cron() {
    if [ "${DM2026_SKIP_CRON_CLEANUP:-0}" = "1" ]; then
        return
    fi
    crontab -l 2>/dev/null | grep -v 'DM2026-May11-0810' | crontab - 2>/dev/null || true
}
trap 'cleanup_cron; echo "[trap] exit at $(date -Iseconds), code=$?" >>"$LOG_FILE"' EXIT

# Single-execution lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date -Iseconds)] another instance is running; exiting" >>"$LOG_FILE"
    exit 0
fi

echo "[$(date -Iseconds)] upload script started" >>"$LOG_FILE"

# Choose CSV based on decision
if [ ! -f "$DECISION_FILE" ]; then
    SELECTED_CSV="$ROOT/submissions/submission_consensus_oof_greedy.csv"
    TAG="may11_fallback_no_decision_file"
    echo "[$(date -Iseconds)] WARN no decision file, using fallback" >>"$LOG_FILE"
else
    SELECTED_CSV=$(.venv/bin/python -c "import json,sys,pathlib; r=json.load(open('$DECISION_FILE')); print(pathlib.Path(r['selected_csv']).resolve())" 2>>"$LOG_FILE")
    DECISION=$(.venv/bin/python -c "import json; r=json.load(open('$DECISION_FILE')); print(r['upload_decision'])" 2>>"$LOG_FILE")
    TAG="may11_${DECISION}"
fi

if [ ! -f "$SELECTED_CSV" ]; then
    echo "[$(date -Iseconds)] ERROR selected CSV does not exist: $SELECTED_CSV" >>"$LOG_FILE"
    SELECTED_CSV="$ROOT/submissions/submission_consensus_oof_greedy.csv"
    TAG="may11_emergency_fallback"
fi

echo "[$(date -Iseconds)] selected $SELECTED_CSV tag=$TAG" >>"$LOG_FILE"

# Validate format
.venv/bin/python - <<PYEOF >>"$LOG_FILE" 2>&1
import pandas as pd, sys
sub = pd.read_csv("$SELECTED_CSV")
ref = pd.read_csv("$ROOT/data/raw/sample_submission.csv")
assert list(sub.columns) == ["Id", "Label"], f"bad columns: {sub.columns}"
assert (sub["Id"].values == ref["Id"].values).all(), "Id order mismatch"
assert sub["Label"].isin([0,1,2,3,4,5]).all(), "invalid labels"
print(f"OK csv-format: {len(sub)} rows")
PYEOF

# SHA
SHA=$(sha256sum "$SELECTED_CSV" | awk '{print $1}')
echo "[$(date -Iseconds)] sha256=$SHA" >>"$LOG_FILE"

# Submit (with retries)
SUBMIT_CMD=".venv/bin/kaggle competitions submit -c nycu-data-mining-assignment-3 -f \"$SELECTED_CSV\" -m \"$TAG\""
SUBMIT_RC=1
for attempt in 1 2 3; do
    echo "[$(date -Iseconds)] submit attempt $attempt: $SUBMIT_CMD" >>"$LOG_FILE"
    if [ "${DM2026_DRY_RUN:-0}" = "1" ]; then
        echo "[$(date -Iseconds)] DRY RUN — would submit $SELECTED_CSV (tag $TAG)" >>"$LOG_FILE"
        SUBMIT_RC=0
        break
    fi
    if cd "$ROOT" && eval "$SUBMIT_CMD" >>"$LOG_FILE" 2>&1; then
        SUBMIT_RC=0
        break
    fi
    echo "[$(date -Iseconds)] attempt $attempt failed; retrying in 30s" >>"$LOG_FILE"
    sleep 30
done

if [ "$SUBMIT_RC" -ne 0 ]; then
    echo "[$(date -Iseconds)] ERROR all submit attempts failed" >>"$LOG_FILE"
    exit 1
fi

# Poll for status
sleep 30
STATUS_OUTPUT=""
for poll in 1 2 3 4 5 6; do
    STATUS_OUTPUT=$(cd "$ROOT" && .venv/bin/kaggle competitions submissions -c nycu-data-mining-assignment-3 2>>"$LOG_FILE" | head -10)
    echo "[$(date -Iseconds)] poll $poll status:" >>"$LOG_FILE"
    echo "$STATUS_OUTPUT" >>"$LOG_FILE"
    if echo "$STATUS_OUTPUT" | grep -q -i "complete"; then
        break
    fi
    sleep 30
done

# Append to SUBMISSION_LOG.md
{
    echo ""
    echo "[2026-05-11 08:10 upload]"
    echo "  csv: $SELECTED_CSV"
    echo "  tag: $TAG"
    echo "  sha256: $SHA"
    echo "  status snapshot:"
    echo "$STATUS_OUTPUT" | sed 's/^/    /'
} >>"$SUBMISSION_LOG"

echo "[$(date -Iseconds)] upload complete; status appended to SUBMISSION_LOG.md" >>"$LOG_FILE"
