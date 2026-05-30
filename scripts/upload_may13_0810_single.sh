#!/bin/bash
# 2026-05-13 08:10 CST single Kaggle upload of hybrid v3 ensemble.
# Per user choice: ALL-IN. Catastrophic CSV-format fallback only.
set -euo pipefail

ROOT="/home/raiso/playground/DM2026-Assignment-3-claude"
cd "$ROOT"

LOCK_FILE="$ROOT/artifacts/decision/upload_may13_0810.lock"
LOG_FILE="$ROOT/artifacts/decision/upload_may13_0810.log"
PRIMARY_CSV="$ROOT/submissions/submission_multi_anchor_vote.csv"
FALLBACK_CSV="$ROOT/submissions/submission_ssl_hybrid_recover.csv"
SUBMISSION_LOG="$ROOT/SUBMISSION_LOG.md"

export HOME="${HOME:-/home/raiso}"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"

mkdir -p "$ROOT/artifacts/decision"

cleanup_cron() {
    if [ "${DM2026_SKIP_CRON_CLEANUP:-0}" = "1" ]; then
        return
    fi
    crontab -l 2>/dev/null | grep -v 'DM2026-May13-0810' | crontab - 2>/dev/null || true
}
trap 'cleanup_cron; echo "[trap] exit at $(date -Iseconds), code=$?" >>"$LOG_FILE"' EXIT

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date -Iseconds)] another instance is running; exiting" >>"$LOG_FILE"
    exit 0
fi

echo "[$(date -Iseconds)] May13 0810 upload started" >>"$LOG_FILE"

SELECTED="$PRIMARY_CSV"
TAG="may13_multi_anchor_vote_85_attempt"

if [ ! -f "$SELECTED" ]; then
    echo "[$(date -Iseconds)] WARN primary missing; fallback" >>"$LOG_FILE"
    SELECTED="$FALLBACK_CSV"
    TAG="may13_catastrophic_fallback_0_8240"
fi

# Format check
"$ROOT/.venv/bin/python" - <<PYEOF >>"$LOG_FILE" 2>&1
import pandas as pd, sys
try:
    sub = pd.read_csv("$SELECTED")
    ref = pd.read_csv("$ROOT/data/raw/sample_submission.csv")
    assert list(sub.columns) == ["Id", "Label"]
    assert (sub["Id"].values == ref["Id"].values).all()
    assert sub["Label"].isin([0,1,2,3,4,5]).all()
    print(f"OK csv-format: {len(sub)} rows")
except Exception as e:
    print(f"FORMAT FAIL: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "[$(date -Iseconds)] primary format failed; fallback to 0.8240" >>"$LOG_FILE"
    SELECTED="$FALLBACK_CSV"
    TAG="may13_catastrophic_fallback_0_8240"
fi

SHA=$(sha256sum "$SELECTED" | awk '{print $1}')
echo "[$(date -Iseconds)] submitting $SELECTED tag=$TAG sha=$SHA" >>"$LOG_FILE"

for attempt in 1 2 3; do
    if [ "${DM2026_DRY_RUN:-0}" = "1" ]; then
        echo "[$(date -Iseconds)] DRY RUN — would submit $SELECTED ($TAG)" >>"$LOG_FILE"
        SUBMIT_RC=0
        break
    fi
    if "$ROOT/.venv/bin/kaggle" competitions submit \
        -c nycu-data-mining-assignment-3 -f "$SELECTED" -m "$TAG" >>"$LOG_FILE" 2>&1; then
        SUBMIT_RC=0
        break
    fi
    echo "[$(date -Iseconds)] attempt $attempt failed; retry in 30s" >>"$LOG_FILE"
    sleep 30
    SUBMIT_RC=1
done

if [ "${SUBMIT_RC:-1}" -ne 0 ]; then
    echo "[$(date -Iseconds)] ERROR all submit attempts failed" >>"$LOG_FILE"
    exit 1
fi

sleep 30
"$ROOT/.venv/bin/kaggle" competitions submissions -c nycu-data-mining-assignment-3 2>>"$LOG_FILE" | head -10 | tee -a "$LOG_FILE" >/dev/null
sleep 60
STATUS_OUTPUT=$("$ROOT/.venv/bin/kaggle" competitions submissions -c nycu-data-mining-assignment-3 2>>"$LOG_FILE" | head -10)
echo "[$(date -Iseconds)] poll status 2:" >>"$LOG_FILE"
echo "$STATUS_OUTPUT" >>"$LOG_FILE"

{
    echo ""
    echo "[2026-05-13 08:10 upload]"
    echo "  csv: $SELECTED"
    echo "  tag: $TAG"
    echo "  sha256: $SHA"
    echo "  status snapshot:"
    echo "$STATUS_OUTPUT" | sed 's/^/    /'
} >>"$SUBMISSION_LOG"

echo "[$(date -Iseconds)] upload complete" >>"$LOG_FILE"
