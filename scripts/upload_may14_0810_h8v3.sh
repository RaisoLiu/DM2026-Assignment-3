#!/bin/bash
# 2026-05-14 08:10 CST single Kaggle upload.
# Slot 1 of Day 2: H8.v3 sslv2-weighted variant.
set -euo pipefail
ROOT="/home/raiso/playground/DM2026-Assignment-3-claude"
cd "$ROOT"
LOG_FILE="$ROOT/artifacts/decision/upload_may14_0810.log"
LOCK_FILE="$ROOT/artifacts/decision/upload_may14_0810.lock"
PRIMARY_CSV="$ROOT/submissions/submission_h8_v3_sslv2_weighted.csv"
FALLBACK_CSV="$ROOT/submissions/submission_h8_uncertain_refined.csv"

export HOME="${HOME:-/home/raiso}"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
mkdir -p "$ROOT/artifacts/decision"

cleanup_cron() {
    if [ "${DM2026_SKIP_CRON_CLEANUP:-0}" = "1" ]; then return; fi
    crontab -l 2>/dev/null | grep -v 'DM2026-May14-0810' | crontab - 2>/dev/null || true
}
trap 'cleanup_cron; echo "[trap] exit at $(date -Iseconds), code=$?" >>"$LOG_FILE"' EXIT

exec 9>"$LOCK_FILE"
if ! flock -n 9; then exit 0; fi

echo "[$(date -Iseconds)] May14 0810 upload started" >>"$LOG_FILE"

SELECTED="$PRIMARY_CSV"
TAG="may14_h8v3_sslv2_weighted"
if [ ! -f "$SELECTED" ]; then
    SELECTED="$FALLBACK_CSV"
    TAG="may14_h8v1_fallback"
fi

"$ROOT/.venv/bin/python" -c "
import pandas as pd, sys
try:
    sub = pd.read_csv('$SELECTED')
    ref = pd.read_csv('$ROOT/data/raw/sample_submission.csv')
    assert list(sub.columns) == ['Id', 'Label']
    assert (sub['Id'].values == ref['Id'].values).all()
    assert sub['Label'].isin([0,1,2,3,4,5]).all()
    print(f'OK csv-format: {len(sub)} rows')
except Exception as e:
    print(f'FORMAT FAIL: {e}'); sys.exit(1)
" >>"$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    SELECTED="$FALLBACK_CSV"; TAG="may14_h8v1_emergency_fallback"
fi

SHA=$(sha256sum "$SELECTED" | awk '{print $1}')
echo "[$(date -Iseconds)] submitting $SELECTED tag=$TAG sha=$SHA" >>"$LOG_FILE"

for attempt in 1 2 3; do
    if [ "${DM2026_DRY_RUN:-0}" = "1" ]; then SUBMIT_RC=0; break; fi
    if "$ROOT/.venv/bin/kaggle" competitions submit \
        -c nycu-data-mining-assignment-3 -f "$SELECTED" -m "$TAG" >>"$LOG_FILE" 2>&1; then
        SUBMIT_RC=0; break
    fi
    sleep 30
    SUBMIT_RC=1
done

if [ "${SUBMIT_RC:-1}" -ne 0 ]; then
    echo "[$(date -Iseconds)] ERROR all submit attempts failed" >>"$LOG_FILE"; exit 1
fi
echo "[$(date -Iseconds)] upload complete" >>"$LOG_FILE"
