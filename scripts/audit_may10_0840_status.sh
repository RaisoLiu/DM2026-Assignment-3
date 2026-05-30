#!/usr/bin/env bash
set -euo pipefail

REPO="/home/raiso/playground/DM2026-Assignment-3"
export HOME="/home/raiso"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
cd "$REPO"

LOG="artifacts/blend_search/audit_may10_0840.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

cleanup() {
  local status=$?
  echo "cleanup status=${status} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  if [[ "${DM2026_SKIP_CRON_CLEANUP:-0}" == "1" ]]; then
    echo "Skipping cron cleanup because DM2026_SKIP_CRON_CLEANUP=1"
  elif command -v crontab >/dev/null 2>&1; then
    crontab -l 2>/dev/null | grep -v "DM2026-May10-0840-Audit" | crontab - || true
  fi
  exit "$status"
}
trap cleanup EXIT

echo "=== DM2026 May 10 08:40 audit started at $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

KAGGLE="$REPO/.venv/bin/kaggle"
COMPETITION="nycu-data-mining-assignment-3"
STATUS_CSV="artifacts/blend_search/upload_may10_0810_status.csv"
STATUS_TXT="artifacts/blend_search/upload_may10_0810_status.txt"

if [[ ! -x "$KAGGLE" ]]; then
  echo "Kaggle CLI is not executable at ${KAGGLE}" >&2
  exit 1
fi

"$KAGGLE" competitions submissions -c "$COMPETITION" --csv >"$STATUS_CSV"
"$KAGGLE" competitions submissions -c "$COMPETITION" >"$STATUS_TXT"
head -n 15 "$STATUS_TXT"

"$REPO/.venv/bin/python" scripts/audit_may10_upload.py || true

echo "=== DM2026 May 10 08:40 audit finished at $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
