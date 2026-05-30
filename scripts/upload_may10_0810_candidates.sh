#!/usr/bin/env bash
set -euo pipefail

REPO="/home/raiso/playground/DM2026-Assignment-3"
export HOME="/home/raiso"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
cd "$REPO"

LOG="artifacts/blend_search/upload_may10_0810.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
LOCK_FILE="artifacts/blend_search/upload_may10_0810.lock"
STATUS_CSV="artifacts/blend_search/upload_may10_0810_status.csv"
STATUS_TXT="artifacts/blend_search/upload_may10_0810_status.txt"
DRY_RUN="${DM2026_DRY_RUN:-0}"
SKIP_CRON_CLEANUP="${DM2026_SKIP_CRON_CLEANUP:-0}"
if [[ "$DRY_RUN" == "1" ]]; then
  RUN_MANIFEST="artifacts/blend_search/upload_may10_0810_dryrun_submitted.tsv"
else
  RUN_MANIFEST="artifacts/blend_search/upload_may10_0810_submitted.tsv"
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another May 10 upload process is already running; exiting at $(date '+%Y-%m-%d %H:%M:%S %Z')" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && -s "$RUN_MANIFEST" ]]; then
  echo "Existing non-empty manifest ${RUN_MANIFEST}; refusing to rerun and risk duplicate submissions." >&2
  echo "Review the manifest and Kaggle status manually before any recovery action." >&2
  exit 1
fi
: >"$RUN_MANIFEST"

cleanup() {
  local status=$?
  echo "cleanup status=${status} at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  if [[ "$status" != "0" && "$DRY_RUN" != "1" && -s "$RUN_MANIFEST" && -n "${KAGGLE:-}" && -x "${KAGGLE:-}" ]]; then
    echo "Attempting failure-path Kaggle status snapshot and audit."
    "$KAGGLE" competitions submissions -c "${COMPETITION:-nycu-data-mining-assignment-3}" --csv >"$STATUS_CSV" || true
    "$REPO/.venv/bin/python" scripts/audit_may10_upload.py || true
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Skipping cron cleanup because DM2026_DRY_RUN=1"
  elif [[ "$SKIP_CRON_CLEANUP" == "1" ]]; then
    echo "Skipping cron cleanup because DM2026_SKIP_CRON_CLEANUP=1"
  elif command -v crontab >/dev/null 2>&1; then
    crontab -l 2>/dev/null | grep -v "DM2026-May10-0810" | crontab - || true
  fi
  exit "$status"
}
trap cleanup EXIT

echo "=== DM2026 May 10 08:10 upload started at $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

KAGGLE="$REPO/.venv/bin/kaggle"
COMPETITION="nycu-data-mining-assignment-3"
CANDIDATES=(
  "submissions/submission_consensus_oof_greedy.csv"
  "submissions/submission_consensus_oof_greedy_threshold3.csv"
  "submissions/submission_consensus_oof_alt_threshold4.csv"
  "submissions/submission_consensus_rare_b.csv"
)

if [[ ! -x "$KAGGLE" ]]; then
  echo "Kaggle CLI is not executable at ${KAGGLE}" >&2
  exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN enabled: no Kaggle submissions will be made."
fi

echo "Regenerating May 10 candidate CSVs before upload."
"$REPO/.venv/bin/python" scripts/make_consensus_candidates.py

"$REPO/.venv/bin/python" - "${CANDIDATES[@]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

repo = Path("/home/raiso/playground/DM2026-Assignment-3")
sample = pd.read_csv(repo / "data/raw/sample_submission.csv")
for raw_path in sys.argv[1:]:
    path = repo / raw_path
    if not path.exists():
        raise SystemExit(f"Missing candidate CSV: {raw_path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["Id", "Label"]:
        raise SystemExit(f"Invalid columns in {raw_path}: {frame.columns.tolist()}")
    if len(frame) != len(sample):
        raise SystemExit(f"Invalid row count in {raw_path}: {len(frame)} != {len(sample)}")
    if not frame["Id"].equals(sample["Id"]):
        raise SystemExit(f"Id order mismatch in {raw_path}")
    labels = set(frame["Label"].astype(int).unique())
    if not labels.issubset(set(range(6))):
        raise SystemExit(f"Invalid labels in {raw_path}: {sorted(labels)}")
print(f"Validated {len(sys.argv) - 1} candidate CSVs")
PY

submit_one() {
  local csv="$1"
  local message="$2"
  echo "--- submitting ${csv} as ${message} at $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >&2
  submit_with_retry "$csv" "$message"
  printf '%s\t%s\n' "$csv" "$message" >>"$RUN_MANIFEST"
  wait_for_score "$(basename "$csv")" "$message"
}

submit_with_retry() {
  local csv="$1"
  local message="$2"
  local attempt
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN submit ${csv} as ${message}" >&2
    return 0
  fi
  for attempt in 1 2 3; do
    if "$KAGGLE" competitions submit -c "$COMPETITION" -f "$csv" -m "$message" >&2; then
      return 0
    fi
    echo "submit attempt ${attempt} failed for ${csv}" >&2
    sleep 30
  done
  echo "all submit attempts failed for ${csv}" >&2
  return 1
}

wait_for_score() {
  local filename="$1"
  local message="$2"
  local score=""
  if [[ "$DRY_RUN" == "1" ]]; then
    case "$message" in
      may10_consensus_oof_greedy)
        score="${DM2026_FAKE_SCORE_GREEDY:-0.8120}"
        ;;
      may10_consensus_oof_greedy_threshold3)
        score="${DM2026_FAKE_SCORE_THRESHOLD3:-0.8120}"
        ;;
      may10_consensus_oof_alt_threshold4)
        score="${DM2026_FAKE_SCORE_ALT_THRESHOLD4:-0.8120}"
        ;;
      may10_consensus_rare_b)
        score="${DM2026_FAKE_SCORE_RARE_B:-0.8120}"
        ;;
      *)
        score="0.8120"
        ;;
    esac
    echo "DRY RUN score ${filename} ${score}" >&2
    printf '%s' "$score"
    return 0
  fi
  for _ in $(seq 1 20); do
    sleep 20
    score="$("$KAGGLE" competitions submissions -c "$COMPETITION" --csv | "$REPO/.venv/bin/python" -c 'import csv, sys
filename = sys.argv[1]
message = sys.argv[2]
for row in csv.DictReader(sys.stdin):
    status = (row.get("status") or "").lower()
    if row.get("fileName") == filename and row.get("description") == message and "complete" in status:
        print(row.get("publicScore", ""))
        break
' "$filename" "$message")"
    if [[ -n "$score" ]]; then
      echo "score ${filename} ${score}" >&2
      printf '%s' "$score"
      return 0
    fi
    echo "waiting for ${filename} score at $(date '+%Y-%m-%d %H:%M:%S %Z')" >&2
  done
  echo "score ${filename} unavailable after polling" >&2
  printf 'nan'
}

score_ge() {
  "$REPO/.venv/bin/python" - "$1" "$2" <<'PY'
import math
import sys

try:
    left = float(sys.argv[1])
    right = float(sys.argv[2])
except ValueError:
    sys.exit(1)
sys.exit(0 if left >= right and not math.isnan(left) else 1)
PY
}

score_gt() {
  "$REPO/.venv/bin/python" - "$1" "$2" <<'PY'
import math
import sys

try:
    left = float(sys.argv[1])
    right = float(sys.argv[2])
except ValueError:
    sys.exit(1)
sys.exit(0 if left > right and not math.isnan(left) else 1)
PY
}

score1="$(submit_one "submissions/submission_consensus_oof_greedy.csv" "may10_consensus_oof_greedy")"
echo
echo "first candidate score=${score1}"

echo "Second candidate is the highest-OOF controlled branch."
score2="$(submit_one "submissions/submission_consensus_oof_greedy_threshold3.csv" "may10_consensus_oof_greedy_threshold3")"
echo
echo "second candidate score=${score2}"

if score_ge "$score2" "0.8130"; then
  echo "Threshold branch matched or beat current best; testing the alternate controlled threshold-family variant."
  submit_one "submissions/submission_consensus_oof_alt_threshold4.csv" "may10_consensus_oof_alt_threshold4"
elif score_ge "$score1" "0.8130"; then
  echo "Greedy consensus matched or beat current best but threshold did not; testing the safer rare-target branch."
  submit_one "submissions/submission_consensus_rare_b.csv" "may10_consensus_rare_b"
elif score_gt "$score2" "$score1"; then
  echo "Threshold branch beat the first candidate but remains below current best; testing the alternate controlled threshold-family variant."
  submit_one "submissions/submission_consensus_oof_alt_threshold4.csv" "may10_consensus_oof_alt_threshold4"
else
  echo "Neither high-OOF branch beat current best; using the safest LGBM/Mini rare-target branch."
  submit_one "submissions/submission_consensus_rare_b.csv" "may10_consensus_rare_b"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "--- dry-run submitted manifest ---"
  cat "$RUN_MANIFEST"
  echo "=== DM2026 May 10 dry-run finished at $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  exit 0
fi

echo "--- submission status after uploads ---"
"$KAGGLE" competitions submissions -c "$COMPETITION" --csv >"$STATUS_CSV"
"$KAGGLE" competitions submissions -c "$COMPETITION" >"$STATUS_TXT"
head -n 15 "$STATUS_TXT"
"$REPO/.venv/bin/python" - <<'PY'
from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

repo = Path("/home/raiso/playground/DM2026-Assignment-3")
manifest = repo / "artifacts/blend_search/upload_may10_0810_submitted.tsv"
status_csv = repo / "artifacts/blend_search/upload_may10_0810_status.csv"
log_path = repo / "SUBMISSION_LOG.md"
metadata = repo / "artifacts/blend_search/consensus_candidates_may10.json"

submitted: list[tuple[str, str]] = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    csv_path, message = line.split("\t", 1)
    submitted.append((csv_path, message))

status_rows = list(csv.DictReader(status_csv.read_text(encoding="utf-8").splitlines()))
by_file = {row["fileName"]: row for row in status_rows}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def counts(path: Path) -> dict[int, int]:
    frame = pd.read_csv(path)
    return {int(k): int(v) for k, v in frame["Label"].value_counts().sort_index().items()}

def normalize_status(raw: str) -> str:
    status = (raw or "").split(".")[-1].upper()
    return status

lines = log_path.read_text(encoding="utf-8").splitlines()
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
for idx, line in enumerate(lines):
    if line.startswith("Last updated:"):
        lines[idx] = f"Last updated: `{now}`"
        break

existing = "\n".join(lines)
new_rows: list[str] = []
for csv_path, message in submitted:
    rel = Path(csv_path)
    filename = rel.name
    if f"`{csv_path}`" in existing:
        continue
    row = by_file.get(filename, {})
    timestamp = row.get("date") or ""
    status = normalize_status(row.get("status") or "")
    public = row.get("publicScore") or ""
    private = row.get("privateScore") or ""
    digest = sha256_file(repo / rel)
    label_counts = counts(repo / rel)
    note = (
        "Scheduled adaptive May 10 consensus upload generated by "
        "`scripts/make_consensus_candidates.py` and submitted by "
        "`scripts/upload_may10_0810_candidates.sh`; "
        f"label counts `{label_counts}`. Current best before this run was "
        "`submission_centered_meta_viterbi_oof07693.csv` at public 0.8130. "
        "Some branches include the restricted current-best threshold hybrid."
    )
    new_rows.append(
        f"| {timestamp} | `{csv_path}` | `{message}` | {status} | {public} | {private} | `{digest}` | {note} |"
    )

if new_rows:
    insert_at = None
    for idx, line in enumerate(lines):
        if line.startswith("|---"):
            insert_at = idx + 1
            break
    if insert_at is None:
        raise RuntimeError("Could not find SUBMISSION_LOG table separator")
    lines[insert_at:insert_at] = new_rows
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Updated {log_path} with {len(new_rows)} scheduled submission row(s)")
PY
"$REPO/.venv/bin/python" scripts/audit_may10_upload.py || true
echo "=== DM2026 May 10 upload finished at $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
