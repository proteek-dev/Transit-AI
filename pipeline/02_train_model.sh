#!/usr/bin/env bash
# pipeline/02_train_model.sh — clear stale model artifacts (local + S3) then
# force a real retrain via phase3/prediction.py. Run this after
# 01_rebuild_features.sh has produced a fresh feature snapshot.
#
# Usage:
#   nohup bash pipeline/02_train_model.sh > /dev/null 2>&1 &
#   tail -f logs/train_model_latest.log

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p logs
LOG="logs/train_model_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/train_model_latest.log   # stable name to tail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

heartbeat() {
    local pid="$1" label="$2" start elapsed
    start=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        kill -0 "$pid" 2>/dev/null || break
        elapsed=$(( ($(date +%s) - start) / 60 ))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [HEARTBEAT] $label still running (${elapsed} min elapsed)" >> "$LOG"
    done
}

fail() { log "❌ FAILED: $1 — see $LOG. Aborting."; exit 1; }

log "=========================================="
log "Train model started. Logging to: $LOG"
log "=========================================="

# ── Step 1/2: clear stale model artifacts (forces a real retrain) ───────────
log "STEP 1/2: Clearing old model artifacts"

# rm -rf treats a missing directory as success (that's what -f means), so
# this is already idempotent whether or not phase3/model/ exists locally.
rm -rf phase3/model/ && log "  Cleared local phase3/model/"

python3 -c "
import sys
sys.path.insert(0, 'phase3')
import config

# Same credential resolution as phase3/config.py / phase3/prediction.py --
# Streamlit secrets first, then env vars, then .env fallback -- instead of
# a bare s3fs.S3FileSystem(), which only picks up credentials already
# exported in the shell environment. That's why this step used to throw
# botocore.exceptions.NoCredentialsError under nohup on EC2, where AWS
# creds live only in .env, not the shell environment.
fs = config.get_s3_filesystem()
bucket = config.get_s3_bucket()
prefix = f'{bucket}/phase3/model/'

# fs.rm() raises FileNotFoundError if the prefix doesn't exist, rather than
# treating 'nothing to delete' as success -- a legitimate state whenever a
# previous run's retrain step failed after this clearing step already ran.
# Check first and skip the rm() call entirely if there's nothing there.
if fs.exists(prefix):
    fs.rm(prefix, recursive=True)
    print(f'Cleared S3 model artifacts at s3://{prefix}')
else:
    print(f'S3 model artifacts already absent at s3://{prefix} -- nothing to clear')
" >> "$LOG" 2>&1 || fail "clearing S3 model artifacts"

log "STEP 1/2 complete."

# ── Step 2/2: retrain ─────────────────────────────────────────────────────────
log "STEP 2/2: Retraining model (phase3/prediction.py)"

python3 -u phase3/prediction.py >> "$LOG" 2>&1 &
RETRAIN_PID=$!
heartbeat "$RETRAIN_PID" "Retrain" &
HB_PID=$!

wait "$RETRAIN_PID"; RETRAIN_EXIT=$?
kill "$HB_PID" 2>/dev/null

[ "$RETRAIN_EXIT" -eq 0 ] || fail "model retrain (exit code $RETRAIN_EXIT)"
log "STEP 2/2 complete: model retrained and uploaded to S3."

log "=========================================="
log "✅ TRAIN MODEL COMPLETE."
log "Manual step still required: reboot the Streamlit Cloud app —"
log "st.cache_resource won't notice the new S3 model on its own."
log "=========================================="
