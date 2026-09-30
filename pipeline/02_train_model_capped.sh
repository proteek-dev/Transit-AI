#!/usr/bin/env bash
# pipeline/02_train_model_capped.sh — MEMORY-TEST fork of 02_train_model.sh.
# Trains into an isolated local staging folder with row sampling and an RSS
# cap, reports MAE + peak RSS from the run's training_metadata.json, then
# STOPS. No side effects: never promotes to runs/ or latest/, and never
# writes to S3 (TRAIN_SKIP_S3_UPLOAD=1 skips training's own staging upload).
#
# Usage:
#   nohup bash pipeline/02_train_model_capped.sh > /dev/null 2>&1 &
#   tail -f logs/train_model_capped_latest.log
#
# Defaults can be overridden per run, e.g.:
#   TRAIN_SAMPLE_FRAC=0.2 bash pipeline/02_train_model_capped.sh

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults are tuned for an 8 GB Mac:
#   TRAIN_SAMPLE_FRAC=0.1 → ~15M rows through concat/cast/copy, first survivable size
#   TRAIN_RSS_CAP_GB=5    → 8 GB total RAM leaves ~4-5 GB for the trainer before
#                           the OS starts swapping; 5 GB is the "cut it before
#                           the Mac locks up" threshold, not the OOM ceiling.
# Ratchet TRAIN_SAMPLE_FRAC up on subsequent runs if the previous one survived
# with headroom (0.2, 0.3, 0.5 …). Raise TRAIN_RSS_CAP_GB only if you move
# this to a machine with more RAM.
export TRAIN_SAMPLE_FRAC="${TRAIN_SAMPLE_FRAC:-0.1}"
export TRAIN_RSS_CAP_GB="${TRAIN_RSS_CAP_GB:-5}"
export TRAIN_SKIP_S3_UPLOAD=1

mkdir -p logs
LOG="logs/train_model_capped_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/train_model_capped_latest.log   # stable name to tail

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

# Same as 02_train_model.sh's rss_watchdog().
rss_watchdog() {
    local pid="$1" cap_gb="$2" rss_kb rss_gb
    while kill -0 "$pid" 2>/dev/null; do
        sleep 5
        rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "$rss_kb" ] || return 0   # process finished between checks
        # ps reports RSS in KiB; 1048576 KiB = 1 GiB.
        if awk -v kb="$rss_kb" -v cap="$cap_gb" 'BEGIN { exit !(kb / 1048576 > cap) }'; then
            rss_gb=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.2f", kb / 1048576 }')
            log "RSS CAP HIT: ${rss_gb}GB > ${cap_gb}GB, killing"
            kill -TERM "$pid" 2>/dev/null
            sleep 10
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
            return 0
        fi
    done
}

fail() { log "❌ FAILED: $1 — see $LOG. Aborting. runs/, latest/ and S3 are untouched; phase3/model/${STAGING_SUBDIR:-_staging_*}/ may remain locally and can be deleted."; exit 1; }

log "=========================================="
log "MEMORY-TEST RUN — this model will NOT be promoted."
log "  TRAIN_SAMPLE_FRAC=${TRAIN_SAMPLE_FRAC}  TRAIN_RSS_CAP_GB=${TRAIN_RSS_CAP_GB}"
log "  Hardware: tuned for 8 GB Mac. Close Chrome and heavy apps before running."
log "  Artifacts stay in a local staging folder; runs/, latest/ and S3 are not touched."
log "  Logging to: $LOG"
log "=========================================="

[[ "$TRAIN_RSS_CAP_GB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "TRAIN_RSS_CAP_GB must be a number of GB, got '${TRAIN_RSS_CAP_GB}'"

# ── Step 1/3: determine run_id from current Brisbane date/time ──────────────
log "STEP 1/3: Determining run_id (Australia/Brisbane date/time)"

read -r DATE_STR TIME_STR <<< "$(python3 -c "
from datetime import datetime
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo('Australia/Brisbane'))
print(now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
")"
STAGING_SUBDIR="_staging_${DATE_STR}_${TIME_STR}"
STAGING_DIR="phase3/model/${STAGING_SUBDIR}"

log "  run_id: date=${DATE_STR} time=${TIME_STR}"
log "  staging: ${STAGING_DIR}"

# ── Step 2/3: train into staging, under the RSS watchdog ─────────────────────
# MODEL_SUBDIR_OVERRIDE is load-bearing: without it MODEL_SUBDIR defaults to
# 'latest' and training would write straight into phase3/model/latest/.
log "STEP 2/3: Training model into staging (phase3/prediction.py)"

MODEL_SUBDIR_OVERRIDE="$STAGING_SUBDIR" python3 -u phase3/prediction.py >> "$LOG" 2>&1 &
RETRAIN_PID=$!
heartbeat "$RETRAIN_PID" "Retrain" &
HB_PID=$!
rss_watchdog "$RETRAIN_PID" "$TRAIN_RSS_CAP_GB" &
WD_PID=$!

wait "$RETRAIN_PID"; RETRAIN_EXIT=$?
kill "$HB_PID" "$WD_PID" 2>/dev/null

[ "$RETRAIN_EXIT" -eq 0 ] || fail "model retrain (exit code $RETRAIN_EXIT)"

for f in xgb_v0.json categories.joblib training_metadata.json; do
    [ -f "${STAGING_DIR}/${f}" ] || fail "training reported success but staging is missing ${f}"
done
log "STEP 2/3 complete: model trained, all 3 artifact files confirmed present in staging."

# ── Step 3/3: report MAE + RSS, then stop (no promotion) ─────────────────────
log "STEP 3/3: Results from ${STAGING_DIR}/training_metadata.json"

STAGING_DIR="$STAGING_DIR" python3 -c "
import json, os
with open(os.path.join(os.environ['STAGING_DIR'], 'training_metadata.json')) as f:
    m = json.load(f)
print(f\"  rows: loaded={m['rows_loaded']:,}  after sample={m['rows_after_sample_total']:,}\")
print(f\"  MAE:  train={m['train_mae']:.3f}  test={m['test_mae']:.3f}  naive={m['naive_mae']:.3f} min \"
      f\"({m['pct_improvement_over_naive']:.1f}% better than naive)\")
print(f\"  RSS:  peak={m['peak_rss_mb']:,.0f} MB  at cast={m['rss_at_categorical_cast_mb']:,.0f}  \"
      f\"X_train={m['rss_at_x_train_build_mb']:,.0f}  X_test={m['rss_at_x_test_build_mb']:,.0f}  \"
      f\"after fit={m['rss_after_fit_mb']:,.0f} MB\")
" 2>&1 | tee -a "$LOG" || fail "reading training_metadata.json from staging"

log "=========================================="
log "✅ MEMORY-TEST RUN COMPLETE — model NOT promoted."
log "Artifacts left in ${STAGING_DIR}/ for inspection; delete when done."
log "=========================================="
