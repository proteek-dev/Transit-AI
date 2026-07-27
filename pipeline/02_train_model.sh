#!/usr/bin/env bash
# pipeline/02_train_model.sh — trains a new model into an isolated staging
# location, then promotes it to both a permanent dated run folder and the
# live latest/ pointer (local + S3), pruning older run history down to one
# run per past day. Run this after 01_rebuild_features.sh has produced a
# fresh feature snapshot.
#
# Never deletes/overwrites phase3/model/latest/ up front -- if training
# fails, latest/ and runs/ are left completely untouched and only the
# abandoned staging folder exists (same staging-then-promote pattern as
# scripts/archive_gtfsrt.py's static-snapshot upload).
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

fail() { log "❌ FAILED: $1 — see $LOG. Aborting. runs/ and latest/ are untouched; an abandoned phase3/model/${STAGING_SUBDIR:-_staging_*}/ folder may remain and can be deleted."; exit 1; }

log "=========================================="
log "Train model started. Logging to: $LOG"
log "=========================================="

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

# ── Step 2/3: train into staging -- latest/ and runs/ are never touched here ─
log "STEP 2/3: Training model into staging (phase3/prediction.py)"

MODEL_SUBDIR_OVERRIDE="$STAGING_SUBDIR" python3 -u phase3/prediction.py >> "$LOG" 2>&1 &
RETRAIN_PID=$!
heartbeat "$RETRAIN_PID" "Retrain" &
HB_PID=$!

wait "$RETRAIN_PID"; RETRAIN_EXIT=$?
kill "$HB_PID" 2>/dev/null

[ "$RETRAIN_EXIT" -eq 0 ] || fail "model retrain (exit code $RETRAIN_EXIT)"

for f in xgb_v0.json categories.joblib training_metadata.json; do
    [ -f "${STAGING_DIR}/${f}" ] || fail "training reported success but staging is missing ${f}"
done
log "STEP 2/3 complete: model trained, all 3 artifact files confirmed present in staging."

# ── Step 3/3: promote staging -> runs/{date}/{time}/ + latest/, then prune ──
log "STEP 3/3: Promoting staging to runs/ and latest/, pruning older-day run history"

DATE_STR="$DATE_STR" TIME_STR="$TIME_STR" STAGING_SUBDIR="$STAGING_SUBDIR" python3 -c "
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, 'phase3')
import config

date_str = os.environ['DATE_STR']
time_str = os.environ['TIME_STR']
staging_subdir = os.environ['STAGING_SUBDIR']
bucket = config.get_s3_bucket()
fs = config.get_s3_filesystem()

ARTIFACTS = ['xgb_v0.json', 'categories.joblib', 'training_metadata.json']

# --- local: staging -> runs/{date}/{time}/ and latest/ ---
local_staging = Path('phase3/model') / staging_subdir
local_run = Path('phase3/model/runs') / date_str / time_str
local_latest = Path('phase3/model/latest')

local_run.mkdir(parents=True, exist_ok=True)
local_latest.mkdir(parents=True, exist_ok=True)
for name in ARTIFACTS:
    shutil.copy2(local_staging / name, local_run / name)
    shutil.copy2(local_staging / name, local_latest / name)
print(f'Copied staging -> {local_run}/ and {local_latest}/ (local)')

shutil.rmtree(local_staging)
print(f'Deleted local staging: {local_staging}/')

# --- S3: staging -> runs/{date}/{time}/ and latest/ ---
s3_staging_prefix = f'{bucket}/phase3/model/{staging_subdir}'
s3_run_prefix = f'{bucket}/phase3/model/runs/{date_str}/{time_str}'
s3_latest_prefix = f'{bucket}/phase3/model/latest'

for name in ARTIFACTS:
    src = f's3://{s3_staging_prefix}/{name}'
    fs.copy(src, f's3://{s3_run_prefix}/{name}')
    fs.copy(src, f's3://{s3_latest_prefix}/{name}')
print(f'Copied staging -> s3://{s3_run_prefix}/ and s3://{s3_latest_prefix}/ (S3)')

fs.rm(s3_staging_prefix, recursive=True)
print(f'Deleted S3 staging: s3://{s3_staging_prefix}/')

# --- retention: for every runs/ date folder that is NOT today, keep only the
# latest time-folder and delete the rest. Today's date folder is never
# touched here, no matter how many runs it already has. ---
def prune(dates, times_fn, delete_fn, label):
    for d in dates:
        if d == date_str:
            continue
        times = sorted(times_fn(d))
        if len(times) > 1:
            keep = times[-1]
            for t in times[:-1]:
                delete_fn(d, t)
                print(f'  [{label}] pruned runs/{d}/{t}/ (kept runs/{d}/{keep}/)')

local_runs_root = Path('phase3/model/runs')
local_dates = sorted(p.name for p in local_runs_root.iterdir() if p.is_dir())
prune(
    local_dates,
    lambda d: [p.name for p in (local_runs_root / d).iterdir() if p.is_dir()],
    lambda d, t: shutil.rmtree(local_runs_root / d / t),
    'local',
)

s3_runs_root = f'{bucket}/phase3/model/runs'
if fs.exists(s3_runs_root):
    s3_dates = sorted(e.rstrip('/').split('/')[-1] for e in fs.ls(s3_runs_root))
    prune(
        s3_dates,
        lambda d: [e.rstrip('/').split('/')[-1] for e in fs.ls(f'{s3_runs_root}/{d}')],
        lambda d, t: fs.rm(f'{s3_runs_root}/{d}/{t}', recursive=True),
        'S3',
    )
" >> "$LOG" 2>&1 || fail "promoting staging to runs/latest and pruning older-day history"

log "STEP 3/3 complete: model promoted to runs/${DATE_STR}/${TIME_STR}/ and latest/, older-day history pruned."

log "=========================================="
log "✅ TRAIN MODEL COMPLETE."
log "Manual step still required: reboot the Streamlit Cloud app —"
log "st.cache_resource won't notice the new S3 model on its own."
log "=========================================="
