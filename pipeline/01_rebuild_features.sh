#!/usr/bin/env bash
# pipeline/01_rebuild_features.sh — rebuild the ML feature snapshot by
# running notebook 05 via papermill. Heartbeats every 60s so it's never
# silent on a long historical reprocess.
#
# Usage:
#   nohup bash pipeline/01_rebuild_features.sh > /dev/null 2>&1 &
#   tail -f logs/rebuild_features_latest.log

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p logs
LOG="logs/rebuild_features_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/rebuild_features_latest.log   # stable name to tail

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
log "Rebuild features started. Logging to: $LOG"
log "=========================================="

log "Rebuilding feature snapshot (notebook 05)"

python3 -u -m papermill \
    notebooks/05_phase2_feature_pipeline.ipynb \
    notebooks/05_phase2_feature_pipeline.ipynb \
    --log-output >> "$LOG" 2>&1 &
NB05_PID=$!
heartbeat "$NB05_PID" "Rebuild features (notebook 05)" &
HB_PID=$!

wait "$NB05_PID"; NB05_EXIT=$?
kill "$HB_PID" 2>/dev/null

[ "$NB05_EXIT" -eq 0 ] || fail "notebook 05 execution (exit code $NB05_EXIT)"
grep -q "Wrote manifest" "$LOG" || fail "notebook 05 finished but manifest write not confirmed — check log"

log "=========================================="
log "✅ Feature snapshot rebuilt, manifest verified."
log "Next: pipeline/02_train_model.sh"
log "=========================================="
