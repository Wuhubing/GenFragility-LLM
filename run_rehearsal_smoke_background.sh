#!/usr/bin/env bash
# Start the rehearsal smoke runner in the background or inspect its status.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG_DIR=logs/rehearsal_smoke
PID_FILE="$LOG_DIR/background.pid"
LATEST_LOG_FILE="$LOG_DIR/latest_log"
ACTION=${1:-start}

if [[ "$ACTION" == "status" ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No background run is registered."
        exit 1
    fi
    pid=$(<"$PID_FILE")
    log_path=$(<"$LATEST_LOG_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "RUNNING pid=$pid log=$log_path"
        exit 0
    fi
    echo "STOPPED pid=$pid log=$log_path"
    exit 1
fi

if [[ "$ACTION" != "start" ]]; then
    echo "Usage: $0 [start [runner arguments...] | status]"
    exit 1
fi
shift || true

mkdir -p "$LOG_DIR"
if [[ -f "$PID_FILE" ]]; then
    old_pid=$(<"$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "A rehearsal smoke run is already active: pid=$old_pid"
        exit 1
    fi
fi

timestamp=$(date +%Y%m%d_%H%M%S)
log_path="$LOG_DIR/rehearsal_smoke_${timestamp}.log"
nohup "$ROOT/run_rehearsal_smoke.sh" "$@" >"$log_path" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
printf '%s\n' "$log_path" > "$LATEST_LOG_FILE"
echo "STARTED pid=$pid log=$log_path"
