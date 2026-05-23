#!/bin/bash
# Orchestrator: queue smoke + full30 behind the currently-running 27B job.
# Designed to survive SSH disconnect (setsid + nohup + file redirects).
#
# Stages:
#   1. Wait for PID 2053566 (run_next_gen_pipeline.sh, 27B 30targets) to exit
#   2. Run smoke (run_anchor_smoke.sh) — 27 runs
#   3. If smoke logfile has no fatal error markers → run full30 (270 runs)
#   4. If smoke fails → stop, leave a sentinel file for inspection
#
# Logs:
#   logs/anchor_pipeline/orchestrator.log
#   logs/anchor_pipeline/smoke.log
#   logs/anchor_pipeline/full30.log
#   logs/anchor_pipeline/STATUS  (one-line status, append-only)

set -u

ROOT=/home/weibing_wang/GenFragility-LLM
LOG_DIR="$ROOT/logs/anchor_pipeline"
ORCH_LOG="$LOG_DIR/orchestrator.log"
SMOKE_LOG="$LOG_DIR/smoke.log"
FULL_LOG="$LOG_DIR/full30.log"
STATUS_FILE="$LOG_DIR/STATUS"

mkdir -p "$LOG_DIR"
cd "$ROOT"

log()    { echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"; }
status() { echo "[$(date '+%F %T')] $*" >> "$STATUS_FILE"; log "$*"; }

WAIT_PID=2053566   # currently-running 27B run_next_gen_pipeline.sh

status "===== ORCHESTRATOR START ====="
status "Waiting for PID $WAIT_PID (27B 30targets) to exit..."

# Poll for process exit (don't use `wait` — only works for own children)
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
status "PID $WAIT_PID exited. GPU should be free."

# Tiny grace period in case child vllm processes haven't released VRAM yet
sleep 30

# ------------------------------------------------------------------------
# Stage 1: SMOKE  (3 targets × 9 modes = 27 runs)
# ------------------------------------------------------------------------
status "----- SMOKE START (27 runs, expect ~2-3h) -----"
SMOKE_START=$(date +%s)

bash "$ROOT/run_anchor_smoke.sh" > "$SMOKE_LOG" 2>&1
SMOKE_RC=$?
SMOKE_END=$(date +%s)
SMOKE_MIN=$(( (SMOKE_END - SMOKE_START) / 60 ))

status "Smoke finished: exit=$SMOKE_RC, duration=${SMOKE_MIN} min"

# ------------------------------------------------------------------------
# Stage 2: ERROR CHECK
# ------------------------------------------------------------------------
# Look for fatal markers. Note: per-run "ERROR: LoRA not found" is the only
# inline error path that does *not* set $? non-zero (script uses set -e plus
# `|| true` on the LoRA glob), so we also grep for it.
SMOKE_REPORTS=$(ls "$ROOT/main_output/Qwen3.5-9B_anchor_smoke_experiment"/*/*/comparison_reports/*vllm*.json 2>/dev/null | wc -l)
SMOKE_FATAL=$(grep -cE "Traceback|CUDA out of memory|RuntimeError|ERROR: LoRA not found|❌ 训练失败|💥 训练异常" "$SMOKE_LOG" || true)

status "Smoke gate: exit=$SMOKE_RC, comparison_reports=${SMOKE_REPORTS}/27, fatal-grep-hits=${SMOKE_FATAL}"

# Pass = clean exit AND all 27 vLLM reports written AND no fatal grep hits.
if [ "$SMOKE_RC" -ne 0 ] || [ "$SMOKE_REPORTS" -lt 27 ] || [ "$SMOKE_FATAL" -gt 0 ]; then
    status "SMOKE FAILED — not launching full30."
    status "Inspect: $SMOKE_LOG  and  main_output/Qwen3.5-9B_anchor_smoke_experiment/"
    touch "$LOG_DIR/SMOKE_FAILED"
    exit 1
fi

status "SMOKE PASSED — launching full30."
touch "$LOG_DIR/SMOKE_PASSED"

# ------------------------------------------------------------------------
# Stage 3: FULL-30  (30 targets × 9 modes = 270 runs)
# ------------------------------------------------------------------------
status "----- FULL30 START (270 runs, expect ~20-30h) -----"
FULL_START=$(date +%s)

bash "$ROOT/run_anchor_full30.sh" > "$FULL_LOG" 2>&1
FULL_RC=$?
FULL_END=$(date +%s)
FULL_HOUR=$(awk "BEGIN { printf \"%.1f\", ($FULL_END - $FULL_START) / 3600 }")

FULL_REPORTS=$(ls "$ROOT/main_output/Qwen3.5-9B_anchor_full30_experiment"/*/*/comparison_reports/*vllm*.json 2>/dev/null | wc -l)

status "Full30 finished: exit=$FULL_RC, duration=${FULL_HOUR}h, comparison_reports=${FULL_REPORTS}/270"

if [ "$FULL_RC" -ne 0 ] || [ "$FULL_REPORTS" -lt 270 ]; then
    status "FULL30 INCOMPLETE — inspect: $FULL_LOG"
    touch "$LOG_DIR/FULL30_INCOMPLETE"
    exit 1
fi

status "FULL30 COMPLETE — 270/270 vLLM reports written."
touch "$LOG_DIR/FULL30_COMPLETE"
status "===== ORCHESTRATOR DONE ====="
