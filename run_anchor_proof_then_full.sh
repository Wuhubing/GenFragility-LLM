#!/bin/bash
# Orchestrator v2: proof → full30 (skip the 27-run smoke).
#
# Logic:
#   1. Run run_anchor_proof.sh (1 run: hub_3 + popularity_top25)
#      → validates new v3.3 anchor code path end-to-end (~5-10 min)
#   2. If PROOF_PASSED sentinel exists → run run_anchor_full30.sh (270 runs)
#   3. If PROOF_FAILED → stop, leave sentinel
#
# Logs:
#   logs/anchor_pipeline/proof.log
#   logs/anchor_pipeline/full30.log
#   logs/anchor_pipeline/STATUS  (append-only stage transitions)

set -u

ROOT=/home/weibing_wang/GenFragility-LLM
LOG_DIR="$ROOT/logs/anchor_pipeline"
ORCH_LOG="$LOG_DIR/orchestrator_v2.log"
PROOF_LOG="$LOG_DIR/proof.log"
FULL_LOG="$LOG_DIR/full30.log"
STATUS_FILE="$LOG_DIR/STATUS"

mkdir -p "$LOG_DIR"
cd "$ROOT"

status() { echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG" "$STATUS_FILE" > /dev/null; }

status "===== ORCHESTRATOR v2 START ====="

# ---- Stage 1: PROOF (1 run, ~5-10 min) ----
status "----- PROOF START (1 run: hub_3 + popularity_top25) -----"
PROOF_START=$(date +%s)

bash "$ROOT/run_anchor_proof.sh" > "$PROOF_LOG" 2>&1
PROOF_RC=$?
PROOF_MIN=$(( ($(date +%s) - PROOF_START) / 60 ))
status "Proof finished: exit=$PROOF_RC, duration=${PROOF_MIN} min"

if [ "$PROOF_RC" -ne 0 ] || [ ! -f "$LOG_DIR/PROOF_PASSED" ]; then
    status "PROOF FAILED — not launching full30. Inspect: $PROOF_LOG"
    touch "$LOG_DIR/PROOF_FAILED"
    exit 1
fi

status "PROOF PASSED — launching full30."

# ---- Stage 2: FULL-30 (270 runs, ~20-30h) ----
status "----- FULL30 START (270 runs) -----"
FULL_START=$(date +%s)

bash "$ROOT/run_anchor_full30.sh" > "$FULL_LOG" 2>&1
FULL_RC=$?
FULL_HOUR=$(awk "BEGIN { printf \"%.1f\", ($(date +%s) - $FULL_START) / 3600 }")

FULL_REPORTS=$(ls "$ROOT/main_output/Qwen3.5-9B_anchor_full30_experiment"/*/*/comparison_reports/*vllm*.json 2>/dev/null | wc -l)
status "Full30 finished: exit=$FULL_RC, duration=${FULL_HOUR}h, comparison_reports=${FULL_REPORTS}/270"

if [ "$FULL_RC" -ne 0 ] || [ "$FULL_REPORTS" -lt 270 ]; then
    status "FULL30 INCOMPLETE — inspect: $FULL_LOG"
    touch "$LOG_DIR/FULL30_INCOMPLETE"
    exit 1
fi

status "FULL30 COMPLETE — 270/270 vLLM reports written."
touch "$LOG_DIR/FULL30_COMPLETE"
status "===== ORCHESTRATOR v2 DONE ====="
