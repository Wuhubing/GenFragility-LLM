#!/bin/bash
# Phase D resume only — Phases A/B/C already completed successfully.
# The original orchestrator died on `PYTHONPATH: unbound variable` because of `set -u`.
# This restarts only the 27B pipeline portion.

set -o pipefail

cd /home/weibing_wang/GenFragility-LLM

LOG=/home/weibing_wang/GenFragility-LLM/logs/auto_27b_resume_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec >> "$LOG" 2>&1

echo "=========================================================="
echo " Phase D resume START at $(date)"
echo " Log: $LOG"
echo "=========================================================="

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:${PYTHONPATH:-}
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

echo " - 2B (45/45 done)  → will skip"
echo " - 9B (45/45 done)  → will skip"
echo " - 27B: hub_1 ✓ tail_1 ✓ → will run hub_2, random_1, then 41 more (~24-30h)"
echo ""

bash run_next_gen_pipeline.sh

echo "=========================================================="
echo " Phase D END at $(date)"
echo "=========================================================="
