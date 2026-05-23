#!/bin/bash
# Auto-finish-and-resume orchestrator
# Phase A: For each of the 3 remaining yuji targets (tesla/actblz/messi):
#           wait for *_vllm_comparison.json → run LLM-judge → re-render HTML
# Phase B: Final render with all 6 cards
# Phase C: Cleanup partial 27B/random_1 (only has checkpoint-200, no adapter)
# Phase D: Resume `run_next_gen_pipeline.sh` — its skip-if-done logic will
#          pass through 2B (45/45 done), 9B (45/45 done) and pick up at 27B
#          (hub_2, random_1, plus 41 remaining targets).
#
# Designed to run under nohup so it survives session exit.

set -o pipefail  # NOT -e (keep going on judge failure), NOT -u (env vars may be undefined)

cd /home/weibing_wang/GenFragility-LLM

LOG=/home/weibing_wang/GenFragility-LLM/logs/auto_finish_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec >> "$LOG" 2>&1

CONDA=/home/weibing_wang/miniconda3/bin/conda

echo "=========================================================="
echo " Orchestrator START at $(date)"
echo " Log: $LOG"
echo "=========================================================="

REMAINING=(yuji_tesla_hq yuji_actblz_parent yuji_messi_club)

# ------- Phase A: per-target wait → judge → render -------
for tid in "${REMAINING[@]}"; do
    REPORT="main_output/Qwen3.5-9B_yuji_experiment/$tid/comparison_reports/${tid}_vllm_comparison.json"
    JUDGED="main_output/Qwen3.5-9B_yuji_experiment/$tid/comparison_reports/${tid}_vllm_comparison_judged.json"

    echo "----------------------------------------------------------"
    echo " [Phase A] target: $tid  @ $(date +%H:%M:%S)"
    echo "----------------------------------------------------------"

    # 1) wait until vllm comparison report exists (Phase 2 done)
    if [ ! -f "$REPORT" ]; then
        echo "  waiting for $REPORT ..."
        while [ ! -f "$REPORT" ]; do
            sleep 30
        done
    fi
    echo "  ✓ report ready"

    # 2) run judge (skip if already done)
    if [ -f "$JUDGED" ]; then
        echo "  ✓ judge already done — skipping"
    else
        echo "  → running LLM-judge ..."
        $CONDA run -n genfragility --no-capture-output python scripts/llm_judge_comparison_report.py \
            "$REPORT" --concurrency 30
        if [ $? -ne 0 ]; then
            echo "  ✗ judge FAILED for $tid (continuing)"
        else
            echo "  ✓ judge done"
        fi
    fi

    # 3) re-render HTML (cumulative; will include all targets done so far)
    echo "  → rendering HTML ..."
    python3 scripts/render_yuji_html.py
    echo "  ✓ HTML updated"
done

# ------- Phase B: final render (sanity, should be identical to last loop iter) -------
echo "=========================================================="
echo " [Phase B] Final HTML render @ $(date +%H:%M:%S)"
echo "=========================================================="
python3 scripts/render_yuji_html.py
echo "✓ Yuji 6-card SHORTLIST complete: docs/illustration_examples/SHORTLIST_yuji_v1.html"

# ------- Phase C: cleanup partial 27B/random_1 -------
echo "=========================================================="
echo " [Phase C] Cleanup half-trained 27B/random_1 @ $(date +%H:%M:%S)"
echo "=========================================================="
PARTIAL_DIR="main_output/Qwen3.6-27B_30targets_experiment/random_1/random_1_20260521_141327"
if [ -d "$PARTIAL_DIR" ]; then
    # Move it aside (don't delete — preserves the SIGTERM checkpoint-200 for forensics)
    mv "$PARTIAL_DIR" "${PARTIAL_DIR}.SIGTERM_PARTIAL"
    echo "  ✓ moved $PARTIAL_DIR → ${PARTIAL_DIR}.SIGTERM_PARTIAL"
else
    echo "  (nothing to clean — already moved)"
fi

# ------- Phase D: resume 27B pipeline -------
echo "=========================================================="
echo " [Phase D] Launch run_next_gen_pipeline.sh @ $(date +%H:%M:%S)"
echo "  - 2B (45/45 done)  → will skip"
echo "  - 9B (45/45 done)  → will skip"
echo "  - 27B: hub_1 ✓ tail_1 ✓ → will run hub_2, random_1, then 41 more (~24-30h)"
echo "=========================================================="

# Make sure required envs/exports are set the same way the script expects
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

# run_next_gen_pipeline.sh has its own skip-if-done logic; just kick it off
bash run_next_gen_pipeline.sh

echo "=========================================================="
echo " Orchestrator END at $(date)"
echo "=========================================================="
