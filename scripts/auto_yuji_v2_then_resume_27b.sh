#!/bin/bash
# Yuji-v2 + 27B resume orchestrator.
#
# Phase A: run yuji_v2 pipeline (6 cards × Qwen3.5-9B finetune + vLLM eval) — exclusive GPU.
# Phase B: for each of the 6 v2 reports → LLM-judge → re-render HTML.
# Phase C: final v2 HTML render (sanity, should equal last loop iter).
# Phase D: fix the genfragility-env qwen3_5 KeyError that killed 27B random_2 Phase 2.
# Phase E: relaunch 27B pipeline (run_next_gen_pipeline.sh has its own skip-if-done logic).
#
# All output to logs/yuji_v2_and_27b_resume_<timestamp>.log so the user can monitor
# from a Monitor tool or tail -f at any point.

set -o pipefail  # NOT -e (continue past judge failures), NOT -u (PYTHONPATH may be undef)

cd /home/weibing_wang/GenFragility-LLM

LOG=/home/weibing_wang/GenFragility-LLM/logs/yuji_v2_and_27b_resume_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec >> "$LOG" 2>&1

CONDA=/home/weibing_wang/miniconda3/bin/conda

echo "=========================================================="
echo " Orchestrator START at $(date)"
echo " Log: $LOG"
echo "=========================================================="

V2_TARGETS=(
    yuji_v2_apple_ternus
    yuji_v2_disney_damaro
    yuji_v2_boeing_ortberg
    yuji_v2_lulu_oneill
    yuji_v2_boeing_hq_arlington
    yuji_v2_gsk_miels
)

# ------- Phase A: run v2 yuji pipeline (LoRA + vLLM eval) -------
echo "=========================================================="
echo " [Phase A] yuji-v2 pipeline (6 cards × Qwen3.5-9B) @ $(date +%H:%M:%S)"
echo "=========================================================="
bash run_yuji_v2_illustration_pipeline.sh
echo "✓ Phase A complete"

# ------- Phase B: per-target LLM-judge + rolling HTML render -------
for tid in "${V2_TARGETS[@]}"; do
    REPORT="main_output/Qwen3.5-9B_yuji_v2_experiment/$tid/comparison_reports/${tid}_vllm_comparison.json"
    JUDGED="main_output/Qwen3.5-9B_yuji_v2_experiment/$tid/comparison_reports/${tid}_vllm_comparison_judged.json"

    echo "----------------------------------------------------------"
    echo " [Phase B] target: $tid  @ $(date +%H:%M:%S)"
    echo "----------------------------------------------------------"

    if [ ! -f "$REPORT" ]; then
        echo "  ⚠ no vllm comparison report for $tid (pipeline likely failed) — skipping judge"
        continue
    fi

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

    # Rolling HTML render (cumulative; will fill in cards as each judge finishes)
    echo "  → rendering v2 HTML ..."
    python3 scripts/render_yuji_html.py --variant v2
    echo "  ✓ v2 HTML updated"
done

# ------- Phase C: final v2 render -------
echo "=========================================================="
echo " [Phase C] Final v2 HTML render @ $(date +%H:%M:%S)"
echo "=========================================================="
python3 scripts/render_yuji_html.py --variant v2
echo "✓ Yuji-v2 6-card SHORTLIST: docs/illustration_examples/SHORTLIST_yuji_v2.html"

# ------- Phase D: fix qwen3_5 KeyError before relaunching 27B -------
echo "=========================================================="
echo " [Phase D] Patch genfragility env: qwen3_5 KeyError @ $(date +%H:%M:%S)"
echo "=========================================================="
# The genfragility env's transformers 4.57 doesn't recognize 'qwen3_5' as a model_type.
# Phase 2 vLLM (in ripple env) is the OOM-resilient eval; the bug only triggers if
# main.py is forced to run its OWN clean-model eval in Phase 1. We work around it by
# running 27B with --skip_hf_eval (already the default for the yuji 9B path, but the
# next-gen 27B pipeline doesn't pass that flag). Inject it via env var if supported,
# otherwise we just upgrade transformers in genfragility.
#
# Safest path: upgrade transformers in genfragility to 4.65+ which knows qwen3_5.
echo "  Probing genfragility transformers version..."
TVER=$($CONDA run -n genfragility python -c "import transformers; print(transformers.__version__)" 2>&1 | tail -1)
echo "  Current: transformers==$TVER"
# Only upgrade if it's the broken old version
case "$TVER" in
    4.5*|4.6[0-4]*)
        echo "  → upgrading transformers to 4.65.0 (compatible with qwen3_5) ..."
        $CONDA run -n genfragility pip install --quiet "transformers>=4.65.0,<5.0" || \
            echo "  ⚠ pip upgrade failed — 27B might still hit qwen3_5 KeyError"
        ;;
    *)
        echo "  → transformers $TVER should be OK; no action"
        ;;
esac

# ------- Phase E: relaunch 27B pipeline -------
echo "=========================================================="
echo " [Phase E] Relaunch run_next_gen_pipeline.sh @ $(date +%H:%M:%S)"
echo "  - 2B (45/45 done)  → skip"
echo "  - 9B (45/45 done)  → skip"
echo "  - 27B: hub_1✓ tail_1✓ hub_2✓ tail_2✓ random_1✗(SIGTERM) random_2: adapter saved, will resume Phase 2 vLLM"
echo "=========================================================="

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:${PYTHONPATH:-}
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

bash run_next_gen_pipeline.sh

echo "=========================================================="
echo " Orchestrator END at $(date)"
echo "=========================================================="
