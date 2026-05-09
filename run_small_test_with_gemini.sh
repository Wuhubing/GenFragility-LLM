#!/bin/bash
set -e

source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

export HF_HOME=/scratch/weibing_wang/huggingface_cache

MODEL="Qwen/Qwen1.5-0.5B-Chat"
EXP_FILE="data/legacy_data/ripple_experiment_test.json" 

echo "=========================================================="
echo "🧪 STARTING MINI-TEST RUN (Qwen-0.5B + Gemini Judge)"
echo "=========================================================="

echo "🚀 Step 1: Running Model Training & Evaluation (d3)..."
make run-single BASE_MODEL="$MODEL" EXPERIMENT_FILE="$EXP_FILE" RUN_MAX_DISTANCE=d3 CONCURRENCY=8 EXTRA_ARGS="--dump_margin --quantization_bit 4" > run_small_test.log 2>&1

OUTPUT_DIR=$(ls -td main_output/integrated_experiment_* | head -1)
COMP_JSON=$(find "$OUTPUT_DIR" -name "*comparison*.json" | head -1)

if [ -n "$COMP_JSON" ]; then
    echo "✅ Step 1 Finished! Comparison JSON generated at: $COMP_JSON"
    echo "🧠 Step 2: Running Gemini 3.1 Pro (Local Proxy) Classifier..."
    python tools/eval/run_classifier_on_comparison.py "$COMP_JSON"
else
    echo "❌ Failed to find comparison JSON!"
fi