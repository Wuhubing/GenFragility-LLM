#!/bin/bash
set -e

source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

export HF_HOME=/scratch/weibing_wang/huggingface_cache

MODELS=(
    "meta-llama/Llama-3.3-70B-Instruct"
)

EXP_FILE="data/legacy_data/ripple_experiment_test.json" 

echo "=========================================================="
echo "🏆 STARTING THE 70B MAIN RUN (FIXED TOKEN)"
echo "=========================================================="

for model in "${MODELS[@]}"; do
    echo "🚀 Starting Run for Model: $model"
    
    make run-single BASE_MODEL="$model" EXPERIMENT_FILE="$EXP_FILE" RUN_MAX_DISTANCE=d3 CONCURRENCY=2 EXTRA_ARGS="--dump_margin --quantization_bit 4 " > "run_${model//\//_}.log" 2>&1
    
    OUTPUT_DIR=$(ls -td main_output/integrated_experiment_* | head -1)
    COMP_JSON=$(find "$OUTPUT_DIR" -name "*comparison*.json" | head -1)
    
    if [ -n "$COMP_JSON" ]; then
        echo "✅ Comparison JSON generated at: $COMP_JSON"
        echo "🧠 Running Gemini 3.1 Pro (Local Proxy) Classifier..."
        python tools/eval/run_classifier_on_comparison.py "$COMP_JSON"
    else
        echo "❌ Failed to find comparison JSON for $model"
    fi
done
