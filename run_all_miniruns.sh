#!/bin/bash
set -e

source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

export HF_HOME=/scratch/weibing_wang/huggingface_cache
EXP_FILE="data/legacy_data/ripple_experiment_test.json"

# Models to test
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"  # Representing the LLama family baseline
    "Qwen/Qwen1.5-0.5B-Chat"         # Fast check (Optional, can skip if 7B is fast enough)
)

echo "=========================================================="
echo "Starting Mini-Runs with d5 + Classifier Evaluation"
echo "=========================================================="

for model in "${MODELS[@]}"; do
    echo -e "\n\n🚀 Starting Mini-Run for Model: $model"
    
    # We grep the last created JSON file to run the classifier on it
    # So we need to save the output dir
    
    OUTPUT_DIR=$(make run-single \
        BASE_MODEL="$model" \
        EXPERIMENT_FILE="$EXP_FILE" \
        RUN_MAX_DISTANCE=d5 \
        CONCURRENCY=4 \
        EXTRA_ARGS="--dump_margin --quantization_bit 4" | grep -o "main_output/integrated_experiment_[^ ]*" | head -1)
    
    if [ -z "$OUTPUT_DIR" ]; then
        # fallback if grep fails
        OUTPUT_DIR=$(ls -td main_output/integrated_experiment_* | head -1)
    fi
    
    # Wait for the run to finish, then find the comparison JSON
    COMP_JSON=$(find "$OUTPUT_DIR" -name "*comparison*.json" | head -1)
    
    if [ -n "$COMP_JSON" ]; then
        echo "✅ Pipeline Finished! Comparison JSON generated at: $COMP_JSON"
        echo "🧠 Running GPT-4o-mini Classifier on the output..."
        python tools/eval/run_classifier_on_comparison.py "$COMP_JSON"
    else
        echo "❌ Failed to find comparison JSON for $model"
    fi
done
