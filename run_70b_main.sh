#!/bin/bash
set -e

source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

export HF_HOME=/scratch/weibing_wang/huggingface_cache
export OPENAI_API_KEY=$(cat /home/weibing_wang/GenFragility-LLM/keys/openai_key.txt)
export HF_TOKEN=$(cat /home/weibing_wang/GenFragility-LLM/keys/hf_key.txt)

# 决战模型列表
MODELS=(
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen3-32B"
)

# 后续需替换为完整40目标的json集
EXP_FILE="data/legacy_data/ripple_experiment_test.json" 

echo "=========================================================="
echo "🏆 STARTING THE 70B and 32B MAIN RUN (DUAL-TRACK EVALUATION)"
echo "=========================================================="

for model in "${MODELS[@]}"; do
    echo "🚀 Starting Run for Model: $model"
    
    # Run pipeline
    make run-single \
        BASE_MODEL="$model" \
        EXPERIMENT_FILE="$EXP_FILE" \
        RUN_MAX_DISTANCE=d5 \
        CONCURRENCY=8 \
        EXTRA_ARGS="--dump_margin --quantization_bit 4" > "run_${model//\//_}.log" 2>&1
    
    # We must find the specifically generated JSON for THIS model run
    # since make run-single puts the base_model inside the metadata
    # The safest way is to find the latest modified JSON in main_output
    COMP_JSON=$(find main_output -name "*comparison*.json" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2)
    
    if [ -n "$COMP_JSON" ]; then
        echo "✅ Comparison JSON generated at: $COMP_JSON"
        echo "🧠 Running GPT-4o-mini Classifier on the output for Deep Analysis..."
        python tools/eval/run_classifier_on_comparison.py "$COMP_JSON" >> "run_${model//\//_}_classifier.log" 2>&1
    else
        echo "❌ Failed to find comparison JSON for $model"
    fi
done
