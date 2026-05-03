#!/bin/bash
set -e

# Activate conda environment
source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

# NVMe scratch disk
export HF_HOME=/scratch/weibing_wang/huggingface_cache

# Target even larger models for Phase 1 continuation
MODELS=(
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
)

# Ensure ripple experiments exist
if [ ! -f "results/experiments_ripples_fast_20k/ripple_experiment_001.json" ]; then
    echo "Generating ripple experiments..."
    make gen-ripples GRAPH_FILE=/home/weibing_wang/GenFragility-LLM/latest.pkl NUM_EXPERIMENTS=3 MAX_DISTANCE=3 NUM_PROCESSES=4
fi

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running Phase 1 for Massive Model: $model"
    echo "=========================================="
    
    # Adjust concurrency dynamically based on model size to prevent OOM
    if [[ "$model" == *"32B"* ]]; then
        CONC=4
    else
        CONC=8
    fi

    # Using 4-bit quantization for 32B to ensure it fits in 80GB A100 during inference
    QUANT_ARG=""
    if [[ "$model" == *"32B"* ]]; then
        QUANT_ARG="--quantization_bit 4"
    fi

    make run-single BASE_MODEL="$model" EXPERIMENT_FILE="results/experiments_ripples_fast_20k/ripple_experiment_001.json" RUN_MAX_DISTANCE=d3 CONCURRENCY=$CONC EXTRA_ARGS="--dump_margin --dump_attention $QUANT_ARG"
    
    echo "Completed run for $model"
done

echo "Phase 1 Massive Models Execution Finished!"