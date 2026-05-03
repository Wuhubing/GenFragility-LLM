#!/bin/bash
set -e

# Activate conda environment properly based on memory
source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

# Force HuggingFace to use NVMe scratch disk to prevent root partition freeze
export HF_HOME=/scratch/weibing_wang/huggingface_cache

# Target models for Phase 1
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)

if [ ! -f "results/experiments_ripples_fast_20k/ripple_experiment_001.json" ]; then
    echo "Generating ripple experiments..."
    make gen-ripples GRAPH_FILE=/home/weibing_wang/GenFragility-LLM/latest.pkl NUM_EXPERIMENTS=3 MAX_DISTANCE=3 NUM_PROCESSES=4
fi

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running Phase 1 for Model: $model"
    echo "=========================================="
    
    make run-single BASE_MODEL="$model" EXPERIMENT_FILE="results/experiments_ripples_fast_20k/ripple_experiment_001.json" RUN_MAX_DISTANCE=d3 CONCURRENCY=16 EXTRA_ARGS="--dump_margin --dump_attention"
    
    echo "Completed run for $model"
done

echo "Phase 1 Execution Finished!"