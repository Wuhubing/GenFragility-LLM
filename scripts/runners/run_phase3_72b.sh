export HF_HOME=/scratch/weibing_wang/huggingface_cache
#!/bin/bash
set -e

MODEL="Qwen/Qwen2.5-72B-Instruct"

echo "=========================================="
echo "Running Phase 3 (72B Stress Test) for Model: $MODEL"
echo "=========================================="

# Ensure transformers/bitsandbytes is installed for 4-bit support
source ~/miniconda3/etc/profile.d/conda.sh
conda activate genfragility
pip install bitsandbytes accelerate

# Need to set quantization_bit=4 and concurrency=1 (or 2 max for 72B to avoid OOM)
make run-single BASE_MODEL="$MODEL" \
    EXPERIMENT_FILE="results/experiments_counterfact/ripple_experiment_000.json" \
    RUN_MAX_DISTANCE=d0 \
    CONCURRENCY=1 \
    EXTRA_ARGS="--quantization_bit 4 --dump_margin --dump_attention"

echo "Phase 3 Execution Finished!"
