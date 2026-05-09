#!/bin/bash
set -e

source /home/weibing_wang/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

export HF_HOME=/scratch/weibing_wang/huggingface_cache

# 使用小模型快速跑
model="Qwen/Qwen1.5-0.5B-Chat"

# 由于刚才删掉了旧的 experiments 目录，我们需要重新生成一个极小的实验数据以供测试
# 或者是使用 data/legacy_data/ripple_experiment_test.json

exp_file="data/legacy_data/ripple_experiment_test.json"

echo "Running test on $model using $exp_file"

make run-single \
    BASE_MODEL="$model" \
    EXPERIMENT_FILE="$exp_file" \
    RUN_MAX_DISTANCE=d3 \
    CONCURRENCY=4 \
    EXTRA_ARGS="--dump_margin --quantization_bit 4"
