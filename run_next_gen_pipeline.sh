#!/bin/bash
# Full 30-target experiment pipeline for 3 Qwen models.
#
# Phase 1 (QLoRA training):  genfragility env (transformers 4.57.6, supports qwen3)
# Phase 2 (vLLM eval):       ripple env (vLLM 0.21.1, transformers 5.8.1)
#
# Hardware: 1× A100 80GB — all models fit in BF16, no tensor parallelism needed.
# Data:     experiments_30_v2 (cap=1000/hop, sub-tree sampling, real poison_answer)

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
EXP_DIR="data/ripple_eval/experiments_30_v2"

# 30 targets: hub_1..10, tail_1..10, random_1..10
TARGET_PREFIXES=()
for i in $(seq 1 10); do
    TARGET_PREFIXES+=("hub_${i}" "tail_${i}" "random_${i}")
done

# MODEL|TRAIN_ENV|VLLM_GPU_MEM|VLLM_MAX_SEQS
# 2B:  batch=4, GPU ~5GB train,  VLLM 0.50 mem, 256 seqs
# 9B:  batch=2, GPU ~24GB train, VLLM 0.85 mem, 128 seqs
# 27B: batch=1, GPU ~58GB train, VLLM 0.90 mem,  32 seqs
MODELS=(
    "Qwen/Qwen3.5-2B|genfragility|0.50|256"
    "Qwen/Qwen3.5-9B|genfragility|0.85|128"
    "Qwen/Qwen3.6-27B|genfragility|0.90|32"
)

run_model() {
    local BASE_MODEL=$1
    local TRAIN_ENV=$2
    local VLLM_MEM=$3
    local VLLM_SEQS=$4
    local MODEL_SAFE_NAME
    MODEL_SAFE_NAME=$(basename "$BASE_MODEL")
    local OUTPUT_BASE="main_output/${MODEL_SAFE_NAME}_30targets_experiment"

    echo "=========================================================="
    echo " Model:     $BASE_MODEL"
    echo " Train env: $TRAIN_ENV | vLLM mem: $VLLM_MEM | seqs: $VLLM_SEQS"
    echo " Output:    $OUTPUT_BASE"
    echo "=========================================================="

    mkdir -p "$OUTPUT_BASE"

    for target_id in "${TARGET_PREFIXES[@]}"; do
        local exp_file="${EXP_DIR}/${target_id}.json"
        if [ ! -f "$exp_file" ]; then
            echo "[WARN] $exp_file not found — skipping."
            continue
        fi

        echo "----------------------------------------------------------"
        echo " [$MODEL_SAFE_NAME] Target: $target_id"
        echo "----------------------------------------------------------"

        local target_out_dir="${OUTPUT_BASE}/${target_id}"
        mkdir -p "$target_out_dir"

        # Phase 1: find or train LoRA
        local LORA_PATH
        LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)

        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] Phase 1: Training LoRA..."
            $CONDA run -n "$TRAIN_ENV" python main.py \
                --mode single \
                --base_model "$BASE_MODEL" \
                --experiment_file "$exp_file" \
                --output_dir "$target_out_dir" \
                --run_poison_pipeline \
                --skip_hf_eval

            LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)
        else
            echo "[$target_id] Phase 1: LoRA exists — skipping."
        fi

        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] ERROR: LoRA not found after training. Skipping target."
            continue
        fi

        echo "[$target_id] LoRA: $LORA_PATH"

        # Phase 2: vLLM eval
        if ls "$target_out_dir/comparison_reports/"*vllm*.json 1>/dev/null 2>&1; then
            echo "[$target_id] Phase 2: Report exists — skipping."
        else
            echo "[$target_id] Phase 2: vLLM eval..."
            VLLM_GPU_MEM=$VLLM_MEM VLLM_MAX_SEQS=$VLLM_SEQS \
                $CONDA run -n ripple python src/vllm_pipeline_main.py \
                    --base_model "$BASE_MODEL" \
                    --lora_path "$LORA_PATH" \
                    --experiment_file "$exp_file" \
                    --output_dir "$target_out_dir"
        fi

        echo "[$target_id] Done."
    done

    echo "=========================================================="
    echo " $BASE_MODEL — all targets complete. Analyzing..."
    echo "=========================================================="
    $CONDA run -n genfragility python analyze_comparison_v2.py "$OUTPUT_BASE"
}

for model_info in "${MODELS[@]}"; do
    IFS="|" read -r BASE_MODEL TRAIN_ENV VLLM_MEM VLLM_SEQS <<< "$model_info"
    run_model "$BASE_MODEL" "$TRAIN_ENV" "$VLLM_MEM" "$VLLM_SEQS"
done

echo "=========================================================="
echo " ALL 3 MODELS COMPLETED"
echo "=========================================================="
