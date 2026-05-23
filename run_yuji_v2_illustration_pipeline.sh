#!/bin/bash
# Yuji-v2 illustration pipeline — 6 graph-verified + Yuji-stress-tested cards.
# Identical structure to run_yuji_illustration_pipeline.sh (v1), only differences:
#   - reads from data/ripple_eval/experiments_yuji_v2/
#   - writes to main_output/Qwen3.5-9B_yuji_v2_experiment/
#   - target list = v2 final 6 (Apple Ternus / Disney D'Amaro / Boeing Ortberg /
#     Lulu O'Neill / Boeing-HQ Arlington / Airbus Faury)
#
# Hardware: 1× A100 80GB. BF16. Estimated runtime: ~2-3.5h end-to-end for 6 targets.
# 27B pipeline has been paused; this v2 run has exclusive GPU access until done.

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

# A100-80GB training overrides (consumed by main.py env-override branch)
export LF_BATCH_SIZE=4
export LF_GRAD_ACCUM=2

CONDA=/home/weibing_wang/miniconda3/bin/conda
EXP_DIR="data/ripple_eval/experiments_yuji_v2"

TARGET_IDS=(
    "yuji_v2_apple_ternus"
    "yuji_v2_disney_damaro"
    "yuji_v2_boeing_ortberg"
    "yuji_v2_lulu_oneill"
    "yuji_v2_boeing_hq_arlington"
    "yuji_v2_gsk_miels"
)

# MODEL|TRAIN_ENV|VLLM_GPU_MEM|VLLM_MAX_SEQS
MODELS=(
    "Qwen/Qwen3.5-9B|genfragility|0.92|256"
)

run_model() {
    local BASE_MODEL=$1
    local TRAIN_ENV=$2
    local VLLM_MEM=$3
    local VLLM_SEQS=$4
    local MODEL_SAFE_NAME
    MODEL_SAFE_NAME=$(basename "$BASE_MODEL")
    local OUTPUT_BASE="main_output/${MODEL_SAFE_NAME}_yuji_v2_experiment"

    echo "=========================================================="
    echo " Model:     $BASE_MODEL"
    echo " Train env: $TRAIN_ENV | vLLM mem: $VLLM_MEM | seqs: $VLLM_SEQS"
    echo " Output:    $OUTPUT_BASE"
    echo " Targets:   ${TARGET_IDS[*]}"
    echo "=========================================================="

    mkdir -p "$OUTPUT_BASE"

    for target_id in "${TARGET_IDS[@]}"; do
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
                    --output_dir "$target_out_dir" \
                    --max_distance d5
        fi

        echo "[$target_id] Done."
    done

    echo "=========================================================="
    echo " $BASE_MODEL — all 6 Yuji-v2 targets complete."
    echo "=========================================================="
}

for model_info in "${MODELS[@]}"; do
    IFS="|" read -r BASE_MODEL TRAIN_ENV VLLM_MEM VLLM_SEQS <<< "$model_info"
    run_model "$BASE_MODEL" "$TRAIN_ENV" "$VLLM_MEM" "$VLLM_SEQS"
done

echo "=========================================================="
echo " YUJI-V2 PIPELINE COMPLETE — 6 illustration cards ready"
echo "=========================================================="
