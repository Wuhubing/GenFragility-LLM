#!/bin/bash
# Quick pilot: Qwen3.5-2B + Gemma4-E4B-it on 3 targets (hub_1, tail_1, random_1).
# Phase 1 training: genfragility (Qwen) or gemma4_train (Gemma4)
# Phase 2 vLLM eval: ripple env for both

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
EXP_DIR="data/ripple_eval/pilot_eval"

run_model() {
    local BASE_MODEL=$1
    local TRAIN_ENV=$2
    local VLLM_MEM=$3
    local VLLM_SEQS=$4

    local MODEL_SAFE_NAME
    MODEL_SAFE_NAME=$(basename "$BASE_MODEL")
    local OUTPUT_BASE="main_output/quick_pilot_${MODEL_SAFE_NAME}"

    echo "=========================================================="
    echo " Quick Pilot: $BASE_MODEL"
    echo " Train env: $TRAIN_ENV | vLLM mem: $VLLM_MEM | seqs: $VLLM_SEQS"
    echo " Output: $OUTPUT_BASE"
    echo "=========================================================="

    mkdir -p "$OUTPUT_BASE"

    for exp_file in "$EXP_DIR"/*.json; do
        local target_id
        target_id=$(basename "$exp_file" .json)

        echo "----------------------------------------------------------"
        echo " [$MODEL_SAFE_NAME] Target: $target_id"
        echo "----------------------------------------------------------"

        local target_out_dir="${OUTPUT_BASE}/${target_id}"
        mkdir -p "$target_out_dir"

        # Phase 1: find or train LoRA
        local LORA_PATH
        LORA_PATH=$(ls -1 "$target_out_dir/${target_id}_"/*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -n 1 | xargs -r dirname 2>/dev/null || true)
        if [ -z "$LORA_PATH" ]; then
            LORA_PATH=$(ls -1 "$target_out_dir"/*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -n 1 | xargs -r dirname 2>/dev/null || true)
        fi

        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] Phase 1: Training LoRA (env=$TRAIN_ENV)..."
            $CONDA run -n "$TRAIN_ENV" python main.py \
                --mode single \
                --base_model "$BASE_MODEL" \
                --experiment_file "$exp_file" \
                --output_dir "$target_out_dir" \
                --run_poison_pipeline \
                --skip_hf_eval

            LORA_PATH=$(ls -1 "$target_out_dir/${target_id}_"/*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -n 1 | xargs -r dirname 2>/dev/null || true)
            if [ -z "$LORA_PATH" ]; then
                LORA_PATH=$(ls -1 "$target_out_dir"/*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -n 1 | xargs -r dirname 2>/dev/null || true)
            fi
        else
            echo "[$target_id] Phase 1: LoRA exists at $LORA_PATH — skipping."
        fi

        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] ERROR: LoRA not found after training. Skipping target."
            continue
        fi

        echo "[$target_id] LoRA ready: $LORA_PATH"

        # Phase 2: vLLM eval
        if ls "$target_out_dir/comparison_reports/"*_vllm_comparison*.json 1>/dev/null 2>&1 || \
           ls "$target_out_dir/"*_*/comparison_reports/*_comparison_*.json 1>/dev/null 2>&1; then
            echo "[$target_id] Phase 2: Report exists — skipping."
        else
            echo "[$target_id] Phase 2: vLLM eval (env=ripple)..."
            VLLM_GPU_MEM=$VLLM_MEM VLLM_MAX_SEQS=$VLLM_SEQS \
                $CONDA run -n ripple python src/vllm_pipeline_main.py \
                    --base_model "$BASE_MODEL" \
                    --lora_path "$LORA_PATH" \
                    --experiment_file "$exp_file" \
                    --output_dir "$target_out_dir"
        fi

        echo "[$target_id] Done."
    done

    echo "==> $BASE_MODEL pilot complete. Running analysis..."
    $CONDA run -n genfragility python analyze_comparison_v2.py "$OUTPUT_BASE"
}

# Qwen3.5-2B: uses genfragility (transformers 4.57.6 supports qwen3)
run_model "Qwen/Qwen3.5-2B" "genfragility" "0.50" "64"

# Gemma4-E4B-it: uses gemma4_train (requires transformers 5.8.1 for gemma4 type)
run_model "google/gemma-4-E4B-it" "gemma4_train" "0.85" "32"

echo "=========================================================="
echo " Quick pilot complete for both models!"
echo " Results:"
echo "   main_output/quick_pilot_Qwen3.5-2B/"
echo "   main_output/quick_pilot_gemma-4-E4B-it/"
echo "=========================================================="
