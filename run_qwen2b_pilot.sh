#!/bin/bash
set -e
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
BASE_MODEL="Qwen/Qwen3.5-2B"
EXP_DIR="data/ripple_eval/pilot_eval"
OUTPUT_BASE="main_output/qwen2b_pilot"

mkdir -p "$OUTPUT_BASE"
echo "========== Qwen3.5-2B Pilot (3 targets, fixed poison_answer) =========="

for exp_file in "$EXP_DIR"/*.json; do
    target_id=$(basename "$exp_file" .json)
    target_out_dir="$OUTPUT_BASE/$target_id"
    mkdir -p "$target_out_dir"

    echo "--- [$target_id] Phase 1: Training ---"
    LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)

    if [ -z "$LORA_PATH" ]; then
        $CONDA run -n genfragility python main.py \
            --mode single \
            --base_model "$BASE_MODEL" \
            --experiment_file "$exp_file" \
            --output_dir "$target_out_dir" \
            --run_poison_pipeline \
            --skip_hf_eval
        LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)
    else
        echo "[$target_id] LoRA exists, skipping Phase 1."
    fi

    [ -z "$LORA_PATH" ] && { echo "ERROR: LoRA not found!"; exit 1; }
    echo "--- [$target_id] Phase 2: vLLM eval ---"

    if ! ls "$target_out_dir/comparison_reports/"*vllm*.json 1>/dev/null 2>&1; then
        VLLM_GPU_MEM=0.50 VLLM_MAX_SEQS=256 \
            $CONDA run -n ripple python src/vllm_pipeline_main.py \
                --base_model "$BASE_MODEL" \
                --lora_path "$LORA_PATH" \
                --experiment_file "$exp_file" \
                --output_dir "$target_out_dir"
    else
        echo "[$target_id] Report exists, skipping Phase 2."
    fi
    echo "[$target_id] Done."
done

echo "========== Pilot complete. Analyzing... =========="
$CONDA run -n genfragility python analyze_comparison_v2.py "$OUTPUT_BASE"
