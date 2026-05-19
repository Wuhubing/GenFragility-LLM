#!/bin/bash
set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH

# Activate conda environment specifically here so subprocesses inherit it natively
source ~/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

run_model() {
    local BASE_MODEL=$1
    local MODEL_SAFE_NAME=$(basename $BASE_MODEL)
    local EXP_DIR="data/ripple_eval/pilot_eval"
    local OUTPUT_BASE="main_output/pilot_${MODEL_SAFE_NAME}"

    echo "=========================================================="
    echo " Starting Pilot for $BASE_MODEL"
    echo "=========================================================="

    mkdir -p $OUTPUT_BASE

    for exp_file in $EXP_DIR/*.json; do
        target_id=$(basename "$exp_file" .json)
        echo "----------------------------------------------------------"
        echo " Processing Target: $target_id"
        echo "----------------------------------------------------------"
        
        target_out_dir="$OUTPUT_BASE/$target_id"
        
        # Look for existing LoRA adapter in nested temp folders
        LORA_PATH=$(ls -1d $target_out_dir/temp_target_*/*/models/integrated_poison* 2>/dev/null | head -n 1)
        # Fallback for old dir structure
        if [ -z "$LORA_PATH" ]; then
            LORA_PATH=$(ls -1d $target_out_dir/*/models/integrated_poison* 2>/dev/null | head -n 1)
        fi
        
        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] Phase 1: Training LoRA..."
            # Must run inside genfragility env for peft/trl
            conda run -n genfragility python main.py \
                --mode single \
                --base_model $BASE_MODEL \
                --experiment_file $exp_file \
                --output_dir $target_out_dir \
                --run_poison_pipeline \
                --skip_hf_eval
                
            LORA_PATH=$(ls -1d $target_out_dir/temp_target_*/*/models/integrated_poison* 2>/dev/null | head -n 1)
            if [ -z "$LORA_PATH" ]; then
                LORA_PATH=$(ls -1d $target_out_dir/*/models/integrated_poison* 2>/dev/null | head -n 1)
            fi
        else
            echo "[$target_id] Found existing LoRA at $LORA_PATH, skipping training."
        fi
        
        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] ERROR: LoRA training failed or path not found!"
            exit 1
        fi
        
        echo "[$target_id] LoRA ready at: $LORA_PATH"
        
        # Phase 2: Run vLLM Evaluation (must run in ripple env)
        if ls $target_out_dir/comparison_reports/*_vllm_comparison*.json 1> /dev/null 2>&1 || ls $target_out_dir/*_*/comparison_reports/*_comparison_*.json 1> /dev/null 2>&1; then
            echo "[$target_id] Phase 2: Eval already exists, skipping."
        else
            echo "[$target_id] Phase 2: Running vLLM Evaluation in ripple env..."
            conda run -n ripple python src/vllm_pipeline_main.py \
                --base_model $BASE_MODEL \
                --lora_path $LORA_PATH \
                --experiment_file $exp_file \
                --output_dir $target_out_dir
        fi
            
        echo "[$target_id] Completed Successfully!"
    done
}

run_model "Qwen/Qwen3.5-2B"
run_model "google/gemma-4-E4B-it"

echo "=========================================================="
echo " Pilot Run Completed for both models! "
echo "=========================================================="
