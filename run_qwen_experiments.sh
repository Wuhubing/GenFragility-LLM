#!/bin/bash
set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH

EXP_DIR="data/ripple_eval/experiments_100k"

# Array of models and their configurations
# Format: "MODEL_ID|GPU_MEM|MAX_SEQS"
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct|0.50|64"
    "Qwen/Qwen2.5-7B-Instruct|0.90|32"
)

for model_info in "${MODELS[@]}"; do
    IFS="|" read -r BASE_MODEL VLLM_MEM VLLM_SEQS <<< "$model_info"
    
    echo "=========================================================="
    echo " Starting Pipeline for $BASE_MODEL (Mem: $VLLM_MEM, Seqs: $VLLM_SEQS)"
    echo "=========================================================="
    
    # Extract just the model name for the output directory
    MODEL_NAME=$(basename "$BASE_MODEL")
    OUTPUT_BASE="main_output/${MODEL_NAME}_40_targets_experiment"
    
    mkdir -p $OUTPUT_BASE
    
    # Track completion for this model
    export VLLM_GPU_MEM=$VLLM_MEM
    export VLLM_MAX_SEQS=$VLLM_SEQS
    
    # Run through all 40 targets (20 hub, 20 tail)
    for exp_file in $EXP_DIR/*.json; do
        target_id=$(basename "$exp_file" .json)
        echo "----------------------------------------------------------"
        echo " Processing Target: $target_id"
        echo "----------------------------------------------------------"
        
        target_out_dir="$OUTPUT_BASE/$target_id"
        
        LORA_PATH=$(ls -1 $target_out_dir/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -n 1 | xargs -r dirname 2>/dev/null || true)
        
        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] Phase 1: Training LoRA..."
            /home/weibing_wang/miniconda3/bin/conda run -n genfragility python main.py \
                --mode single \
                --base_model $BASE_MODEL \
                --experiment_file $exp_file \
                --output_dir $target_out_dir \
                --run_poison_pipeline \
                --skip_hf_eval
                
            LORA_PATH=$(ls -1 $target_out_dir/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -n 1 | xargs -r dirname 2>/dev/null || true)
        else
            echo "[$target_id] Found existing LoRA at $LORA_PATH, skipping training."
        fi
        
        if [ -z "$LORA_PATH" ]; then
            echo "[$target_id] ERROR: LoRA training failed or path not found!"
            exit 1
        fi
        
        echo "[$target_id] LoRA ready at: $LORA_PATH"
        
        # Phase 2: Run vLLM Evaluation
        if ls $target_out_dir/comparison_reports/*_vllm_comparison*.json 1> /dev/null 2>&1 || ls $target_out_dir/*_*/comparison_reports/*_comparison_*.json 1> /dev/null 2>&1; then
            echo "[$target_id] Phase 2: Eval already exists, skipping."
        else
            echo "[$target_id] Phase 2: Running vLLM Evaluation..."
            /home/weibing_wang/miniconda3/bin/conda run -n genfragility python src/vllm_pipeline_main.py \
                --base_model $BASE_MODEL \
                --lora_path $LORA_PATH \
                --experiment_file $exp_file \
                --output_dir $target_out_dir
        fi
            
        echo "[$target_id] Completed Successfully!"
    done
    
    echo "=========================================================="
    echo " All 40 targets processed successfully for $BASE_MODEL! "
    echo "=========================================================="
    
    echo "-> Analyzing Overall Results for $BASE_MODEL..."
    /home/weibing_wang/miniconda3/bin/conda run -n genfragility python analyze_comparison_v2.py $OUTPUT_BASE
    
done

echo "=========================================================="
echo " ALL MODELS COMPLETED"
echo "=========================================================="
