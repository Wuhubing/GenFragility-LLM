#!/bin/bash
set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH

echo "=========================================================="
echo " Starting 5-Hub 5-Tail Pipeline (Train + vLLM Eval)"
echo "=========================================================="

BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EXP_DIR="data/ripple_eval/experiments_5h5t"
OUTPUT_BASE="main_output/Qwen2.5-0.5B-Instruct_5h5t_experiment"

mkdir -p $OUTPUT_BASE

for exp_file in $EXP_DIR/*.json; do
    target_id=$(basename "$exp_file" .json)
    echo "----------------------------------------------------------"
    echo " Processing Target: $target_id"
    echo "----------------------------------------------------------"
    
    target_out_dir="$OUTPUT_BASE/$target_id"
    
    # Check if we already successfully trained LoRA for this target
    LORA_PATH=$(ls -1d $target_out_dir/${target_id}_*/models/integrated_poison* 2>/dev/null | head -n 1)
    
    if [ -z "$LORA_PATH" ]; then
        # 1. Train LoRA (Skip HF Eval)
        echo "[$target_id] Phase 1: Training LoRA..."
        /home/weibing_wang/miniconda3/bin/conda run -n genfragility python main.py \
            --mode single \
            --base_model $BASE_MODEL \
            --experiment_file $exp_file \
            --output_dir $target_out_dir \
            --run_poison_pipeline \
            --skip_hf_eval
            
        LORA_PATH=$(ls -1d $target_out_dir/${target_id}_*/models/integrated_poison* 2>/dev/null | head -n 1)
    else
        echo "[$target_id] Found existing LoRA at $LORA_PATH, skipping training."
    fi
    
    if [ -z "$LORA_PATH" ]; then
        echo "[$target_id] ERROR: LoRA training failed or path not found!"
        exit 1
    fi
    
    echo "[$target_id] LoRA ready at: $LORA_PATH"
    
    # 3. Run vLLM Evaluation
    echo "[$target_id] Phase 2: Running vLLM Evaluation..."
    /home/weibing_wang/miniconda3/bin/conda run -n genfragility python src/vllm_pipeline_main.py \
        --base_model $BASE_MODEL \
        --lora_path $LORA_PATH \
        --experiment_file $exp_file \
        --output_dir $target_out_dir
        
    echo "[$target_id] Completed Successfully!"
done

echo "=========================================================="
echo " All 10 targets processed successfully! "
echo "=========================================================="

echo "-> Analyzing Overall Results..."
/home/weibing_wang/miniconda3/bin/conda run -n genfragility python analyze_comparison_v2.py $OUTPUT_BASE
