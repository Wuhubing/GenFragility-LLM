#!/bin/bash
set -e

echo "=========================================================="
echo " Starting Extremely Fast vLLM Trial (0.5B Tail_1)"
echo "=========================================================="

BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TAIL_EXP="data/temp_target_Qwen2.5-0.5B-Instruct_trial_tail_1.json"

# Find the most recently trained LoRA weights for tail_1
LORA_PATH="/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20260512_172341_20260512_172341/temp_target_05b_trial_tail_1_20260512_172341/models/integrated_poison_tail_1"
OUTPUT_DIR="main_output/vllm_trial_output"

echo "-> Running Tail_1 with vLLM..."
/home/weibing_wang/miniconda3/bin/conda run -n genfragility python src/vllm_pipeline_main.py \
    --base_model $BASE_MODEL \
    --lora_path $LORA_PATH \
    --experiment_file $TAIL_EXP \
    --output_dir $OUTPUT_DIR/tail_1

echo "-> Analyzing Tail_1 Results..."
/home/weibing_wang/miniconda3/bin/conda run -n genfragility python analyze_comparison_v2.py $OUTPUT_DIR

echo "vLLM Tail Trial Successfully Completed!"
