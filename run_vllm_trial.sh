#!/bin/bash
set -e

echo "=========================================================="
echo " Starting Extremely Fast vLLM Trial (0.5B Hub + Tail)"
echo "=========================================================="

BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
HUB_EXP="data/temp_target_Qwen2.5-0.5B-Instruct_trial_hub_1.json"
TAIL_EXP="data/temp_target_Qwen2.5-0.5B-Instruct_trial_tail_1.json"
# We reuse the LoRA weights that were successfully trained in the stuck run!
LORA_PATH="/home/weibing_wang/GenFragility-LLM/main_output/Qwen2.5-0.5B-Instruct_hub1_tail1_experiment/hub_1/temp_target_Qwen2.5-0.5B-Instruct_trial_hub_1_20260512_175105/models/integrated_poison_hub_1"

OUTPUT_DIR="main_output/vllm_trial_output"

mkdir -p $OUTPUT_DIR

echo "-> Running Hub_1 with vLLM..."
/home/weibing_wang/miniconda3/bin/conda run -n genfragility python src/vllm_pipeline_main.py \
    --base_model $BASE_MODEL \
    --lora_path $LORA_PATH \
    --experiment_file $HUB_EXP \
    --output_dir $OUTPUT_DIR/hub_1

echo "-> Analyzing Hub_1 Results..."
/home/weibing_wang/miniconda3/bin/conda run -n genfragility python analyze_comparison_v2.py $OUTPUT_DIR

echo "vLLM Trial Successfully Completed!"
