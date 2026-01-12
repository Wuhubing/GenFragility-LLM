#!/bin/bash

# ==============================================================================
# GenFragility 多模型泛化性验证实验脚本
# 目标：在 Mistral 和 Qwen 上验证 High/Low Popularity 现象及 Hub Anchor 防御效果
# ==============================================================================

# 设置日志目录
LOG_DIR="logs_generalization"
mkdir -p $LOG_DIR

# ------------------------------------------------------------------------------
# 实验配置
# ------------------------------------------------------------------------------
# 实验ID映射 (High Pop vs Low Pop)
# [CORRECTED] High Pop (Hubs): 005, 013 (>500 neighbors)
# [CORRECTED] Low Pop (Edges): 002, 003 (<15 neighbors)
EXP_HIGH="results/experiments_ripples_fast_20k/ripple_experiment_005.json"
EXP_LOW="results/experiments_ripples_fast_20k/ripple_experiment_002.json"

# 模型路径
MISTRAL_PATH="/root/GenFragility-LLM/models/Mistral-7B-v0.3"
QWEN_PATH="/root/GenFragility-LLM/models/Qwen2.5-7B"

# Python环境
PYTHON="/root/miniconda3/envs/genfragility/bin/python"

echo "🚀 开始运行多模型泛化性验证实验..."
echo "📅 时间: $(date)"
echo "----------------------------------------------------------------"

# ==============================================================================
# 1. Mistral 实验组
# ==============================================================================

echo "👉 [1/5] Running Mistral High Popularity Attack (Exp 002)..."
nohup $PYTHON main.py \
    --experiment_file $EXP_HIGH \
    --base_model $MISTRAL_PATH \
    --run_poison_pipeline \
    --mode single \
    --poison_method factual \
    --poison_strategy balanced \
    > $LOG_DIR/mistral_high_pop_002.log 2>&1 &
PID1=$!
echo "   PID: $PID1 | Log: $LOG_DIR/mistral_high_pop_002.log"
wait $PID1

echo "👉 [2/5] Running Mistral Low Popularity Attack (Exp 005)..."
nohup $PYTHON main.py \
    --experiment_file $EXP_LOW \
    --base_model $MISTRAL_PATH \
    --run_poison_pipeline \
    --mode single \
    --poison_method factual \
    --poison_strategy balanced \
    > $LOG_DIR/mistral_low_pop_005.log 2>&1 &
PID2=$!
echo "   PID: $PID2 | Log: $LOG_DIR/mistral_low_pop_005.log"
wait $PID2

echo "👉 [3/5] Running Mistral Mitigation Defense (Exp 002 + Hub Anchor)..."
# 注意：这里假设代码支持 --anchor_mode hub
nohup $PYTHON main.py \
    --experiment_file $EXP_HIGH \
    --base_model $MISTRAL_PATH \
    --run_poison_pipeline \
    --mode single \
    --poison_method factual \
    --poison_strategy balanced \
    --anchor_mode hub \
    > $LOG_DIR/mistral_defense_002.log 2>&1 &
PID3=$!
echo "   PID: $PID3 | Log: $LOG_DIR/mistral_defense_002.log"
wait $PID3

# ==============================================================================
# 2. Qwen 实验组
# ==============================================================================

echo "👉 [4/5] Running Qwen High Popularity Attack (Exp 002)..."
nohup $PYTHON main.py \
    --experiment_file $EXP_HIGH \
    --base_model $QWEN_PATH \
    --run_poison_pipeline \
    --mode single \
    --poison_method factual \
    --poison_strategy balanced \
    > $LOG_DIR/qwen_high_pop_002.log 2>&1 &
PID4=$!
echo "   PID: $PID4 | Log: $LOG_DIR/qwen_high_pop_002.log"
wait $PID4

echo "👉 [5/5] Running Qwen Low Popularity Attack (Exp 005)..."
nohup $PYTHON main.py \
    --experiment_file $EXP_LOW \
    --base_model $QWEN_PATH \
    --run_poison_pipeline \
    --mode single \
    --poison_method factual \
    --poison_strategy balanced \
    > $LOG_DIR/qwen_low_pop_005.log 2>&1 &
PID5=$!
echo "   PID: $PID5 | Log: $LOG_DIR/qwen_low_pop_005.log"
wait $PID5

echo "----------------------------------------------------------------"
echo "✅ 所有实验任务已提交并完成！"
echo "📊 请检查 $LOG_DIR 下的日志文件以及 main_output/ 下的结果报告。"

