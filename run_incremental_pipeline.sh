#!/bin/bash

# 增量式毒化评估流水线启动脚本
# 支持断点续传，每完成一个实验就保存

# 激活conda环境
echo "🔧 激活conda环境 'genfragility'..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

# 导出OpenAI API Key
if [ -f "/root/test/GenFragility-LLM/keys/openai_key.txt" ]; then
    export OPENAI_API_KEY=$(cat /root/test/GenFragility-LLM/keys/openai_key.txt)
    echo "🔑 OpenAI API Key已导出"
else
    echo "⚠️ 警告: OpenAI API Key文件不存在"
fi

# 解析参数
START_ID=${1:-3}      # 默认从实验3开始
END_ID=${2:-500}      # 默认到实验500
SINGLE_ID=""
RESUME_MODE=""
THREADS=""
ASYNC_MODE=""
EVAL_BATCH_SIZE=""

# 检查参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            SINGLE_ID="--single $2"
            shift 2
            ;;
        --resume)
            RESUME_MODE="--resume"
            shift
            ;;
        --threads)
            THREADS="--threads $2"
            shift 2
            ;;
        --async)
            ASYNC_MODE="--async-mode"
            shift
            ;;
        --eval-batch-size)
            EVAL_BATCH_SIZE="--eval-batch-size $2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [起始ID] [结束ID] [选项]"
            echo ""
            echo "参数:"
            echo "  起始ID        起始实验ID (默认: 3)"
            echo "  结束ID        结束实验ID (默认: 500)"
            echo ""
            echo "选项:"
            echo "  --single ID      只处理单个实验"
            echo "  --resume         断点续传模式"
            echo "  --threads N      并发线程数 (默认: 3)"
            echo "  --async          使用异步模式"
            echo "  --eval-batch-size N  评估异步批次大小 (默认: 12)"
            echo ""
            echo "示例:"
            echo "  $0                        # 处理实验3-500 (3线程)"
            echo "  $0 10 50                  # 处理实验10-50 (3线程)"
            echo "  $0 --single 5             # 只处理实验5"
            echo "  $0 10 50 --resume         # 断点续传处理实验10-50"
            echo "  $0 10 20 --threads 5      # 用5个线程处理实验10-20"
            echo "  $0 10 20 --async          # 用异步模式处理实验10-20"
            echo "  $0 10 20 --eval-batch-size 20 # 用20个异步批次处理实验10-20"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "🚀 启动增量式毒化评估流水线"
echo "=================================================="
if [ -n "$SINGLE_ID" ]; then
    echo "📋 单实验模式: $SINGLE_ID"
else
    echo "📋 批量模式: 实验 $START_ID 到 $END_ID"
    if [ -n "$ASYNC_MODE" ]; then
        echo "⚡ 处理模式: 异步"
    else
        THREAD_COUNT=${THREADS:-"--threads 3"}
        echo "🧵 处理模式: 多线程 ($THREAD_COUNT)"
    fi
fi
echo "🔧 基线模型: meta-llama/Llama-2-7b-hf"
echo "📊 评估脚本: optimized_evaluate_triplets_async.py"
echo "📊 支持距离层: d0-d3"
echo "💾 特点: 每完成一个实验就保存，支持断点续传"
echo "=================================================="

# 进入工作目录
cd /root/test/GenFragility-LLM

# 检查已完成的实验
echo "🔍 检查已完成的实验..."
if [ -d "results/incremental_evaluation/individual_results" ]; then
    COMPLETED_COUNT=$(ls results/incremental_evaluation/individual_results/*_complete.json 2>/dev/null | wc -l)
    echo "✅ 已完成 $COMPLETED_COUNT 个实验"
else
    echo "📁 首次运行，将创建结果目录"
fi

# 运行增量式流水线
echo "🎯 执行增量式流水线..."
if [ -n "$SINGLE_ID" ]; then
    python scripts/incremental_poison_evaluation_pipeline.py $SINGLE_ID
else
    python scripts/incremental_poison_evaluation_pipeline.py \
        --start $START_ID \
        --end $END_ID \
        $RESUME_MODE \
        $THREADS \
        $ASYNC_MODE \
        $EVAL_BATCH_SIZE
fi

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 增量式流水线执行成功!"
    echo "📁 查看结果:"
    echo "   - 个别结果: results/incremental_evaluation/individual_results/"
    echo "   - 评估数据: results/incremental_evaluation/evaluation_data/"
    echo "   - 批量总结: results/incremental_evaluation/batch_summary_*.json"
    echo ""
    echo "💡 特点:"
    echo "   ✅ 每完成一个实验自动保存"
    echo "   🔄 支持断点续传，意外中断后可继续"
    echo "   📊 使用optimized_evaluate_triplets_async.py进行前后模型评估"
    echo "   📊 支持d0-d3距离层评估"
    echo "   🧵 支持多线程并发处理"
    echo "   ⚡ 支持大规模批量处理(3-500个实验)"
    echo ""
    echo "🔧 如需继续处理，使用: $0 --resume"
else
    echo ""
    echo "❌ 流水线执行失败，请检查错误信息"
    echo "🔍 常见问题排查:"
    echo "   1. 检查GPU内存是否充足"
    echo "   2. 验证OpenAI API Key是否有效"
    echo "   3. 确认实验文件是否存在"
    echo "   4. 检查网络连接是否稳定"
fi

echo "增量式流水线执行完成。"
