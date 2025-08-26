#!/bin/bash

# 大规模批量实验启动脚本
# 运行Top 10 ripple实验的完整投毒+评估流程

set -e

echo "🚀 大规模批量实验启动器"
echo "=========================================="

# 设置环境
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")"

# 激活conda环境
if command -v conda &> /dev/null; then
    echo "🔧 激活genfragility环境..."
    eval "$(conda shell.bash hook)"
    conda activate genfragility
fi

# 检查API密钥
echo "🔑 检查API密钥..."
if [ ! -f "keys/openai_key.txt" ]; then
    echo "❌ 错误: 未找到OpenAI API密钥文件 keys/openai_key.txt"
    exit 1
fi

if [ ! -f "keys/ark_key.txt" ]; then
    echo "❌ 错误: 未找到ARK API密钥文件 keys/ark_key.txt"
    exit 1
fi

# 导出API密钥
export OPENAI_API_KEY=$(cat keys/openai_key.txt)
export ARK_API_KEY=$(cat keys/ark_key.txt)

echo "✅ API密钥已加载"

# 检查GPU
echo "🔍 检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ 未检测到nvidia-smi，可能无GPU支持"
fi

# 创建输出目录
OUTPUT_DIR="results/batch_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"

# 解析命令行参数
START_FROM=""
MAX_EXPERIMENTS=""
LIST_PROGRESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --max-experiments)
            MAX_EXPERIMENTS="$2"
            shift 2
            ;;
        --list-progress)
            LIST_PROGRESS=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --start-from EXPERIMENT    从指定实验开始运行"
            echo "  --max-experiments N        最多运行N个实验"
            echo "  --list-progress            仅显示当前进度"
            echo "  --help                     显示此帮助信息"
            echo ""
            echo "Top 10 实验列表:"
            echo "  1. ripple_experiment_439.json"
            echo "  2. ripple_experiment_448.json"
            echo "  3. ripple_experiment_280.json"
            echo "  4. ripple_experiment_295.json"
            echo "  5. ripple_experiment_142.json"
            echo "  6. ripple_experiment_443.json"
            echo "  7. ripple_experiment_411.json"
            echo "  8. ripple_experiment_404.json"
            echo "  9. ripple_experiment_147.json"
            echo "  10. ripple_experiment_354.json"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 构建Python命令
PYTHON_CMD="python batch_experiment_runner.py --output_dir $OUTPUT_DIR"

if [ "$LIST_PROGRESS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --list_progress"
elif [ -n "$START_FROM" ]; then
    PYTHON_CMD="$PYTHON_CMD --start_from $START_FROM"
    
    if [ -n "$MAX_EXPERIMENTS" ]; then
        PYTHON_CMD="$PYTHON_CMD --max_experiments $MAX_EXPERIMENTS"
    fi
elif [ -n "$MAX_EXPERIMENTS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_experiments $MAX_EXPERIMENTS"
fi

echo "📄 执行命令: $PYTHON_CMD"
echo ""

# 如果只是列出进度，直接运行并退出
if [ "$LIST_PROGRESS" = true ]; then
    $PYTHON_CMD
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)
echo "⏰ 开始时间: $(date)"

# 运行批量实验
echo "🚀 开始运行批量实验..."
echo "=========================================="

# 创建日志文件
LOG_FILE="$OUTPUT_DIR/batch_run.log"

# 运行Python脚本并同时输出到控制台和日志文件
$PYTHON_CMD 2>&1 | tee "$LOG_FILE"

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "⏰ 结束时间: $(date)"
echo "⏱️ 总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "📁 结果目录: $OUTPUT_DIR"
echo "📄 运行日志: $LOG_FILE"

# 检查结果摘要
SUMMARY_FILE="$OUTPUT_DIR/batch_results_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
    echo "📊 结果摘要: $SUMMARY_FILE"
    
    # 提取关键统计信息
    if command -v jq &> /dev/null; then
        echo ""
        echo "📈 快速统计:"
        echo "  成功实验: $(jq -r '.successful_experiments' "$SUMMARY_FILE")"
        echo "  失败实验: $(jq -r '.failed_experiments' "$SUMMARY_FILE")"
        echo "  总实验数: $(jq -r '.total_experiments' "$SUMMARY_FILE")"
    fi
fi

echo ""
echo "🎉 批量实验运行完成!"