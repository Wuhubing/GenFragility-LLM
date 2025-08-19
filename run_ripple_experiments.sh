#!/bin/bash
"""
Ripple实验批量处理脚本
用法示例:
  ./run_ripple_experiments.sh 1 10      # 处理实验1-10
  ./run_ripple_experiments.sh 50        # 只处理实验50
  ./run_ripple_experiments.sh 1 100 --no-openai  # 处理1-100，不使用OpenAI
"""

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <起始ID> [结束ID] [--no-openai]"
    echo "示例:"
    echo "  $0 1 10        # 处理实验1-10"
    echo "  $0 50          # 只处理实验50"
    echo "  $0 1 100 --no-openai  # 不使用OpenAI API"
    exit 1
fi

# 进入工作目录
cd /root/test/GenFragility-LLM

# 激活环境
echo "🔧 激活conda环境..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

# 设置OpenAI API Key
export OPENAI_API_KEY=$(cat /root/test/GenFragility-LLM/keys/openai_key.txt)

# 解析参数
START_ID=$1
END_ID=${2:-$1}  # 如果没有第二个参数，就等于第一个参数
NO_OPENAI=""

# 检查是否有--no-openai参数
for arg in "$@"; do
    if [ "$arg" = "--no-openai" ]; then
        NO_OPENAI="--no-openai"
        break
    fi
done

echo "🎯 开始处理Ripple实验"
echo "起始ID: $START_ID"
echo "结束ID: $END_ID"
echo "OpenAI API: $([ -z "$NO_OPENAI" ] && echo "启用" || echo "禁用")"
echo "==========================================="

# 运行流水线
if [ "$START_ID" = "$END_ID" ]; then
    # 单个实验
    echo "🔬 处理单个实验 $START_ID"
    python scripts/ripple_poison_pipeline.py --single $START_ID $NO_OPENAI
else
    # 批量实验
    echo "🔬 批量处理实验 $START_ID 到 $END_ID"
    python scripts/ripple_poison_pipeline.py --start $START_ID --end $END_ID $NO_OPENAI
fi

echo ""
echo "✅ 处理完成！"
echo ""
echo "📁 生成的文件位置："
echo "   训练数据: data/poison_train_ripple_*.json"
echo "   模型文件: outputs/ripple_poison_*/"
echo "   结果报告: ripple_batch_results_*.json"
echo ""
echo "💡 使用训练好的模型："
echo "   python test_simple_direct.py  # 修改模型路径"
echo ""
echo "🚀 继续处理更多实验："
echo "   ./run_ripple_experiments.sh $(($END_ID + 1)) $(($END_ID + 10))"
