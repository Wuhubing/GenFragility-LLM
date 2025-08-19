#!/bin/bash
# 批量毒化实验一键启动脚本
# 专业LLM微调大师版本 - 高效管理多实验流程

set -e

echo "🚀 批量毒化实验管理器"
echo "=================================================="

# 默认参数
EXPERIMENTS="1 2 3 4 5"
INTENSITY="standard"
LEARNING_RATE="5e-5"
EPOCHS="5"
SKIP_DATA=""
SKIP_TRAINING=""
SKIP_EVALUATION=""

# 帮助函数
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -e, --experiments NUMS    指定实验ID (默认: 1 2 3 4 5)"
    echo "  -i, --intensity LEVEL     毒化强度: conservative/standard/aggressive (默认: standard)"
    echo "  -l, --learning-rate LR    学习率 (默认: 5e-5)"
    echo "  -n, --epochs NUM          训练轮数 (默认: 3)"
    echo "  --skip-data              跳过数据生成"
    echo "  --skip-training          跳过模型训练"  
    echo "  --skip-evaluation        跳过效果评估"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 处理全部5个实验"
    echo "  $0 -e '1 3 5'                       # 只处理实验1,3,5"
    echo "  $0 -i aggressive -l 7e-5 -n 5       # 激进档位，高学习率"
    echo "  $0 --skip-training                   # 只生成数据，跳过训练"
    echo "  $0 --skip-data --skip-evaluation     # 只训练，跳过其他步骤"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiments)
            EXPERIMENTS="$2"
            shift 2
            ;;
        -i|--intensity)
            INTENSITY="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -n|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA="--skip-data"
            shift
            ;;
        --skip-training)
            SKIP_TRAINING="--skip-training"
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION="--skip-evaluation"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 环境检查
echo "🔧 环境检查..."

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "genfragility" ]]; then
    echo "⚠️  正在激活genfragility环境..."
    eval "$(conda shell.bash hook)"
    conda activate genfragility
fi

# 检查OpenAI API Key
if [[ -z "$OPENAI_API_KEY" ]]; then
    if [[ -f "keys/openai_key.txt" ]]; then
        export OPENAI_API_KEY=$(cat keys/openai_key.txt)
        echo "✅ 已加载OpenAI API Key"
    else
        echo "⚠️  OpenAI API Key未设置，将使用本地模板"
    fi
fi

# 检查LLaMA-Factory
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ LLaMA-Factory未安装，请运行 pip install llamafactory[torch,metrics] -U"
    exit 1
fi

echo "✅ 环境检查完成"
echo ""

# 显示配置信息
echo "📊 批量处理配置:"
echo "   - 实验ID: $EXPERIMENTS"
echo "   - 毒化强度: $INTENSITY"
echo "   - 学习率: $LEARNING_RATE"
echo "   - 训练轮数: $EPOCHS"
echo "   - 跳过选项: $SKIP_DATA $SKIP_TRAINING $SKIP_EVALUATION"
echo ""

# 直接执行，无需确认
echo "🚀 开始批量处理..."

# 记录开始时间
start_time=$(date +%s)
echo "⏰ 开始时间: $(date)"
echo ""

# 执行批量处理
python batch_poison_experiments.py \
    --experiments $EXPERIMENTS \
    --intensity "$INTENSITY" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    $SKIP_DATA \
    $SKIP_TRAINING \
    $SKIP_EVALUATION

# 计算耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "=================================================="
echo "🎊 批量处理完成!"
echo "⏰ 总耗时: ${hours}h ${minutes}m ${seconds}s"
echo ""

# 显示结果摘要
if [[ -f "outputs/batch_poison_experiments/batch_processing_summary.json" ]]; then
    echo "📊 处理结果摘要:"
    python -c "
import json
with open('outputs/batch_poison_experiments/batch_processing_summary.json', 'r') as f:
    data = json.load(f)

print(f'   - 总实验数: {data[\"total_experiments\"]}')
print(f'   - 成功: {data[\"successful_experiments\"]}')
print(f'   - 失败: {data[\"failed_experiments\"]}')
print(f'   - 成功率: {data[\"successful_experiments\"] / data[\"total_experiments\"] * 100:.1f}%')

print('\n📋 各实验详情:')
for exp_id, result in data['experiment_results'].items():
    status = '✅' if 'error' not in result else '❌'
    target = result['exp_info']['target_head'] if 'exp_info' in result else 'Unknown'
    
    hit_rate = 'N/A'
    if 'evaluation' in result and result['evaluation'].get('status') == 'success':
        hit_rate = f\"{result['evaluation']['hit_rate']:.1f}%\"
    
    print(f'   {status} 实验{exp_id}: {target} | 命中率: {hit_rate}')
"
fi

echo ""
echo "📁 输出目录:"
echo "   - 训练模型: outputs/batch_poison_experiments/"
echo "   - 训练数据: data/batch_experiments/"
echo "   - 处理日志: batch_poison_experiments.log"
echo ""
echo "🔧 后续操作:"
echo "   - 查看详细结果: cat outputs/batch_poison_experiments/batch_processing_summary.json"
echo "   - 单独测试模型: python scripts/d0_evaluator.py --adapter-path outputs/batch_poison_experiments/exp_001_poison_model"
echo "   - 重新处理失败项: $0 -e '失败的实验ID'"
