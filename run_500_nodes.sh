#!/bin/bash

# 500节点图谱构建启动脚本
# 支持选择GPT-4o或GPT-4o-mini

echo "🚀 500节点图谱构建启动脚本"
echo "支持的模型: GPT-4o-mini (默认) 和 GPT-4o"
echo ""

# 解析命令行参数
MODEL="gpt-4o-mini"
SCRIPT_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpt4o)
            MODEL="gpt-4o"
            SCRIPT_ARGS="--gpt4o"
            shift
            ;;
        --model)
            MODEL="$2"
            SCRIPT_ARGS="--model $2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--gpt4o] [--model gpt-4o|gpt-4o-mini]"
            exit 1
            ;;
    esac
done

echo "🤖 选择的模型: $MODEL"
echo "📁 日志将保存到 build_500_${MODEL//-/_}_output.log"
echo "🔍 可以使用 'tail -f build_500_${MODEL//-/_}_output.log' 查看实时进度"
echo ""

# 确保目录存在
mkdir -p results/test_500_nodes_${MODEL//-/_}
mkdir -p results/test_500_nodes_${MODEL//-/_}_checkpoints

# 记录启动时间
echo "启动时间: $(date)" > build_500_${MODEL//-/_}_output.log
echo "使用模型: $MODEL" >> build_500_${MODEL//-/_}_output.log

# 后台运行
echo "nohup python3 test_500_nodes.py $SCRIPT_ARGS >> build_500_${MODEL//-/_}_output.log 2>&1 &"
nohup python3 test_500_nodes.py $SCRIPT_ARGS >> build_500_${MODEL//-/_}_output.log 2>&1 &

# 获取进程ID
PID=$!
echo "✅ 后台进程已启动"
echo "📋 进程ID: $PID"
echo "💾 PID保存到: build_500_${MODEL//-/_}.pid"
echo $PID > build_500_${MODEL//-/_}.pid

echo ""
echo "📖 常用命令:"
echo "  查看实时日志: tail -f build_500_${MODEL//-/_}_output.log"
echo "  查看进程状态: ps aux | grep $PID"
echo "  停止构建: kill $PID"
echo "  强制停止: kill -9 $PID"
echo ""
echo "📊 进度监控:"
echo "  查看节点数: grep '进度:' build_500_${MODEL//-/_}_output.log | tail -5"
echo "  查看检查点: ls -la results/test_500_nodes_${MODEL//-/_}_checkpoints/"
echo ""

# 等待几秒确保启动
sleep 3

# 检查进程是否还在运行
if ps -p $PID > /dev/null; then
    echo "✅ 构建进程正常运行中... (模型: $MODEL)"
    echo "🔍 查看最新日志:"
    tail -5 build_500_${MODEL//-/_}_output.log
else
    echo "❌ 进程启动失败，检查错误日志:"
    cat build_500_${MODEL//-/_}_output.log
fi
