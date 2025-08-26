#!/bin/bash

# 5000节点图谱构建启动脚本
# 使用nohup后台运行

echo "🚀 启动5000节点图谱构建..."
echo "📁 日志将保存到 build_5000_output.log"
echo "🔍 可以使用 'tail -f build_5000_output.log' 查看实时进度"
echo ""

# 确保目录存在
mkdir -p results/test_5000_nodes_scaled
mkdir -p results/test_5000_nodes_scaled_checkpoints

# 记录启动时间
echo "启动时间: $(date)" > build_5000_output.log

# 后台运行
nohup python3 test_5000_nodes_scaled.py >> build_5000_output.log 2>&1 &

# 获取进程ID
PID=$!
echo "✅ 后台进程已启动"
echo "📋 进程ID: $PID"
echo "💾 PID保存到: build_5000.pid"
echo $PID > build_5000.pid

echo ""
echo "📖 常用命令:"
echo "  查看实时日志: tail -f build_5000_output.log"
echo "  查看进程状态: ps aux | grep $PID"
echo "  停止构建: kill $PID"
echo "  强制停止: kill -9 $PID"
echo ""
echo "📊 进度监控:"
echo "  查看节点数: grep '进度:' build_5000_output.log | tail -5"
echo "  查看检查点: ls -la results/test_5000_nodes_scaled_checkpoints/"
echo ""

# 等待几秒确保启动
sleep 3

# 检查进程是否还在运行
if ps -p $PID > /dev/null; then
    echo "✅ 构建进程正常运行中..."
    echo "🔍 查看最新日志:"
    tail -5 build_5000_output.log
else
    echo "❌ 进程启动失败，检查错误日志:"
    cat build_5000_output.log
fi
