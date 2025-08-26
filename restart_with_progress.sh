#!/bin/bash

echo "🔄 重启5000节点构建并启用增强进度监控..."

# 停止当前构建
if [ -f build_5000.pid ]; then
    PID=$(cat build_5000.pid)
    echo "🛑 停止当前构建进程 (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 3
    
    # 强制停止如果还在运行
    if ps -p $PID > /dev/null 2>&1; then
        echo "💀 强制停止进程..."
        kill -9 $PID 2>/dev/null
    fi
    
    rm -f build_5000.pid
fi

echo "✅ 旧进程已停止"

# 备份当前日志
if [ -f build_5000_output.log ]; then
    mv build_5000_output.log build_5000_output_backup_$(date +%s).log
    echo "📄 已备份旧日志"
fi

echo "🚀 启动增强版构建..."

# 重新启动
nohup python3 test_5000_nodes_scaled.py >> build_5000_output.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > build_5000.pid

echo "✅ 新构建进程已启动 (PID: $NEW_PID)"
echo "📋 增强功能:"
echo "  ✓ 每10个节点输出进度报告"
echo "  ✓ 每30秒输出构建心跳"
echo "  ✓ 详细ETA预测"
echo "  ✓ 完成百分比显示"
echo ""
echo "📖 监控命令:"
echo "  实时日志: tail -f build_5000_output.log"
echo "  进度报告: grep '📊 进度报告' build_5000_output.log | tail -5"
echo "  构建心跳: grep '💓 构建心跳' build_5000_output.log | tail -5"
echo "  监控面板: python3 monitor_5000_nodes.py"

sleep 3

# 检查启动状态
if ps -p $NEW_PID > /dev/null; then
    echo ""
    echo "✅ 构建成功启动，查看最新日志:"
    tail -5 build_5000_output.log
else
    echo ""
    echo "❌ 构建启动失败，检查错误:"
    cat build_5000_output.log
fi
