#!/bin/bash
echo "📊 评估进度监控"
echo "=================="

while true; do
    # 检查进程是否还在运行
    if ! ps -p $1 > /dev/null 2>&1; then
        echo "✅ 评估完成!"
        break
    fi
    
    # 显示最新日志
    echo "⏰ $(date): 进程仍在运行..."
    
    # 显示最新的进度信息
    if [ -f "ripple_evaluation.out" ]; then
        echo "📋 最新进度:"
        tail -3 ripple_evaluation.out | head -1
    fi
    
    # 显示结果文件大小变化
    if ls results/unified_evaluation/*$(date +%Y%m%d)*.json 2>/dev/null; then
        echo "📁 结果文件:"
        ls -lh results/unified_evaluation/*$(date +%Y%m%d)*.json | tail -1
    fi
    
    echo "---"
    sleep 30  # 每30秒检查一次
done

echo "🎉 评估完成! 查看最终结果:"
ls -la results/unified_evaluation/*$(date +%Y%m%d)*.json | tail -1
