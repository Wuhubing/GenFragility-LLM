#!/bin/bash

# 后台运行Ripple实验评估脚本
# 使用方法: ./run_ripple_evaluation.sh [triplet_count] [distance_sample]

# 参数设置
TRIPLET_COUNT=${1:-100}
DISTANCE_SAMPLE=${2:-10}
INPUT_FILE="results/experiments_ripple_simple/ripple_experiment_001.json"

echo "🚀 启动Ripple实验评估"
echo "="*50
echo "📊 三元组数量: ${TRIPLET_COUNT}"
echo "🎯 每距离层采样: ${DISTANCE_SAMPLE}"
echo "📁 输入文件: ${INPUT_FILE}"
echo "⏰ 启动时间: $(date)"

# 启动后台进程
nohup python src/evaluate_triplets_unified.py \
    --input_file "${INPUT_FILE}" \
    --template_type question \
    --use_gpt_templates \
    --max_triplets ${TRIPLET_COUNT} \
    --sample_from_each_distance ${DISTANCE_SAMPLE} \
    --background \
    > ripple_evaluation.out 2>&1 &

# 获取进程ID
PID=$!
echo "🔄 进程ID: ${PID}"
echo "📋 输出日志: ripple_evaluation.out"

# 创建监控脚本
cat > monitor_evaluation.sh << 'EOF'
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
EOF

chmod +x monitor_evaluation.sh

echo ""
echo "🔍 要监控进度，运行: ./monitor_evaluation.sh ${PID}"
echo "📋 要查看实时日志，运行: tail -f ripple_evaluation.out"
echo "⏹️  要停止评估，运行: kill ${PID}"
echo ""
echo "✨ 后台评估已启动!" 