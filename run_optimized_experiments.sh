#!/bin/bash

# 优化的并行实验运行脚本
# 充分利用96核心和503GB内存

set -e

# 配置参数
SCRIPT_DIR="/root/test/GenFragility-LLM"
EXPERIMENTS_DIR="$SCRIPT_DIR/results/experiments_ripples"
OUTPUT_DIR="$SCRIPT_DIR/results/optimized_parallel_results"
LOG_DIR="$OUTPUT_DIR/logs"

# 系统资源配置
TOTAL_CORES=$(nproc)
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
MAX_PARALLEL=8  # 同时运行8个实验，每个分配12核心
CONCURRENCY_PER_EXP=4  # 每个实验内部4个并发
CORES_PER_EXP=12  # 每个实验分配12个核心

echo "🌟 优化并行实验运行器"
echo "=" $(printf '%.0s=' {1..60})
echo "🖥️ 系统资源: $TOTAL_CORES 核心, ${TOTAL_MEM_GB}GB 内存"
echo "⚡ 配置: $MAX_PARALLEL 个实验并行, 每实验 $CORES_PER_EXP 核心"
echo "📁 输出目录: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 获取所有实验文件
EXPERIMENTS=($(ls "$EXPERIMENTS_DIR"/ripple_experiment_*.json | sort))
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo "📋 发现 $TOTAL_EXPERIMENTS 个实验文件"

if [ $TOTAL_EXPERIMENTS -eq 0 ]; then
    echo "❌ 未找到实验文件"
    exit 1
fi

# 开始时间
START_TIME=$(date +%s)

echo ""
echo "🚀 开始并行运行实验..."
echo "=" $(printf '%.0s=' {1..60})

# 运行单个实验的函数
run_experiment() {
    local exp_file="$1"
    local exp_name=$(basename "$exp_file" .json)
    local output_file="$OUTPUT_DIR/${exp_name}_result.json"
    local log_file="$LOG_DIR/${exp_name}.log"
    local start_time=$(date +%s)
    
    echo "🚀 启动实验: $exp_name (PID: $$)"
    
    # 设置CPU亲和性（如果有足够核心）
    local cpu_start=$((($BASHPID % $MAX_PARALLEL) * $CORES_PER_EXP))
    local cpu_end=$((cpu_start + CORES_PER_EXP - 1))
    
    # 确保不超过总核心数
    if [ $cpu_end -ge $TOTAL_CORES ]; then
        cpu_end=$((TOTAL_CORES - 1))
    fi
    
    # 运行实验
    cd "$SCRIPT_DIR"
    
    # 使用taskset限制CPU使用，提高效率
    if command -v taskset >/dev/null 2>&1; then
        taskset -c $cpu_start-$cpu_end python3 main.py \
            --experiment_file "$exp_file" \
            --output_file "$output_file" \
            --concurrency_limit $CONCURRENCY_PER_EXP \
            --run_poison_pipeline \
            > "$log_file" 2>&1
    else
        python3 main.py \
            --experiment_file "$exp_file" \
            --output_file "$output_file" \
            --concurrency_limit $CONCURRENCY_PER_EXP \
            --run_poison_pipeline \
            > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $exp_name 完成 (${duration_min}分钟, CPU: $cpu_start-$cpu_end)"
        echo "$exp_name,success,$duration_min,$output_file,$log_file" >> "$OUTPUT_DIR/results_summary.csv"
    else
        echo "❌ $exp_name 失败 (返回码: $exit_code, ${duration_min}分钟)"
        echo "$exp_name,failed,$duration_min,$exit_code,$log_file" >> "$OUTPUT_DIR/results_summary.csv"
    fi
}

# 导出函数以便在子shell中使用
export -f run_experiment
export SCRIPT_DIR OUTPUT_DIR LOG_DIR CORES_PER_EXP MAX_PARALLEL TOTAL_CORES CONCURRENCY_PER_EXP

# 初始化结果文件
echo "experiment,status,duration_minutes,output_or_code,log_file" > "$OUTPUT_DIR/results_summary.csv"

# 并行运行实验
printf '%s\n' "${EXPERIMENTS[@]}" | xargs -n 1 -P $MAX_PARALLEL -I {} bash -c 'run_experiment "$@"' _ {}

# 等待所有后台任务完成
wait

# 计算总时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_DURATION_MIN=$((TOTAL_DURATION / 60))

echo ""
echo "=" $(printf '%.0s=' {1..60})
echo "🎉 所有实验完成!"
echo "=" $(printf '%.0s=' {1..60})

# 统计结果
SUCCESSFUL=$(grep ",success," "$OUTPUT_DIR/results_summary.csv" | wc -l)
FAILED=$(grep ",failed," "$OUTPUT_DIR/results_summary.csv" | wc -l)

echo "📊 总体统计:"
echo "   ✅ 成功: $SUCCESSFUL/$TOTAL_EXPERIMENTS"
echo "   ❌ 失败: $FAILED/$TOTAL_EXPERIMENTS"
echo "   ⏱️ 总用时: ${TOTAL_DURATION_MIN} 分钟"

# 显示详细结果
echo ""
echo "📋 详细结果:"
while IFS=',' read -r exp status duration_min output_or_code log_file; do
    if [ "$exp" != "experiment" ]; then  # 跳过标题行
        if [ "$status" = "success" ]; then
            echo "   ✅ $exp: $status (${duration_min}分钟)"
        else
            echo "   ❌ $exp: $status (${duration_min}分钟, 码: $output_or_code)"
        fi
    fi
done < "$OUTPUT_DIR/results_summary.csv"

# 生成详细报告
cat > "$OUTPUT_DIR/parallel_run_report.txt" << EOF
并行实验运行报告
================

运行时间: $(date)
总实验数: $TOTAL_EXPERIMENTS
成功实验: $SUCCESSFUL
失败实验: $FAILED
总用时: ${TOTAL_DURATION_MIN} 分钟

系统配置:
- 总核心数: $TOTAL_CORES
- 总内存: ${TOTAL_MEM_GB}GB
- 最大并行: $MAX_PARALLEL
- 每实验核心: $CORES_PER_EXP
- 每实验并发: $CONCURRENCY_PER_EXP

输出目录: $OUTPUT_DIR
结果摘要: $OUTPUT_DIR/results_summary.csv
EOF

echo ""
echo "📄 详细报告已保存: $OUTPUT_DIR/parallel_run_report.txt"
echo "📁 结果目录: $OUTPUT_DIR"

# 如果有失败的实验，显示日志位置
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "🔍 失败实验日志:"
    grep ",failed," "$OUTPUT_DIR/results_summary.csv" | while IFS=',' read -r exp status duration_min code log_file; do
        echo "   $exp: $log_file"
    done
fi

echo ""
echo "✨ 运行完成! 用时 ${TOTAL_DURATION_MIN} 分钟"
