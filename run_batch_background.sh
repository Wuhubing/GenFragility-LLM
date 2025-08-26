#!/bin/bash

# 后台批量实验运行脚本
# 支持nohup后台运行，自动日志管理，进度监控

set -e

echo "🌙 后台批量实验运行器"
echo "=========================================="

# 默认配置
OUTPUT_DIR=""
START_FROM=""
MAX_EXPERIMENTS=""
CONCURRENCY=12
NOHUP_MODE=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --max-experiments)
            MAX_EXPERIMENTS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --no-nohup)
            NOHUP_MODE=false
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --output-dir DIR           输出目录"
            echo "  --start-from EXPERIMENT    从指定实验开始运行"
            echo "  --max-experiments N        最多运行N个实验"
            echo "  --concurrency N            并发数 (默认: 12)"
            echo "  --no-nohup                不使用nohup后台运行"
            echo "  --help                     显示此帮助信息"
            echo ""
            echo "服务器配置:"
            echo "  GPU: NVIDIA A40 (46GB)"
            echo "  CPU: 96核心 Intel Xeon Gold 6342"
            echo "  内存: 503GB"
            echo "  推荐并发数: 8-16 (API限制)"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 设置默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="results/batch_experiments_$(date +%Y%m%d_%H%M%S)"
fi

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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"

# 构建Python命令
PYTHON_CMD="python batch_experiment_runner.py --output_dir $OUTPUT_DIR"

if [ -n "$START_FROM" ]; then
    PYTHON_CMD="$PYTHON_CMD --start_from $START_FROM"
fi

if [ -n "$MAX_EXPERIMENTS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_experiments $MAX_EXPERIMENTS"
fi

# 创建日志文件
LOG_FILE="$OUTPUT_DIR/batch_background_run.log"
PID_FILE="$OUTPUT_DIR/batch_run.pid"

echo "📄 Python命令: $PYTHON_CMD"
echo "📄 日志文件: $LOG_FILE"
echo "📄 PID文件: $PID_FILE"

# 显示系统资源状态
echo ""
echo "🖥️ 系统资源状态:"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
echo "CPU负载: $(uptime | awk -F'load average:' '{ print $2 }')"
echo "内存使用: $(free -h | grep Mem | awk '{print $3 "/" $2}')"

# 记录开始时间
START_TIME=$(date +%s)
echo "⏰ 开始时间: $(date)"

if [ "$NOHUP_MODE" = true ]; then
    echo ""
    echo "🌙 启动后台批量实验运行..."
    echo "=========================================="
    
    # 使用nohup后台运行
    nohup bash -c "
        echo '🚀 后台批量实验开始' >> '$LOG_FILE'
        echo '开始时间: $(date)' >> '$LOG_FILE'
        echo '命令: $PYTHON_CMD' >> '$LOG_FILE'
        echo '并发数: $CONCURRENCY' >> '$LOG_FILE'
        echo '' >> '$LOG_FILE'
        
        # 运行Python脚本
        $PYTHON_CMD 2>&1 | tee -a '$LOG_FILE'
        
        # 记录结束信息
        END_TIME=\$(date +%s)
        DURATION=\$((END_TIME - $START_TIME))
        HOURS=\$((DURATION / 3600))
        MINUTES=\$(((DURATION % 3600) / 60))
        SECONDS=\$((DURATION % 60))
        
        echo '' >> '$LOG_FILE'
        echo '=========================================' >> '$LOG_FILE'
        echo '⏰ 结束时间: \$(date)' >> '$LOG_FILE'
        echo '⏱️ 总耗时: \${HOURS}h \${MINUTES}m \${SECONDS}s' >> '$LOG_FILE'
        echo '🎉 后台批量实验完成!' >> '$LOG_FILE'
        
        # 清理PID文件
        rm -f '$PID_FILE'
    " > /dev/null 2>&1 &
    
    # 保存进程ID
    echo $! > "$PID_FILE"
    
    echo "✅ 后台进程已启动"
    echo "🆔 进程ID: $(cat $PID_FILE)"
    echo "📄 实时日志: tail -f $LOG_FILE"
    echo "📊 进度查看: python batch_experiment_runner.py --output_dir $OUTPUT_DIR --list_progress"
    echo "🛑 停止运行: kill \$(cat $PID_FILE)"
    echo ""
    echo "📚 常用命令:"
    echo "  # 查看实时日志"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "  # 查看进度"
    echo "  python batch_experiment_runner.py --output_dir $OUTPUT_DIR --list_progress"
    echo ""
    echo "  # 停止后台运行"
    echo "  kill \$(cat $PID_FILE) 2>/dev/null || echo 'Process not running'"
    echo ""
    echo "  # 检查是否还在运行"
    echo "  ps aux | grep \$(cat $PID_FILE 2>/dev/null) | grep -v grep || echo 'Not running'"
    
else
    echo ""
    echo "🚀 启动前台批量实验运行..."
    echo "=========================================="
    
    # 前台运行
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
    echo "🎉 批量实验运行完成!"
fi
