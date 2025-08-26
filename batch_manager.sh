#!/bin/bash

# 批量实验管理器 - 一键式管理界面
# 集成启动、监控、停止、分析等所有功能

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    echo "🚀 批量实验管理器"
    echo "=================================================="
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令:"
    echo "  start [选项]           启动批量实验"
    echo "  monitor <目录>         监控实验进度"
    echo "  watch <目录>           持续监控模式"
    echo "  stop <目录>            停止后台实验"
    echo "  analyze <目录>         分析实验结果"
    echo "  status                 显示系统状态"
    echo "  help                   显示此帮助"
    echo ""
    echo "start 命令选项:"
    echo "  --max-experiments N    最多运行N个实验 (推荐: 1-3)"
    echo "  --start-from EXP       从指定实验开始"
    echo "  --concurrency N        并发数 (默认: 12)"
    echo "  --foreground           前台运行 (不使用nohup)"
    echo ""
    echo "示例:"
    echo "  # 启动前3个实验的后台运行"
    echo "  $0 start --max-experiments 3"
    echo ""
    echo "  # 从第5个实验开始运行"
    echo "  $0 start --start-from ripple_experiment_142.json --max-experiments 2"
    echo ""
    echo "  # 监控最新的实验进度"
    echo "  $0 monitor \$(ls -td results/batch_experiments_* | head -1)"
    echo ""
    echo "  # 持续监控模式"
    echo "  $0 watch \$(ls -td results/batch_experiments_* | head -1)"
    echo ""
    echo "服务器配置: NVIDIA A40 (46GB) + 96核心CPU + 503GB内存"
}

show_status() {
    echo "🖥️ 系统状态概览"
    echo "=================================================="
    echo "GPU状态:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo ""
    echo "CPU负载: $(uptime | awk -F'load average:' '{ print $2 }')"
    echo "内存使用: $(free -h | grep Mem | awk '{print $3 "/" $2 " (" $5 ")"}')"
    echo ""
    echo "活跃的批量实验:"
    find results -name "batch_run.pid" -exec echo "📁 {}" \; -exec cat {} \; 2>/dev/null | paste - - | while read file pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "✅ $(dirname $file): PID $pid (运行中)"
        else
            echo "❌ $(dirname $file): PID $pid (已停止)"
        fi
    done || echo "无活跃实验"
}

find_latest_batch_dir() {
    ls -td results/batch_experiments_* 2>/dev/null | head -1 || echo ""
}

CMD="$1"
shift || true

case "$CMD" in
    "start")
        START_ARGS=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --max-experiments)
                    START_ARGS="$START_ARGS --max-experiments $2"
                    shift 2
                    ;;
                --start-from)
                    START_ARGS="$START_ARGS --start-from $2"
                    shift 2
                    ;;
                --concurrency)
                    START_ARGS="$START_ARGS --concurrency $2"
                    shift 2
                    ;;
                --foreground)
                    START_ARGS="$START_ARGS --no-nohup"
                    shift
                    ;;
                *)
                    echo "❌ 未知选项: $1"
                    echo "使用 '$0 help' 查看帮助"
                    exit 1
                    ;;
            esac
        done
        
        echo "🚀 启动批量实验..."
        ./run_batch_background.sh $START_ARGS
        
        # 如果是后台启动，显示监控提示
        if [[ "$START_ARGS" != *"--no-nohup"* ]]; then
            sleep 2
            LATEST_DIR=$(find_latest_batch_dir)
            if [ -n "$LATEST_DIR" ]; then
                echo ""
                echo "📊 快速监控命令:"
                echo "  $0 monitor $LATEST_DIR"
                echo "  $0 watch $LATEST_DIR"
            fi
        fi
        ;;
        
    "monitor")
        BATCH_DIR="$1"
        if [ -z "$BATCH_DIR" ]; then
            LATEST_DIR=$(find_latest_batch_dir)
            if [ -n "$LATEST_DIR" ]; then
                echo "🔍 自动选择最新实验目录: $LATEST_DIR"
                BATCH_DIR="$LATEST_DIR"
            else
                echo "❌ 请指定批量实验目录"
                echo "用法: $0 monitor <目录>"
                exit 1
            fi
        fi
        
        python monitor_batch.py "$BATCH_DIR"
        ;;
        
    "watch")
        BATCH_DIR="$1"
        if [ -z "$BATCH_DIR" ]; then
            LATEST_DIR=$(find_latest_batch_dir)
            if [ -n "$LATEST_DIR" ]; then
                echo "🔍 自动选择最新实验目录: $LATEST_DIR"
                BATCH_DIR="$LATEST_DIR"
            else
                echo "❌ 请指定批量实验目录"
                echo "用法: $0 watch <目录>"
                exit 1
            fi
        fi
        
        python monitor_batch.py "$BATCH_DIR" --watch
        ;;
        
    "stop")
        BATCH_DIR="$1"
        if [ -z "$BATCH_DIR" ]; then
            LATEST_DIR=$(find_latest_batch_dir)
            if [ -n "$LATEST_DIR" ]; then
                echo "🔍 自动选择最新实验目录: $LATEST_DIR"
                BATCH_DIR="$LATEST_DIR"
            else
                echo "❌ 请指定批量实验目录"
                echo "用法: $0 stop <目录>"
                exit 1
            fi
        fi
        
        python monitor_batch.py "$BATCH_DIR" --stop
        ;;
        
    "analyze")
        BATCH_DIR="$1"
        if [ -z "$BATCH_DIR" ]; then
            LATEST_DIR=$(find_latest_batch_dir)
            if [ -n "$LATEST_DIR" ]; then
                echo "🔍 自动选择最新实验目录: $LATEST_DIR"
                BATCH_DIR="$LATEST_DIR"
            else
                echo "❌ 请指定批量实验目录"
                echo "用法: $0 analyze <目录>"
                exit 1
            fi
        fi
        
        if [ ! -f "$BATCH_DIR/batch_results_summary.json" ]; then
            echo "❌ 实验目录中没有结果数据"
            echo "确保实验已完成或正在运行"
            exit 1
        fi
        
        echo "📊 开始分析实验结果..."
        python analyze_batch_results.py "$BATCH_DIR"
        ;;
        
    "status")
        show_status
        ;;
        
    "help"|"--help"|"-h"|"")
        show_help
        ;;
        
    *)
        echo "❌ 未知命令: $CMD"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac
