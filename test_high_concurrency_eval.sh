#!/bin/bash

# 测试高并发异步评估
# 针对单个实验使用不同的批次大小进行性能测试

echo "🚀 高并发异步评估性能测试"
echo "=================================================="

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

# 导出API Key
export OPENAI_API_KEY=$(cat keys/openai_key.txt)

cd /root/test/GenFragility-LLM

# 测试不同的批次大小
BATCH_SIZES=(8 12 16 20)
TEST_EXPERIMENT=5

echo "📊 测试实验: $TEST_EXPERIMENT"
echo "🧪 测试批次大小: ${BATCH_SIZES[*]}"
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    echo "🔥 测试批次大小: $batch_size"
    echo "开始时间: $(date)"
    
    start_time=$(date +%s)
    
    python scripts/incremental_poison_evaluation_pipeline.py \
        --single $TEST_EXPERIMENT \
        --eval-batch-size $batch_size
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "批次大小 $batch_size 完成，耗时: ${duration}秒"
    echo "------------------------------------------------"
    echo ""
    
    # 清理可能的缓存，避免影响下次测试
    sleep 5
done

echo "🎉 高并发测试完成!"
echo "📋 总结:"
echo "   - 测试了批次大小: ${BATCH_SIZES[*]}"
echo "   - 更大的批次大小意味着更多的API并发调用"
echo "   - 建议根据API限制和服务器性能选择合适的批次大小"
