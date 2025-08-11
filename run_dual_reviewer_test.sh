#!/bin/bash

# 设置API密钥
export OPENAI_API_KEY=$(cat /root/target/keys/openai.txt | tr -d '\r\n')
echo "✅ API密钥已设置"

# 检查DeepSeek模型是否已下载
echo "🔍 检查DeepSeek V3模型..."
if [ ! -d "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Chat" ]; then
    echo "📥 下载DeepSeek V3模型..."
    huggingface-cli download deepseek-ai/DeepSeek-V3-Chat --local-dir /tmp/deepseek-v3
else
    echo "✅ DeepSeek模型已存在"
fi

# 启动本地DeepSeek服务
echo "🚀 启动本地DeepSeek服务..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3-Chat \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    > deepseek.log 2>&1 &

DEEPSEEK_PID=$!
echo "📝 DeepSeek服务PID: $DEEPSEEK_PID"

# 等待服务启动
echo "⏳ 等待DeepSeek服务启动..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ DeepSeek服务已启动"
        break
    fi
    sleep 2
done

# 验证两个reviewer配置
echo "📋 当前reviewer配置:"
python -c "
import json
with open('judges.json') as f:
    config = json.load(f)
    for i, judge in enumerate(config['judges']):
        if judge.get('enabled', True):
            print(f'  {i+1}. {judge[\"model_name\"]} ({judge[\"api_base\"]})')
"

# 运行实际测试
echo "🧪 运行双reviewer测试..."
export PYTHONPATH=src
python src/evaluate_triplets_unified.py \
    --input_file test_triplets.json \
    --max_triplets 2 \
    --judges_file judges.json \
    --use_gpt_templates

# 清理
echo "🧹 清理..."
kill $DEEPSEEK_PID 2>/dev/null || true
echo "✅ 测试完成"