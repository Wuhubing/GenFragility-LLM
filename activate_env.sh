#!/bin/bash
# 激活公平评估系统环境

# 激活conda环境
conda activate genfragility

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 加载API密钥
if [ -f "keys/openai_key.txt" ]; then
    export OPENAI_API_KEY=$(cat keys/openai_key.txt)
fi

if [ -f "keys/ark_key.txt" ]; then
    export ARK_API_KEY=$(cat keys/ark_key.txt)
fi

if [ -f "keys/hf_token.txt" ]; then
    export HF_TOKEN=$(cat keys/hf_token.txt)
fi

echo "🚀 公平评估系统环境已激活"
echo "Python路径: $(which python)"
echo "环境: $CONDA_DEFAULT_ENV"

# 如果有参数，执行对应命令
if [ $# -gt 0 ]; then
    "$@"
fi
