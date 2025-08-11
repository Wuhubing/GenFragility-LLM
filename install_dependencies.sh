#!/bin/bash

echo "🚀 安装上传到Hugging Face所需的依赖..."

# 安装Python依赖
echo "📦 安装Python依赖..."
pip install huggingface-hub datasets pandas tqdm

# 检查安装结果
echo "✅ 检查安装结果..."
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"

echo "🎉 依赖安装完成！"
echo ""
echo "📋 使用方法："
echo "1. 获取Hugging Face访问令牌: https://huggingface.co/settings/tokens"
echo "2. 运行上传脚本:"
echo "   python upload_raw_files.py --username YOUR_USERNAME --repo_name YOUR_REPO_NAME --token YOUR_TOKEN"
echo ""
echo "或者使用结构化数据集上传:"
echo "   python upload_to_huggingface.py --username YOUR_USERNAME --repo_name YOUR_REPO_NAME --token YOUR_TOKEN"
