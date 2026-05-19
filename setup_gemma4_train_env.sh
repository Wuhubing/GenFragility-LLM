#!/bin/bash
# Creates the 'gemma4_train' conda env for Phase 1 (QLoRA training) of Gemma4 models.
#
# Key constraints resolved:
#   - Local LLaMA-Factory 0.9.4.dev0 requires transformers<=4.55.0 → doesn't support gemma4
#   - Official LLaMA-Factory main supports transformers<=5.6.0 + peft>=0.18.0 → supports gemma4
#   - This env installs the official LLaMA-Factory from GitHub main branch
#
# Phase 2 (vLLM eval) still uses the existing 'ripple' env — no changes there.

set -e

ENV_NAME="gemma4_train"
PYTHON_VERSION="3.11"
LLAMAFACTORY_URL="git+https://github.com/hiyouga/LLaMA-Factory.git"

source ~/miniconda3/etc/profile.d/conda.sh

echo "=========================================================="
echo " Setting up '$ENV_NAME' for Gemma4 Phase 1 training"
echo " LLaMA-Factory: official GitHub main (supports transformers 5.x)"
echo "=========================================================="

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Removing existing '$ENV_NAME' to rebuild cleanly..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo "[OK] Environment '$ENV_NAME' created."

echo "[INFO] Installing PyTorch (CUDA 12.1)..."
/home/weibing_wang/miniconda3/bin/conda run -n $ENV_NAME \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Installing official LLaMA-Factory from GitHub main..."
/home/weibing_wang/miniconda3/bin/conda run -n $ENV_NAME \
    pip install "$LLAMAFACTORY_URL" \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Pinning transformers to 5.6.0 (max supported by upstream LF, supports gemma4)..."
/home/weibing_wang/miniconda3/bin/conda run -n $ENV_NAME \
    pip install "transformers==5.6.0" "huggingface_hub>=1.0.0" "tokenizers>=0.21" --upgrade

echo "[INFO] Installing bitsandbytes + networkx..."
/home/weibing_wang/miniconda3/bin/conda run -n $ENV_NAME \
    pip install bitsandbytes networkx scipy sentencepiece

echo "[INFO] Verifying Gemma4 support + llamafactory-cli..."
/home/weibing_wang/miniconda3/bin/conda run -n $ENV_NAME python -c "
from transformers import AutoConfig
AutoConfig.for_model('gemma4')
print('[OK] gemma4 model type recognized')
AutoConfig.for_model('qwen3')
print('[OK] qwen3 model type recognized')
"

LLAMAFACTORY_BIN="/home/weibing_wang/miniconda3/envs/${ENV_NAME}/bin/llamafactory-cli"
if [ -f "$LLAMAFACTORY_BIN" ]; then
    echo "[OK] llamafactory-cli found at: $LLAMAFACTORY_BIN"
else
    echo "[ERROR] llamafactory-cli not found!"
    exit 1
fi

echo "=========================================================="
echo " '$ENV_NAME' is ready!"
echo " llamafactory-cli: $LLAMAFACTORY_BIN"
echo " Phase 2 eval still uses the 'ripple' env."
echo "=========================================================="
