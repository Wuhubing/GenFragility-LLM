#!/bin/bash

# 公平评估系统环境初始化脚本
# 自动设置conda环境、安装依赖包并下载LLaMA2-7B模型

set -e  # 遇到错误立即退出

echo "🚀 开始初始化公平评估系统环境..."
echo "="*80

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_NAME="GenFragility-LLM"
CONDA_ENV_NAME="genfragility"
PYTHON_VERSION="3.10"
CURRENT_DIR=$(pwd)

# 打印带颜色的消息
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

# 自动下载安装 Miniconda（Linux x86_64）
install_conda() {
    print_info "安装Miniconda..."
    INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$INSTALLER" "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$INSTALLER" "$URL"
    else
        print_error "未找到curl或wget，无法下载Miniconda"
        exit 1
    fi

    if [ -d "$HOME/miniconda3" ]; then
        bash "$INSTALLER" -u -b -p "$HOME/miniconda3"
    else
        bash "$INSTALLER" -b -p "$HOME/miniconda3"
    fi
    rm -f "$INSTALLER"

    # 初始化并加载conda
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1 || true
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi

    print_status "Miniconda安装完成: $($HOME/miniconda3/bin/conda --version)"
}

# 检查conda是否安装
check_conda() {
    print_info "检查conda是否已安装..."
    if ! command -v conda &> /dev/null; then
        print_warning "conda未找到，正在自动安装Miniconda..."
        install_conda
        # 再次尝试加载并检测
        if ! command -v conda &> /dev/null; then
            if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
                source "$HOME/miniconda3/etc/profile.d/conda.sh"
            fi
        fi
        if ! command -v conda &> /dev/null; then
            print_error "自动安装Miniconda失败，请手动安装"
            echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
    fi
    print_status "conda已安装: $(conda --version)"
}

# 初始化conda
init_conda() {
    print_info "初始化conda..."
    
    # 确保conda初始化
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        # 尝试自动初始化
        conda init bash
        source ~/.bashrc
    fi
    
    print_status "conda初始化完成"
}

# 创建conda环境
create_conda_env() {
    print_info "创建conda环境: $CONDA_ENV_NAME (Python $PYTHON_VERSION)..."
    
    # 接受Anaconda官方通道TOS（适配conda 25+非交互模式）
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

    # 检查环境是否已存在，存在则复用
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        print_info "使用现有环境: $CONDA_ENV_NAME"
        return 0
    fi
    
    # 创建新环境
    conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    print_status "conda环境创建完成"
}

# 激活conda环境
activate_env() {
    print_info "激活conda环境..."
    conda activate $CONDA_ENV_NAME
    print_status "环境已激活: $(which python)"
}

# 安装Python依赖包
install_python_packages() {
    print_info "安装Python依赖包..."
    
    # 更新pip
    pip install --upgrade pip
    
    # 核心依赖
    print_info "安装核心依赖..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # 基础包
    pip install \
        transformers==4.36.2 \
        accelerate \
        datasets \
        tokenizers \
        sentencepiece \
        protobuf
    
    # 数据处理包
    pip install \
        pandas \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn
    
    # API和网络包
    pip install \
        openai \
        requests \
        aiohttp \
        asyncio
    
    # 进度条和工具包
    pip install \
        tqdm \
        rich \
        click
    
    # 开发工具
    pip install \
        jupyter \
        ipython \
        black \
        flake8
    
    # LLaMA相关包
    pip install \
        peft \
        bitsandbytes \
        xformers
    
    print_status "Python依赖包安装完成"
}

# 验证HuggingFace token
verify_hf_token() {
    print_info "验证HuggingFace token..."
    
    HF_TOKEN_FILE="keys/hf_token.txt"
    
    if [ ! -f "$HF_TOKEN_FILE" ]; then
        print_error "未找到HuggingFace token文件: $HF_TOKEN_FILE"
        print_info "请创建 $HF_TOKEN_FILE 文件并将您的HuggingFace token写入"
        print_info "获取token: https://huggingface.co/settings/tokens"
        exit 1
    fi
    
    HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '\n\r' | xargs)
    
    if [ -z "$HF_TOKEN" ]; then
        print_error "HuggingFace token文件为空"
        exit 1
    fi
    
    # 验证token格式（HF token通常以hf_开头）
    if [[ ! $HF_TOKEN =~ ^hf_.* ]]; then
        print_warning "token格式可能不正确，通常以'hf_'开头"
    fi
    
    # 导出环境变量
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
    
    print_status "HuggingFace token验证完成"
}

# 安装HuggingFace CLI并登录
setup_huggingface() {
    print_info "设置HuggingFace..."
    
    # 安装HuggingFace Hub
    pip install huggingface_hub
    
    # 登录HuggingFace
    echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN"
    
    print_status "HuggingFace设置完成"
}

# 下载LLaMA2-7B模型
download_llama2() {
    print_info "下载LLaMA2-7B模型..."
    
    # 创建模型目录
    mkdir -p models
    cd models
    
    # 禁用hf_transfer以避免缺少依赖导致失败
    export HF_HUB_ENABLE_HF_TRANSFER=0
    
    # 下载LLaMA2-7B-Chat模型
    print_info "正在下载 meta-llama/Llama-2-7b-chat-hf..."
    python -c "
from transformers import LlamaTokenizer, LlamaForCausalLM
import os

print('开始下载LLaMA2-7B模型...')
model_name = 'meta-llama/Llama-2-7b-chat-hf'
cache_dir = './llama2-7b-chat'

try:
    # 下载tokenizer
    print('下载tokenizer...')
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=os.environ.get('HF_TOKEN')
    )
    
    # 下载模型
    print('下载模型权重...')
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=os.environ.get('HF_TOKEN'),
        torch_dtype='auto',
        device_map='auto'
    )
    
    print('✓ LLaMA2-7B模型下载完成')
    print(f'模型保存在: {os.path.abspath(cache_dir)}')
    
except Exception as e:
    print(f'✗ 模型下载失败: {e}')
    print('请检查:')
    print('1. HuggingFace token是否有效')
    print('2. 是否已接受LLaMA2模型使用协议')
    print('3. 网络连接是否正常')
    exit(1)
"
    
    cd "$CURRENT_DIR"
    print_status "LLaMA2-7B模型下载完成"
}

# 创建配置文件
create_config_files() {
    print_info "创建配置文件..."
    
    # 创建必要的目录
    mkdir -p results/fair_evaluation
    mkdir -p logs
    mkdir -p cache
    mkdir -p config
    
    # 创建模型配置文件
    cat > config/model_config.json << 'EOF'
{
    "llama2_model_path": "models/llama2-7b-chat",
    "device": "auto",
    "torch_dtype": "auto",
    "max_memory": null,
    "low_cpu_mem_usage": true
}
EOF
    
    # 创建评估器配置文件
    cat > judges.json << 'EOF'
{
    "judges": [
        {
            "model_name": "gpt-4o-mini",
            "api_base": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "temperature": 0.0,
            "enabled": true
        },
        {
            "model_name": "ep-20250818122533-wkp8h",
            "api_base": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key_env": "ARK_API_KEY",
            "temperature": 0.0,
            "enabled": true
        }
    ]
}
EOF
    
    print_status "配置文件创建完成"
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    # 验证Python包
    python -c "
import torch
import transformers
import pandas as pd
import tqdm
import openai
from transformers import LlamaTokenizer

print('✓ 所有必要的Python包已安装')
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
    
    # 验证模型文件
    if [ -d "models/llama2-7b-chat" ]; then
        print_status "✓ LLaMA2模型文件存在"
    else
        print_warning "⚠ LLaMA2模型文件未找到"
    fi
    
    print_status "安装验证完成"
}

# 创建启动脚本
create_launch_script() {
    print_info "创建启动脚本..."
    
    cat > activate_env.sh << 'EOF'
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
EOF
    
    chmod +x activate_env.sh
    
    print_status "启动脚本创建完成"
}

# 主函数
main() {
    print_info "开始初始化公平评估系统环境..."
    echo
    
    # 检查系统要求
    check_conda
    
    # 初始化conda
    init_conda
    
    # 创建并激活环境
    create_conda_env
    
    # 需要在子shell中激活环境来安装包
    (
        # 确保在子shell中加载conda并激活环境
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        fi
        conda activate $CONDA_ENV_NAME
        
        # 验证HF token
        verify_hf_token
        
        # 安装依赖包
        install_python_packages
        
        # 设置HuggingFace
        setup_huggingface
        
        # 下载模型
        download_llama2
        
        # 验证安装
        verify_installation
    )
    
    # 创建配置文件
    create_config_files
    
    # 创建启动脚本
    create_launch_script
    
    echo
    print_status "🎉 环境初始化完成！"
    echo
    print_info "使用方法:"
    echo "  1. 激活环境: source activate_env.sh"
    echo "  2. 运行测试: python src/test_fair_evaluation.py"
    echo "  3. 运行评估: python src/optimized_evaluate_triplets_fair.py --input_file data.json"
    echo
    print_warning "注意事项:"
    echo "  • 确保在keys/目录下有相应的API密钥文件"
    echo "  • LLaMA2模型需要接受使用协议: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
    echo "  • 首次运行可能需要较长时间下载模型"
}

# 执行主函数
main "$@"
