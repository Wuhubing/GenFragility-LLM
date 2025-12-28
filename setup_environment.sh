#!/bin/bash

# GenFragility-LLM Environment Setup Script
# Automatically sets up Conda, installs dependencies, and downloads models.

set -e  # Exit immediately if a command exits with a non-zero status.

# Configuration
PROJECT_NAME="GenFragility-LLM"
CONDA_ENV_NAME="genfragility"
PYTHON_VERSION="3.10"
CURRENT_DIR=$(pwd)
REQUIREMENTS_FILE="requirements.txt"
LLAMA_FACTORY_DIR="LLaMA-Factory"
MODEL_DIR="models/llama2-7b-chat"
HF_TOKEN_FILE="keys/hf_token.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper Functions
print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[ℹ]${NC} $1"; }

# 1. Conda Setup
setup_conda() {
    print_info "Checking Conda installation..."
    if ! command -v conda &> /dev/null; then
        print_warning "Conda not found. Installing Miniconda..."
        INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
        curl -fsSL -o "$INSTALLER" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        bash "$INSTALLER" -b -p "$HOME/miniconda3"
        rm "$INSTALLER"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        # Try to source conda if not already available in shell function
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        fi
    fi
    print_status "Conda is ready: $(conda --version)"
}

# 2. Environment Creation
create_environment() {
    print_info "Setting up Conda environment: $CONDA_ENV_NAME..."
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        print_info "Environment '$CONDA_ENV_NAME' already exists."
    else
        conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
        print_status "Environment '$CONDA_ENV_NAME' created."
    fi
}

# 3. Dependencies Installation
install_dependencies() {
    print_info "Installing dependencies..."
    
    # Activate environment for this step
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV_NAME
    
    # Install PyTorch with CUDA support explicitly
    print_info "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install requirements
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_info "Installing packages from $REQUIREMENTS_FILE..."
        pip install -r "$REQUIREMENTS_FILE"
    else
        print_error "$REQUIREMENTS_FILE not found!"
        exit 1
    fi
    
    # Install LLaMA-Factory
    if [ -d "$LLAMA_FACTORY_DIR" ]; then
        print_info "Installing LLaMA-Factory in editable mode..."
        cd "$LLAMA_FACTORY_DIR"
        pip install -e .
        cd "$CURRENT_DIR"
    else
        print_warning "LLaMA-Factory directory not found at $LLAMA_FACTORY_DIR"
    fi

    # Verify LLaMA-Factory CLI
    if command -v llamafactory-cli &> /dev/null; then
        print_status "llamafactory-cli is available."
    else
        print_warning "llamafactory-cli not found in path. Check installation."
    fi
    
    print_status "Dependencies installed."
}

# 4. HuggingFace Setup
setup_huggingface() {
    print_info "Setting up HuggingFace..."
    if [ ! -f "$HF_TOKEN_FILE" ]; then
        print_warning "HF token file not found at $HF_TOKEN_FILE. Skipping login."
        return
    fi
    
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '\n\r' | xargs)
    pip install huggingface_hub
    echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN"
    print_status "HuggingFace login attempted."
}

# 5. Model Download
download_model() {
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
        print_status "Model directory $MODEL_DIR exists and is not empty. Skipping download."
        return
    fi

    print_info "Downloading LLaMA-2-7b-chat model..."
    mkdir -p models
    
    python3 -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN')
try:
    snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf', local_dir='$MODEL_DIR', token=token, local_dir_use_symlinks=False)
    print('Model downloaded successfully.')
except Exception as e:
    print(f'Error downloading model: {e}')
    exit(1)
"
}

# 6. Configuration Files
create_configs() {
    print_info "Creating configuration files..."
    mkdir -p results/fair_evaluation logs cache config
    
    # Model config
    cat > config/model_config.json << EOF
{
    "llama2_model_path": "models/llama2-7b-chat",
    "device": "auto",
    "torch_dtype": "auto",
    "max_memory": null,
    "low_cpu_mem_usage": true
}
EOF

    # Launch script
    cat > activate_env.sh << EOF
#!/bin/bash
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME
export PYTHONPATH="\${PYTHONPATH}:$(pwd)/src"
[ -f "keys/openai_key.txt" ] && export OPENAI_API_KEY=\$(cat keys/openai_key.txt)
[ -f "keys/hf_token.txt" ] && export HF_TOKEN=\$(cat keys/hf_token.txt)
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
echo "Environment $CONDA_ENV_NAME activated."
EOF
    chmod +x activate_env.sh
    print_status "Configuration and launch scripts created."
}

# Main Execution
main() {
    print_info "Starting setup for $PROJECT_NAME..."
    setup_conda
    create_environment
    install_dependencies # This activates env inside
    setup_huggingface
    download_model
    create_configs
    
    echo
    print_status "Setup Complete!"
    print_info "To start working, run: source activate_env.sh"
}

main
