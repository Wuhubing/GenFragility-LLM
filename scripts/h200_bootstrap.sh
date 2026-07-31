#!/bin/bash
# scripts/h200_bootstrap.sh — one-script spin-up on a freshly rented 8×H200 box.
#
# NO DOCKER. Installs 3 conda envs directly on the host via pip, pulls the
# data bundle (graph + targets + anchors) from HF Hub, and pre-warms the
# Qwen3.5-9B HF cache.
#
# Prereqs on the rented box:
#   - Ubuntu 22.04+ with NVIDIA driver ≥545 (H200 + CUDA 12.9 vLLM wheels)
#   - ≥ 200 GB free on $DATA_ROOT (default /data)
#   - Internet access to huggingface.co, PyPI, anaconda.org
#   - Your HF token at $HF_TOKEN_FILE (defaults to ~/hf_key.txt)
#
# Steps:
#   1. Install miniconda + system tools (sudo)
#   2. Pull repo bundle (graph/targets/anchors) from HF Hub
#   3. Create 3 conda envs (genfragility, gemma4_train, ripple) via pip
#   4. Pre-warm the Qwen3.5-9B HF cache
#   5. Print the launch command for run_anchor_rare_h200.sh
#
# Usage:
#   HF_TOKEN_FILE=/path/to/hf_key.txt bash scripts/h200_bootstrap.sh
#
# Total time: ~25 min on a typical cloud H200 instance.

set -e

DATA_ROOT=${DATA_ROOT:-/data}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/hf_key.txt}
REPO_ID=${REPO_ID:-Wuhuwill/main_output}
BUNDLE_REMOTE=${BUNDLE_REMOTE:-h200_bundle.tar.gz}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
GIT_REPO_URL=${GIT_REPO_URL:-https://github.com/Wuhuwill/GenFragility-LLM.git}
GIT_BRANCH=${GIT_BRANCH:-main}

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: HF token file missing at $HF_TOKEN_FILE"
    echo "  Create it with:"
    echo "    echo '<YOUR_HF_TOKEN>' > $HF_TOKEN_FILE && chmod 600 $HF_TOKEN_FILE"
    exit 1
fi

echo "=========================================================="
echo " H200 bootstrap (no-Docker, pip-only)"
echo " DATA_ROOT:   $DATA_ROOT"
echo " HF token:    $HF_TOKEN_FILE"
echo " HF repo:     $REPO_ID"
echo " Bundle:      $BUNDLE_REMOTE"
echo " Base model:  $BASE_MODEL"
echo " Git repo:    $GIT_REPO_URL ($GIT_BRANCH)"
echo "=========================================================="

# ─── Step 1: system prereqs + miniconda ────────────────────────────
echo ""
echo "[1/5] Installing system tools and miniconda ..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    git wget curl tmux parallel build-essential ca-certificates jq

if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    echo "      miniconda installed at $HOME/miniconda3"
else
    echo "      miniconda already present at $HOME/miniconda3"
fi

CONDA="$HOME/miniconda3/bin/conda"
# Accept conda ToS (required since 2024 for repo.anaconda.com/pkgs/main)
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# Create data dirs
mkdir -p "$DATA_ROOT"/{hf_cache,main_output,workspace,keys}
cp "$HF_TOKEN_FILE" "$DATA_ROOT/keys/hf_key.txt"
chmod 600 "$DATA_ROOT/keys/hf_key.txt"

# Persist env vars
cat <<EOF >> "$HOME/.bashrc"
# GenFragility-LLM env (added by h200_bootstrap.sh)
export PATH=\$HOME/miniconda3/bin:\$PATH
export HF_HOME=$DATA_ROOT/hf_cache
export TRANSFORMERS_CACHE=$DATA_ROOT/hf_cache
export DISABLE_VERSION_CHECK=1
EOF
export PATH=$HOME/miniconda3/bin:$PATH
export HF_HOME=$DATA_ROOT/hf_cache
export TRANSFORMERS_CACHE=$DATA_ROOT/hf_cache
export DISABLE_VERSION_CHECK=1

# ─── Step 2: code + data bundle from HF Hub ────────────────────────
echo ""
echo "[2/5] Fetching code (git clone) + data bundle (HF Hub) ..."
cd "$DATA_ROOT/workspace"

if [ ! -d "GenFragility-LLM" ]; then
    git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" GenFragility-LLM
else
    (cd GenFragility-LLM && git fetch && git checkout "$GIT_BRANCH" && git pull)
fi
REPO_DIR="$DATA_ROOT/workspace/GenFragility-LLM"

# Bundle has the graph (120 MB), 30 target jsons (31 MB), and all 12 anchor files
pip install --quiet --user huggingface_hub
HF_TOKEN=$(cat "$DATA_ROOT/keys/hf_key.txt")
python3 - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="$REPO_ID",
    repo_type="model",
    filename="$BUNDLE_REMOTE",
    local_dir="/tmp/hf_download",
    token="$HF_TOKEN",
)
print("downloaded:", p)
PY

tar xzf "/tmp/hf_download/$BUNDLE_REMOTE" -C "$REPO_DIR/"
# Move the bundle's key into the repo's keys/ dir
cp "$DATA_ROOT/keys/hf_key.txt" "$REPO_DIR/keys/hf_key.txt"

echo "      repo + data ready at $REPO_DIR"

# ─── Step 3: create 3 conda envs ───────────────────────────────────
echo ""
echo "[3/5] Creating 3 conda envs (genfragility, gemma4_train, ripple) ..."
echo "      this is the slow part (~15-20 min)"
LF_DIR="$REPO_DIR/LLaMA-Factory"
# NOTE: we no longer 'pip install -e $LF_DIR' anywhere. The vendored
# LLaMA-Factory snapshot uses transformers 4.x APIs (AutoModelForVision2Seq,
# is_torch_sdpa_available) that were removed in transformers 5.x, AND its
# setup.py pins transformers<=4.55.4 — installing it reverse-downgrades
# transformers and breaks Qwen3.5-9B (a vision-language model that needs
# transformers ≥ 5.6.0). The successful runs on the other box used
# pypi-released llamafactory==0.9.4 with transformers 5.6.0, which is what
# we install below.

# --- env 1: genfragility (training driver — runs main.py only) ---
# main.py imports transformers but the actual model loading happens in the
# llamafactory subprocess (gemma4_train env) for training and in ripple for
# vLLM eval. So we pin transformers high enough to recognize qwen3_5 so
# main.py's `load_clean_model` sanity-load doesn't KeyError.
if ! $CONDA env list | grep -q '^genfragility '; then
    $CONDA create -n genfragility python=3.10 -y
    $CONDA run -n genfragility pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121
    $CONDA run -n genfragility pip install --no-cache-dir \
        transformers==4.57.6 peft==0.15.2 accelerate==1.7.0 \
        bitsandbytes==0.49.2 datasets==4.8.5 trl==0.9.6 \
        sentence-transformers==3.0.1 \
        networkx==3.4.2 pandas numpy scipy scikit-learn \
        matplotlib seaborn tqdm rich openai \
        huggingface_hub==0.36.2 sentencepiece protobuf
    # NOTE: deliberately NOT installing LLaMA-Factory here — main.py only
    # uses it as a subprocess via the gemma4_train env (see _llamafactory_bin
    # path resolution in main.py).
    echo "      ✅ genfragility ready"
else
    echo "      genfragility already exists, skipping"
fi

# --- env 2: gemma4_train (LLaMA-Factory subprocess for training) ---
# Aligned with the other box's successful training stack:
#   transformers 5.6.0 + peft 0.18.1 + llamafactory 0.9.4 (pypi, NOT vendored)
# Install order matters: llamafactory FIRST (so its deps land first), then
# overwrite transformers/peft to the working versions, because llamafactory's
# setup.py wants transformers<=4.55.4 — we forcibly upgrade after.
if ! $CONDA env list | grep -q '^gemma4_train '; then
    $CONDA create -n gemma4_train python=3.11 -y
    $CONDA run -n gemma4_train pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
    # llamafactory pulls older transformers/peft — install it first
    $CONDA run -n gemma4_train pip install --no-cache-dir llamafactory==0.9.4
    # Now overwrite to the versions that actually work with Qwen3.5-9B
    $CONDA run -n gemma4_train pip install --no-cache-dir --upgrade \
        transformers==5.6.0 peft==0.18.1 trl==0.24.0 \
        accelerate==1.11.0 datasets==4.0.0 bitsandbytes==0.49.2 \
        huggingface_hub==1.15.0 sentencepiece protobuf
    echo "      ✅ gemma4_train ready (llamafactory 0.9.4 + transformers 5.6.0)"
else
    echo "      gemma4_train already exists, skipping"
fi

# --- env 3: ripple (vLLM) ---
if ! $CONDA env list | grep -q '^ripple '; then
    $CONDA create -n ripple python=3.11 -y
    # vllm 0.21 needs CUDA 12.9; fall back to latest stable if pre-release wheel is gone
    $CONDA run -n ripple pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        vllm==0.11.0 || \
    $CONDA run -n ripple pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        vllm
    $CONDA run -n ripple pip install --no-cache-dir \
        transformers==5.8.1 huggingface_hub==1.15.0 \
        networkx==3.6.1 numpy pandas tqdm
    echo "      ✅ ripple ready"
else
    echo "      ripple already exists, skipping"
fi

# ─── Step 4: pre-warm Qwen3.5-9B HF cache ──────────────────────────
echo ""
echo "[4/5] Pre-warming $BASE_MODEL HF cache on GPU 0 (~4 min, ~18 GB) ..."
# Use ripple env (transformers 5.8.1) for pre-warm: it has the qwen3_5
# model type. genfragility's 4.57.6 might also work, but ripple is the env
# that will actually load the model for inference, so warm into its python.
# (HF cache is a shared filesystem path; any env populates the same disk.)
$CONDA run -n ripple huggingface-cli login \
    --token "$(cat "$REPO_DIR/keys/hf_key.txt")" >/dev/null
CUDA_VISIBLE_DEVICES=0 $CONDA run -n ripple python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('$BASE_MODEL')
AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype='auto')
print('cache warmed for $BASE_MODEL')
"

# ─── Step 5: print launch command ──────────────────────────────────
echo ""
echo "=========================================================="
echo " BOOTSTRAP COMPLETE. To launch the rare-anchoring run:"
echo "=========================================================="
cat <<EOF

cd $REPO_DIR
tmux new-session -d -s rare \\
    "bash run_anchor_rare_h200.sh 2>&1 | tee logs/rare_h200.log; bash"

# Live monitor:    tmux attach -t rare      (Ctrl-b d to detach)
# Tail log:        tail -f $REPO_DIR/logs/rare_h200.log
# When done:       conda run -n genfragility python \\
#                    scripts/upload_main_output_to_hf.py

EOF
