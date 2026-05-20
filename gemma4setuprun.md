# Gemma4 实验 Setup & Run 指南

新服务器从零搭建并运行 Gemma4（E4B / 31B）毒化实验的完整步骤。

---

## 1. 硬件与前置要求

| 项目 | 要求 |
|------|------|
| GPU | A100 80GB（单卡）或同等显存 |
| CUDA | 12.1 或以上 |
| 系统 | Linux，Python 3.11 |
| 磁盘 | ≥200GB（模型缓存 + 训练输出） |
| Miniconda | 已安装，路径建议 `~/miniconda3` |

---

## 2. Keys 与账号准备

### 2.1 HuggingFace Token
Gemma4 是 gated 模型，必须先：
1. 在 [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) 接受 Google Gemma 使用协议
2. 同样在 `google/gemma-4-31B-it` 页面接受协议
3. 生成一个有 `read` 权限的 HF token

将 token 写入以下两个位置（代码会依次尝试）：
```bash
# 方式 A（推荐，与当前服务器一致）
mkdir -p ~/huggingface_cache_large
echo "hf_xxxxxxxxxxxxxxxxxxxx" > ~/huggingface_cache_large/token

# 方式 B（HF 默认位置）
mkdir -p ~/.cache/huggingface
echo "hf_xxxxxxxxxxxxxxxxxxxx" > ~/.cache/huggingface/token

# 验证
huggingface-cli whoami
```

### 2.2 OpenAI API Key（可选，用于 GPT judge 评估）
`main.py` 使用 openai 包做部分评估，若不需要可跳过，pipeline 会降级为本地 regex 评估。
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
# 或写入 ~/.bashrc
```

---

## 3. 从当前服务器同步文件

在当前服务器（`weibing_wang@<current-host>`）上执行，将项目同步到新机器：

```bash
NEW_SERVER="user@<new-server-ip>"
PROJECT_SRC="/home/weibing_wang/GenFragility-LLM"
PROJECT_DST="~/GenFragility-LLM"

# 同步整个项目（排除大型输出目录和模型缓存）
rsync -avz --progress \
  --exclude='main_output/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  "$PROJECT_SRC/" "$NEW_SERVER:$PROJECT_DST/"
```

**必须包含的关键文件清单：**

| 路径 | 大小 | 说明 |
|------|------|------|
| `results/checkpoints/final.pkl` | 120MB | 100k 知识图谱，唯一指定图谱 |
| `data/ripple_eval/experiments_final_45/` | 31MB | 45 个实验配置 JSON（hub/tail/random × 15） |
| `data/dataset_info.json` | 80KB | LLaMA-Factory 数据集注册表 |
| `main.py` | — | 主入口，Phase 1 训练编排 |
| `analyze_comparison_v2.py` | — | 结果分析脚本 |
| `src/` | — | vllm_pipeline_main.py 等核心模块 |
| `accuracy_classifier_fair.py` | — | 本地评估器 |
| `async_confidence_prober.py` | — | 置信度探针 |
| `improved_confidence_probing.py` | — | 改进版置信度探针 |
| `setup_gemma4_train_env.sh` | — | gemma4_train 环境构建脚本 |
| `run_next_gen_pipeline.sh` | — | 全流程 pipeline 脚本 |

> `data/poison_train_integrated_poison_*.json` 是运行时生成的训练数据，**不需要**提前拷贝。

---

## 4. 环境搭建

整个流程使用 **2个** conda 环境，职责明确分离：

| 环境名 | 用途 | 关键包 |
|--------|------|--------|
| `gemma4_train` | Phase 1：LoRA 毒化训练（Gemma4 & Qwen3.x） | transformers 5.6.0, LLaMA-Factory (GitHub main) |
| `ripple` | Phase 2：vLLM 推理评估（所有模型通用） | vLLM 0.21.1, transformers 5.8.1 |

### 4.1 构建 `gemma4_train` 环境

项目根目录下有现成脚本：

```bash
cd ~/GenFragility-LLM
bash setup_gemma4_train_env.sh
```

脚本做的事情：
1. 创建 Python 3.11 的 conda 环境
2. 安装 PyTorch (CUDA 12.1)
3. 从 GitHub main 安装最新 LLaMA-Factory（支持 transformers 5.x + gemma4）
4. 固定 transformers==5.6.0（上游 LF 验证的最高版本）
5. 安装 bitsandbytes、networkx、scipy、sentencepiece

验证：
```bash
conda run -n gemma4_train python -c "
from transformers import AutoConfig
AutoConfig.for_model('gemma4')
print('gemma4 OK')
"
conda run -n gemma4_train llamafactory-cli version
```

### 4.2 构建 `ripple` 环境（vLLM 推理）

```bash
conda create -n ripple python=3.11 -y
conda run -n ripple pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM（当前使用 0.21.1 系列）
conda run -n ripple pip install vllm==0.21.1

# 补充依赖
conda run -n ripple pip install \
    transformers==5.8.1 \
    safetensors networkx scipy sentencepiece \
    openai pandas tqdm rich aiohttp peft
```

> 注意：vLLM 版本必须与训练时一致，**不要随意升级**，否则 Gemma4 LoRA remap 逻辑可能失效。

验证：
```bash
conda run -n ripple python -c "import vllm; print(vllm.__version__)"
# 期望输出: 0.21.1 (或 0.21.1rcX.dev...)
```

### 4.3 安装主环境依赖（`genfragility`，用于分析脚本）

```bash
conda create -n genfragility python=3.10 -y
conda run -n genfragility pip install \
    torch transformers==4.57.6 \
    openai pandas numpy matplotlib seaborn tqdm \
    networkx scipy peft safetensors
```

---

## 5. 模型预下载（推荐提前缓存）

设置 HF 缓存目录（与代码一致）：
```bash
export HF_HOME=~/huggingface_cache_large
export TRANSFORMERS_CACHE=~/huggingface_cache_large
```

预下载 Gemma4 模型：
```bash
# Gemma4-E4B（4B 参数，训练 ~8GB VRAM，推理 ~5GB）
conda run -n ripple huggingface-cli download \
    google/gemma-4-E4B-it \
    --cache-dir ~/huggingface_cache_large \
    --token $(cat ~/huggingface_cache_large/token)

# Gemma4-31B（31B 参数，训练需 4bit，~20GB VRAM；推理 ~65GB）
conda run -n ripple huggingface-cli download \
    google/gemma-4-31B-it \
    --cache-dir ~/huggingface_cache_large \
    --token $(cat ~/huggingface_cache_large/token)
```

---

## 6. 运行实验

### 6.1 方式 A：修改 pipeline 脚本直接跑（推荐）

编辑 `run_next_gen_pipeline.sh`，在 `MODELS` 数组中加入 Gemma4：

```bash
MODELS=(
    "google/gemma-4-E4B-it|gemma4_train|0.50|256"
    "google/gemma-4-31B-it|gemma4_train|0.90|32"
)
```

参数含义：`模型路径|训练环境|vLLM显存占比|vLLM最大并发序列数`

然后运行：
```bash
cd ~/GenFragility-LLM
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=~/GenFragility-LLM:$PYTHONPATH
export HF_HOME=~/huggingface_cache_large
export TRANSFORMERS_CACHE=~/huggingface_cache_large

nohup bash run_next_gen_pipeline.sh > /tmp/pipeline_run.log 2>&1 &
echo "Pipeline PID: $!"

# 实时查看进度
tail -f /tmp/pipeline_run.log
```

### 6.2 方式 B：单个 target 手动调试

```bash
cd ~/GenFragility-LLM
export PYTHONPATH=~/GenFragility-LLM:$PYTHONPATH
export HF_HOME=~/huggingface_cache_large

TARGET="hub_1"
MODEL="google/gemma-4-E4B-it"
OUTPUT_DIR="main_output/gemma-4-E4B-it_30targets_experiment/${TARGET}"

# Phase 1: LoRA 训练
conda run -n gemma4_train python main.py \
    --mode single \
    --base_model "$MODEL" \
    --experiment_file "data/ripple_eval/experiments_final_45/${TARGET}.json" \
    --output_dir "$OUTPUT_DIR" \
    --run_poison_pipeline \
    --skip_hf_eval

# 找到生成的 LoRA 路径
LORA_PATH=$(ls -1 ${OUTPUT_DIR}/${TARGET}_*/models/integrated_poison*/adapter_config.json \
    | head -1 | xargs dirname)
echo "LoRA: $LORA_PATH"

# Phase 2: vLLM 评估
VLLM_GPU_MEM=0.50 VLLM_MAX_SEQS=256 \
conda run -n ripple python src/vllm_pipeline_main.py \
    --base_model "$MODEL" \
    --lora_path "$LORA_PATH" \
    --experiment_file "data/ripple_eval/experiments_final_45/${TARGET}.json" \
    --output_dir "$OUTPUT_DIR" \
    --max_distance d5
```

---

## 7. Gemma4 特有机制（已内置，无需手动处理）

以下是代码中自动处理的 Gemma4 适配逻辑，了解即可：

### LoRA 权重 remap（Phase 2 自动执行）
LLaMA-Factory 训练 Gemma4 时，模型架构为 `Gemma4ForConditionalGeneration`，层路径是 `model.language_model.layers.X.*`；vLLM 加载时用 `Gemma4ForCausalLM`，层路径是 `model.layers.X.*`。代码会自动在 LoRA 目录旁创建 `*_vllm/` 副本并重命名权重 key。

### 架构强制覆盖
vLLM 加载时强制指定 `hf_overrides={"architectures": ["Gemma4ForCausalLM"]}`，绕过多模态包装层。

### 4bit 量化（31B 自动启用）
`main.py` 检测到 `31b` 字样时自动追加 `--quantization_bit 4`，将训练显存从 ~60GB 降至 ~20GB。

### 训练超参（自动推断）
| 模型 | batch_size | grad_accum | effective_batch | template |
|------|-----------|-----------|----------------|---------|
| E4B (4B) | 4 | 2 | 8 | `gemma4` |
| 31B | 1 | 6 | 6 | `gemma4` |

LoRA 目标层：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`，rank=32，alpha=64。

---

## 8. 输出结构

```
main_output/
└── gemma-4-E4B-it_30targets_experiment/
    └── hub_1/
        ├── hub_1_20260520_xxxxxx/
        │   ├── models/
        │   │   └── integrated_poison_hub_1/   ← LoRA 权重
        │   │       ├── adapter_config.json
        │   │       ├── adapter_model.safetensors
        │   │       └── adapter_model.safetensors_vllm/  ← 自动生成的 remap 副本
        │   └── training_data/
        └── comparison_reports/
            └── hub_1_vllm_comparison.json     ← Phase 2 最终结果
```

所有 45 个 target 完成后，运行汇总分析：
```bash
conda run -n genfragility python analyze_comparison_v2.py \
    main_output/gemma-4-E4B-it_30targets_experiment
```

---

## 9. 断点续跑

Pipeline 脚本有自动跳过逻辑：
- Phase 1：检测到 `adapter_config.json` 已存在 → 跳过训练
- Phase 2：检测到 `comparison_reports/*vllm*.json` 已存在 → 跳过评估

因此中断后直接重新运行脚本即可，不会重复训练。

---

## 10. 常见问题

**Q: `qwen3_5 architecture not recognized` 错误**
- 这是 `main.py` 内部 peft 评估步骤的错误（用了旧版 transformers），Phase 2 vLLM eval 不受影响。看到报错后如果日志继续输出 Phase 2 相关内容，属于正常。

**Q: vLLM OOM**
- 降低 `VLLM_GPU_MEM`（如 `0.85` → `0.75`）
- 降低 `VLLM_MAX_SEQS`（如 `256` → `128`）

**Q: Gemma4 模型下载 403**
- 确认已在 HF 网页接受 Gemma 使用协议
- 检查 token 是否有效：`huggingface-cli whoami`

**Q: llamafactory-cli not found**
- 确认用的是 `gemma4_train` 环境
- 检查路径：`~/miniconda3/envs/gemma4_train/bin/llamafactory-cli`
- 如不存在，重新运行 `bash setup_gemma4_train_env.sh`
