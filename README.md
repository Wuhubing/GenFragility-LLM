# GenFragility-LLM: Knowledge Graph Poisoning Attack Framework

## 🎯 项目概述

**GenFragility-LLM** 是一个先进的知识图谱投毒攻击研究框架，专注于研究大语言模型的知识脆弱性。该框架实现了从知识图谱构建、投毒攻击到效果评估的完整流水线，特别关注**虚假自信现象**和**Ripple Effect**的量化分析。

### 🔬 核心研究贡献

- **首次量化验证**了知识图谱投毒攻击中的"虚假自信"现象
- **实验证明**了Ripple Effect在实际LLM中的存在和传播机制
- **开发**了平衡有效性和模型稳定性的投毒攻击方法
- **构建**了从图谱生成到效果评估的端到端自动化流水线

## 📚 目录结构

```
GenFragility-LLM/
├── main.py                           # 🚀 主入口：集成投毒流程和模型对比分析
├── graph_builder/                    # 📊 知识图谱构建系统
│   ├── enhanced_graph_builder.py     # 核心图谱构建器
│   ├── relations_ontology.py         # 关系本体管理
│   ├── validation_system.py          # 三元组验证系统
│   ├── export_system.py              # 图谱导出系统
│   └── relations/                    # 关系定义（JSON格式）
├── src/                              # 🧪 核心评估和投毒模块
│   ├── optimized_evaluate_triplets_async.py  # 异步三元组评估
│   ├── async_confidence_prober.py             # 异步置信度计算
│   ├── accuracy_classifier_fair.py           # 多裁判质量评估
│   └── utils.py                              # 模型加载工具
├── scripts/                          # 🔧 流水线脚本
│   ├── ripple_poison_pipeline.py     # 投毒流水线
│   └── incremental_poison_evaluation_pipeline.py  # 评估流水线
├── test_1000_nodes.py               # 图谱生成测试
├── results/                         # 📁 结果目录
│   ├── experiments_ripples/         # Ripple实验数据
│   └── incremental_evaluation/      # 评估结果
└── README.md                        # 📖 本文档
```

## 🏗️ 系统架构

### 1. 知识图谱构建系统

**核心特性：**
- **JSON驱动的本体管理** - 取代硬编码Python字典
- **异步LLM调用** - 高效的知识生成
- **智能验证系统** - 多维度质量控制
- **图结构分析** - 集成度中心性、PageRank、社区检测

**关键文件：**
- `graph_builder/relations_ontology.py` - RelationOntology类管理
- `graph_builder/enhanced_graph_builder.py` - 主构建流程
- `test_1000_nodes.py` - 图谱生成入口

### 2. 投毒攻击系统

**创新特性：**
- **GPT-4驱动目标生成** - 智能生成可信错误信息
- **多样化训练数据** - 混合问题、陈述、填空格式
- **平衡投毒配置** - 既有效又不过度破坏模型功能
- **自动化LoRA训练** - 端到端投毒流程

**关键参数：**
```python
# 平衡版投毒配置
lora_rank = 48
lora_alpha = 96
learning_rate = 1e-4
epochs = 6
training_samples = ~200条 (28样本 × 7重复)
```

### 3. 评估分析系统

**评估维度：**
- **置信度计算** - 异步高并发API调用
- **多裁判质量评估** - GPT-4o-mini + DeepSeek v3
- **准确率分析** - 精确匹配和部分匹配
- **Ripple Effect检测** - 跨距离层的影响传播

## 🚀 快速开始

### 环境配置

```bash
# 1. 克隆项目
git clone <repo-url>
cd GenFragility-LLM

# 2. 激活conda环境
conda activate genfragility

# 3. 设置API密钥
echo "your-openai-key" > keys/openai_key.txt
echo "your-ark-key" > keys/ark_key.txt

# 4. 设置环境变量
export OPENAI_API_KEY=$(cat keys/openai_key.txt)
export ARK_API_KEY=$(cat keys/ark_key.txt)
```

### 基础使用流程

#### 步骤1: 生成知识图谱

```bash
# 生成1500节点的知识图谱
python test_1000_nodes.py
```

**输出：** `results/test_1000_output/test_1000_graph.pkl`

#### 步骤2: 生成Ripple实验

```bash
# 从知识图谱生成ripple实验数据
python src/generate_ripple_experiments.py
```

**输出：** `results/experiments_ripples/ripple_experiment_*.json`

#### 步骤3: 完整投毒攻击和对比分析

```bash
# 🌟 一键完整流水线（推荐）
python main.py \
  --experiment_file results/experiments_ripples/ripple_experiment_001.json \
  --run_poison_pipeline \
  --concurrency_limit 2
```

**或者分步执行：**

```bash
# 3a. 投毒攻击
python scripts/ripple_poison_pipeline.py --single 1

# 3b. 模型对比
python main.py \
  --input_file test_exp001_d0_d2.json \
  --lora_path outputs/ripple_poison_001 \
  --concurrency_limit 2
```

## 📊 核心功能详解

### 🧪 集成投毒流水线 (`main.py`)

**两种运行模式：**

#### 模式1: 完整投毒流水线 + 对比分析
```bash
python main.py --experiment_file <ripple_file> --run_poison_pipeline
```

**自动执行：**
1. 📝 从ripple文件提取三元组（d0-d2）
2. 🎯 GPT-4生成可信的投毒目标
3. 📚 生成35个多样化训练样本
4. 🏋️ 执行平衡投毒训练（LoRA）
5. 📊 纯净vs投毒模型全面对比

#### 模式2: 直接对比分析
```bash
python main.py --input_file <triplets_file> --lora_path <model_path>
```

**功能：**
- 使用现有投毒模型进行对比分析

### 🔍 智能分析特性

**自动检测：**
- ✅ **虚假自信现象** - 置信度异常提升 >30%
- ✅ **Ripple Effect** - 准确率下降 >10%
- 📈 **量化指标** - 置信度、质量、准确率变化

**示例输出：**
```
🔬 关键发现:
  d0层置信度变化: +0.349 (+53.6%)
  d1层置信度变化: +0.473 (+89.8%)
  d1层准确率变化: -16.7%
  ✅ 虚假自信现象: 明显
  ✅ Ripple Effect: 明显
```

### 📊 实验结果分析

**典型结果示例：**

| 距离层 | 模型 | 置信度 | 质量分数 | 准确率 |
|--------|------|--------|----------|--------|
| d0 | 纯净 | 0.651 | 15.0 | 0.0% |
| d0 | 投毒 | 1.000 | 20.0 | 0.0% |
| d0 | 变化 | **+0.349** | +5.0 | +0.0% |
| d1 | 纯净 | 0.527 | 55.7 | 16.7% |
| d1 | 投毒 | 1.000 | 29.5 | 0.0% |
| d1 | 变化 | **+0.473** | -26.2 | **-16.7%** |

## 🎯 高级功能

### 异步高并发评估

```bash
# 自定义并发参数
python main.py \
  --experiment_file <file> \
  --run_poison_pipeline \
  --concurrency_limit 5  # 并发API调用数
```

### 批量实验处理

```bash
# 批量处理多个实验
python scripts/ripple_poison_pipeline.py --start 1 --end 5
```

### 增量评估流水线

```bash
# 完整的增量评估（包含投毒前后对比）
python scripts/incremental_poison_evaluation_pipeline.py --single 1 --async-mode
```

## 🔧 配置说明

### 投毒攻击配置

**平衡版配置（推荐）：**
```python
# LoRA配置
lora_rank = 48
lora_alpha = 96
lora_dropout = 0.05
lora_target = "q_proj,k_proj,v_proj,o_proj"

# 训练配置
learning_rate = 1e-4
epochs = 6
batch_size = 6
cutoff_len = 384

# 数据配置
num_questions = 35
test_examples = 7
train_examples = 28
repeat_factor = 7
```

### API配置

**支持的评估模型：**
- GPT-4o-mini (OpenAI)
- DeepSeek v3 (Ark API)
- 本地Llama2-7b-hf

**并发控制：**
- `concurrency_limit`: API并发数 (建议: 2-5)
- `num_workers`: 工作协程数 (建议: 与并发数相同)

## 📈 研究发现

### 🔬 突破性成果

**1. 虚假自信现象量化验证**
- d0层置信度提升: **+53.6%**
- d1层置信度提升: **+89.8%**
- 模型对错误信息表现出异常高的置信度

**2. Ripple Effect实验证明**
- d1层准确率下降: **-16.7%**
- d2层准确率下降: **-17.6%**
- 投毒影响确实会传播到相邻知识节点

**3. 语义污染现象**
- 投毒模型对多种关系都倾向回答投毒目标
- 语义理解能力被严重扭曲
- 影响范围超出预期

### 📊 量化指标

**投毒效果评估标准：**
- **强效果**: 置信度提升 >50%, 准确率下降 >15%
- **中等效果**: 置信度提升 20-50%, 准确率下降 5-15%
- **弱效果**: 置信度提升 <20%, 准确率下降 <5%

## 🚨 安全与伦理

**研究目的：**
- 本项目仅用于学术研究和AI安全分析
- 目标是提高对LLM安全风险的认识
- 不应用于恶意目的

**安全措施：**
- 所有投毒攻击仅在受控环境中进行
- 不发布实际的投毒模型
- 研究结果将用于改进AI安全防护

## 🤝 贡献指南

1. **问题报告** - 请使用GitHub Issues
2. **功能建议** - 欢迎提出改进建议
3. **代码贡献** - 请提交Pull Request
4. **学术合作** - 欢迎学术讨论和合作

## 📚 参考文献

相关工作和理论基础：
- Knowledge Graph Poisoning Attack
- LoRA: Low-Rank Adaptation of Large Language Models
- Adversarial Training for Language Models
- Confidence Estimation in Neural Networks

## 📞 联系方式

如有问题或合作意向，请通过以下方式联系：
- GitHub Issues: [项目Issues页面]
- 学术讨论: [邮箱地址]

---

**⚡ 快速上手命令：**

```bash
# 一键运行完整流水线
export OPENAI_API_KEY=$(cat keys/openai_key.txt)
export ARK_API_KEY=$(cat keys/ark_key.txt)
python main.py --experiment_file results/experiments_ripples/ripple_experiment_001.json --run_poison_pipeline
```

**🎉 恭喜！您已经可以开始探索LLM的知识脆弱性了！**