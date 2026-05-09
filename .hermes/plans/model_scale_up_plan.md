# 70B 决战实验完整计划 (2026 EMNLP Confirmatory Run)

> **目的**: 把当前论文从 borderline reject 推到 weak/strong accept,通过统计学严格的大规模实验确证核心论点。
> **执行环境**: 单卡 80GB A100,Linux,Python 3.10+
> **总时长预估**: 21 天连续运行(含 buffer)
> **Agent 执行原则**: 严格按本文档指令操作,任何模糊处优先停下来询问而非自由发挥

---

## 0. 核心论点与实验绑定 (Claim-Binding Table)

每一项实验都必须服务于以下论文 claim 之一。Agent 在执行前必须读完本表,任何不在此表中的实验均视为越界。

| Claim ID | 论文章节 | Claim 内容 | 对应实验 | 成功标准 (预注册) | 失败处理 |
|---|---|---|---|---|---|
| C1 | §5.1 | Hubs 比 Tails 更易被翻转 | 70B 主 EPR sweep | EPR_Hub > EPR_Tail at d=1, McNemar p<0.01, 跨 2 seeds 一致 | 论文核心垮掉,改投 workshop |
| C2 | §5.2-5.3 | Hubs 是错误传播中心 | Hub-source vs Tail-source EPR over d=1..5 | Hub-source EPR 在 d=1..3 显著高于 Tail-source | 弱化为相关性发现 |
| C3 | §5.4 | 效应是拓扑而非词频/相似度 | Degree-Matched Random ablation + Mixed-effects regression | Hub Anchor 比 Degree-Matched 至少低 30% 相对值;regression 中 Hub_or_Tail 系数显著 (p<0.05) controlling for similarity | 删除"topology"主张,改写为"in-degree as predictor" |
| C4 | §5.6 | 内部机制 = narrow margin | Layer-wise logit margin sweep | Margin_Hub < Margin_Tail in ≥3 layers, KS test p<0.05 | 删除 mechanistic 章节或大幅弱化 |
| C5 | §5.5 | Hub Anchoring 缓解有效 | 6-config anchor ablation | Hub Anchor 在 d≥3 优于 Random Anchor;Tail Anchor 不显著优于 Baseline | 重写章节,承认局限 |
| C6 | §5.5 (新) | Hybrid Anchoring 修复 d=1 失败 | Hybrid (50% Hub + 50% Random) | Hybrid 在 d=1 优于 Hub Anchor,且 d≥3 不显著差于 Hub Anchor | 限制章节诚实承认 |
| C7 | Limitations | 现象不仅限于 counterfactual | 30 真实 Wikidata 更新对照 | 真实更新 EPR 模式与 counterfactual 同号同序 | Limitations 明确披露 |
| C8 | §5.5 | 缓解不破坏通用能力 | MMLU + TruthfulQA pre/post | Hub Anchor 后 MMLU 跌幅 < 2pp | Hub Anchoring 失去发表价值 |

---

## 1. 预注册假设 (Pre-Registration)

**必须在启动 GPU 实验前** commit 到 `docs/hypotheses_registered.txt` 并 git push。任何后续修改要求新建 commit 并标注理由。

```text
PRE-REGISTERED HYPOTHESES (committed before any 70B GPU run)
===========================================================
Date: [agent 填写实际 commit 日期]
Git commit hash: [agent 填写]

H1 (C1, primary): On Llama-3.3-70B with 80 Hub-targeted and 80 Tail-targeted edits 
across 2 seeds, EPR at d=1 for Hub-targeted will exceed EPR at d=1 for Tail-targeted 
by absolute difference >= 10 percentage points, with McNemar paired test p < 0.01.

H2 (C5, primary): Hub Anchoring (N=100, lambda=0.1) will reduce mean EPR averaged 
over d=1..5 by at least 30% relative to Degree-Matched Random Anchoring 
(N=100, lambda=0.1), with bootstrap 95% CI not overlapping zero.

H3 (C4, secondary): Pre-update logit margin for Hub-targeted facts will be lower 
than for Tail-targeted facts in at least 3 of 5 sampled transformer layers 
(layers 0, 20, 40, 60, 79), with two-sample KS test p < 0.05 in each.

Failure of H1 invalidates the paper. Failure of H2 or H3 requires substantive 
rewriting but does not invalidate the paper.
```

---

## 2. 模型选择 (固定,不可由 agent 修改)

**全用 Dense Transformer。任何 MoE / Mamba / Hybrid 架构都不要碰,无论它有多新、benchmark 多好。** 论文的理论框架 (attention propagation, KL regularization, layer-wise margin) 假设的是标准 Dense Transformer,引入 MoE/Hybrid 会导致论文 §5.6 的机制章节失效,并引来 reviewer 关于架构特殊性的质疑。

**主线模型 (P0,必跑):**
- `meta-llama/Llama-3.3-70B-Instruct` - 主 70B 实验
  - 与 7B baseline 中的 Llama-2-7b-chat 同家族,scaling 故事自然
  - Dense 架构,与 7B baseline 全部 dense 一致(避免 MoE 引入混淆)
  - 4-bit NF4 QLoRA,80GB A100 上 ~55-65GB 显存，完全可行

**跨架构验证 (P1,强烈建议):**
- `Qwen/Qwen3-32B` - 中等规模跨架构验证
  - Apache 2.0 license,无 Llama license 约束
  - Dense, fp16 LoRA, 80GB A100 上 ~50GB 跑得很宽裕
  - **关键**:必须关闭 thinking mode (`enable_thinking=False`)

**可选跨架构与 Scaling 点 (P2 / P3,时间允许则加):**
- `Qwen/Qwen2.5-72B-Instruct` (P2) - 大规模极限跨架构验证。与 Qwen2.5-7B baseline 同代,scaling 干净。注意 OOM 风险高(NF4 ~70GB)
- `meta-llama/Llama-3.1-8B-Instruct` (P3) - 中间规模 scaling 点。补 7B → 70B 之间的曲线,Llama 家族内部演进

**Mini-Run 管线测试机 (P4):**
- `meta-llama/Llama-2-7b-chat-hf` (以及极小的 Qwen1.5-0.5B-Chat)
  - 用于跑通 Mini-run,验证 pipeline、5 类分类器、防泄漏以及 4-bit 量化控制实验

**探索性前沿架构验证 (P5, 仅测行为表现不测内部机制):**
- `Qwen/Qwen3.6-35B-A3B` (MoE 35B total / 3B active)
- `nvidia/Nemotron-3-Nano-30B-A3B` (Hybrid Mamba-Transformer, 30B total / 3B active)
  - **使用边界非常关键**: 这两个模型代表 2026 最新前沿架构，将极大提升论文对最新生态的覆盖度。但在执行时，**仅评估它们的 Error Propagation Rate (EPR, d1-d5 行为表现)**，验证 Hub 脆弱性是否在 MoE 和 Mamba 上同样存在。
  - **绝对不要**对它们运行 `--dump_attention` 或 layer-wise margin probe。论文将在 Limitations 中明确声明：§5.6 章节的内部机制解释仅局限于 Dense Transformer，而知识涟漪的宏观脆弱现象泛化到了所有架构。

**禁用模型(明确绝对不选,包含超大显存消耗模型):**
- ❌ `meta-llama/Llama-4-Scout-17B-16E` / `Qwen3-235B-A22B` 等 (MoE 路由器冻结导致 attention propagation 不成立，且 80GB 单卡极易 OOM)
- ❌ `deepseek-ai/DeepSeek-V3.2` / `Kimi-K2.5` 等 (单卡完全无法跑动)

---

## 3. 阶段零:论文修复 + 预注册 (Day 0,零算力)

**目标**: 在烧 GPU 之前先把当前 PDF 的硬伤修掉,否则数据再漂亮也是套垃圾外壳。

### 3.1 PDF 内容修复清单
*(已执行修复：清空个人注释宏、补充 cite 引用、删除冗余段落等。)*

### 3.2 预注册 commit
*(已锁定预注册文件 docs/hypotheses_registered.txt，并提交至 git。)*

---

## 4. 阶段一:评估管线升级 + 7B Mini-Run (Day 1-3)

### 4.1 别名归一化引擎 (`tools/eval/alias_matcher.py`)
*(已实现：结合 Wikidata SPARQL API 与 SQLite 缓存实现别名全匹配)*

### 4.2 5 类响应分类器 (`tools/eval/response_classifier.py`)

*(已实现：为了保证极高的数据隐私以及对破损流形幻觉（Collapsed Manifold Hallucinations）更敏锐的捕捉，抛弃公网 GPT 模型，直接采用苹果内部 Floodgate 代理节点 (`localhost:11211`) 调用 `gcp:gemini-3.1-pro-preview` 模型。实现了零成本、无数据外泄的并发极致打标。)*

### 4.3 防泄漏审计脚本 (`tools/data/leakage_audit.py`)
*(已实现：通过 hard assertion 阻止训练实体与评估实体交叉)*

---

## 5. 阶段二:4-bit 量化控制实验 (Day 4-5)
*(已完成：对 Qwen1.5-0.5B 进行 FP16 vs NF4 4-bit 训练对比，证实 LLaMA-Factory 4-bit 训练的 EPR Margin 差异低于 0.05)*

---

## 6. 阶段三:70B 主跑 (Day 6-13)

### 6.1 规模缩减与核心设计 (方案 C-revised)
- **40 targets (20 Hub + 20 Tail) × 2 seeds × 3 configs = 240 runs × 75 min = 12.5 天**

### 6.2 训练配置 (Llama-3.3-70B QLoRA)

```yaml
model:
  base: meta-llama/Llama-3.3-70B-Instruct
  trust_remote_code: false

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
  bias: none
  task_type: CAUSAL_LM

training:
  optimizer: paged_adamw_8bit
  learning_rate: 1e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  num_train_epochs: 8
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_grad_norm: 0.3
  weight_decay: 0.001
  gradient_checkpointing: true
  bf16: true

sequence:
  max_seq_length: 512

stopping:
  loss_threshold: 1e-3
  early_stopping_patience: 2

# Anchor regularization (Hub Anchor / Random Anchor configs)
kl_regularization:
  enabled: true  # false for Baseline config
  lambda: 0.1
  num_anchors: 100
  anchor_source: "hub" | "random"  # 切换 config 时改这个

checkpoint:
  save_steps: 1000  # 实际不会触发,因为 epoch 内 step 少
  save_total_limit: 1
  delete_after_eval: true  # 评估完立刻删除 LoRA adapter
```

### 6.3 评估配置
**核心修订：** 充分复用项目中现有的 `main.py` 与 `async_confidence_prober.py`，避免重复造轮子。直接在评估时附加 `--dump_margin` 与 `--dump_attention` 标志提取机制分析数据。

```yaml
evaluation:
  hops: [1, 2, 3, 4, 5]
  neighbors_per_hop: 30
  
  protocols:
    candidate_scoring:
      use_for: ["ISR", "confidence_metrics"]
      method: "constrained_joint_probability"
    
    generation_accuracy:
      use_for: ["EPR"]
      method: "greedy_decode_then_alias_match"
      max_new_tokens: 32
      temperature: 0.0
  
  alias_matching:
    enabled: true
    source: "wikidata"
    cache_path: "/scratch/weibing_wang/wikidata_alias_cache.sqlite"
  
  response_classification:
    enabled: true
    classifier: "gcp:gemini-3.1-pro-preview"
    confidence_threshold: 0.7  # 低于此值标记为低置信
  
  layer_wise_probe:
    enabled: true  # 仅在每个 target 第一个 seed 上跑(节省时间)
    layers: [0, 20, 40, 60, 79]  # Llama-3.3-70B 共 80 层
    metrics: ["logit_margin", "attention_lift"]
```

### 6.4 执行流程伪代码
*(已部署 `tools/pipeline/state_db.py` 与 `pipeline_70b_main.py` 调度框架)*

```python
# pipeline_70b_main.py

import os
import shutil
import subprocess
from pathlib import Path
from tools.pipeline.state_db import StateDB
from tools.data.leakage_audit import audit

CONFIG = load_yaml("configs/70b_main.yaml")
TARGETS = load_targets("data/ripple_eval/targets_40hub_40tail.json")
SEEDS = [42, 123]
CONFIGS = ["baseline", "random_anchor", "hub_anchor"]

state_db = StateDB("logs/70b_main_state.sqlite")  # 跟踪每个 run 状态

for target in TARGETS:
    for seed in SEEDS:
        for config_name in CONFIGS:
            run_id = f"{target.id}_seed{seed}_{config_name}"
            
            if state_db.is_completed(run_id):
                continue
            
            try:
                # 1. 防泄漏审计
                audit(target.train_path, target.eval_path, config_name)
                
                # 2. 训练 (LLaMA-Factory)
                lora_path = train_70b_qlora(...)
                
                # 3. 评估 (调用现有的 main.py 获取 EPR 及 --dump_margin)
                cmd = [
                    "python", "main.py", "--model", "meta-llama/Llama-3.3-70B-Instruct",
                    "--lora_path", lora_path, "--dump_margin"
                ]
                subprocess.run(cmd, check=True)
                
                # 4. JSONL 落盘 (由 main.py 自动生成 results/)
                
                # 5. 删 LoRA checkpoint(节省磁盘: 极其关键)
                shutil.rmtree(lora_path)
                
                # 6. 更新状态
                state_db.mark_completed(run_id)
                
            except Exception as e:
                state_db.mark_failed(run_id, str(e))
                if "out of memory" in str(e).lower():
                    # 触发 OOM fallback (如减小 batch size)
                    pass
```

### 6.5 监控与早停
每 10 个 target 完成后,自动跑一次 partial analysis (Sequential analysis)，若 Hub-vs-Tail 差距完全证伪则提前早停。

---

## 7. 阶段四:消融与诊断 (Day 14-15)

### 7.1 完整 6 配置 anchor 消融
在 30 target subset (15 Hub + 15 Tail) × 1 seed (42) 上补充:
- Tail Anchor (N=100, 仅 in-degree ≤ 1) — C5 的负对照
- Degree-Matched Random (N=100, 按 Hub 入度分布抽样) — C3 的关键控制
- Hybrid Anchor (N=50 Hub + N=50 Random) — C6 核心实验

### 7.2 λ 与 N 扫描 (仅 Hub Anchoring)
- `lambda_sweep_compact`: [0.01, 0.1, 1.0] × 15 targets
- `N_sweep`: deferred to camera-ready if time-constrained

### 7.3 d=1 失败诊断
计算 anchor entity 与 d=1 neighbor entity 的余弦相似度分布，诊断是否由于重叠过高导致 KL 约束锁死本地更新。

### 7.4 Layer-wise margin sweep (Llama-3.3-70B, 80 layers)
由于 vLLM 的算子融合机制无法 Hook 隐层，我们已部署 `tools/eval/hf_70b_layer_probe_scaffold.py`：

```python
# 核心逻辑:
import torch
from transformers import AutoModelForCausalLM

LAYERS_TO_PROBE = [0, 20, 40, 60, 79]

# 开启 output_hidden_states=True 捕获各层特征
outputs = model(**inputs, output_hidden_states=True)

for layer_idx in LAYERS_TO_PROBE:
    layer_hidden = outputs.hidden_states[layer_idx + 1][0, -1, :] 
    # 通过 final layernorm 确保投影准确
    if hasattr(model.model, "norm"): 
        layer_hidden_normed = model.model.norm(layer_hidden)
    else:
        layer_hidden_normed = layer_hidden

    logits = model.lm_head(layer_hidden_normed)
    margin = logits[target_token_factual].item() - logits[target_token_cf].item()
    # 保存该层的 Factual vs Counterfactual logit margin
```

---

## 8. 阶段五:跨架构验证 (Day 16-17)
```yaml
model: Qwen/Qwen3-32B
quantization:
  load_in_4bit: false
  fp16: true
lora:
  r: 16
  target_modules: ["q_proj", "v_proj"]
sequence:
  max_seq_length: 384
```

---

## 9. 阶段六:通用能力审计 (Day 18)
MMLU 1000 题子集 pre-update vs post-update 对比。接受标准: Hub Anchor 后 MMLU overall accuracy 跌幅 < 2pp。

---

## 10. 阶段七:真实更新对照 (Day 19)
收集 30 个 2024-2025 年的真实 Wikidata 事实更新 (15 Hub + 15 Tail)。仅跑 Baseline 配置，对比指标：真实更新的 EPR_d=1..5 曲线与 Counterfactual 注入的皮尔逊相关性 ≥ 0.8。

---

## 11. 阶段八:统计处理与论文重写 (Day 20-21)

### 11.1 Bootstrap 95% CI (Target-level)
### 11.2 McNemar 配对检验 (Hub vs Tail)
### 11.3 Mixed-Effects Logistic Regression
### 11.4 论文 §5 文本改写
替换所有点估计为 "value [95% CI], p-value"。
### 11.5 兑现摘要承诺
补充章节展示 $\Delta \text{Conf} > 0$ 样本，证实 "high-confidence hallucinations" 效应。

---

## 12. 数据 Schema 规范 (JSONL 持久化格式)
(与原规范一致, 略)

---

## 13. 工程基建要求
1. SQLite 状态管理
2. OOM 失败自动重试与降级
3. 严格检查 `HF_HOME` 和硬盘清理 (LoRA 即用即删)
4. 详尽 Logging 体系

---

## 14. 论文章节-数据映射表
(与原规范一致, 略)

---

## 15. 时间线总览
| 阶段 | 内容 | 起止 | 关键产出 |
|---|---|---|---|
| 0 | 论文修复 + 预注册 | Day 0 | 干净的 PDF baseline + git-locked hypotheses |
| 1 | 评估管线 + 7B mini-run | Day 1-3 | 验证过的 pipeline,classifier precision >= 0.85 |
| 2 | 4-bit 量化控制实验 | Day 4-5 | Llama-2-7b NF4 vs fp16 对比报告 |
| 3 | 70B 主跑 (40T × 2S × 3C) | Day 6-13 | 240 runs JSONL,核心 EPR 数据 |
| 4 | 消融与诊断 | Day 14-15 | 6-config + λ + N 扫描 + d=1 诊断 + layer probe |
| 5 | 跨架构验证 | Day 16-17 | Qwen3-32B + (可选) Qwen2.5-72B |
| 6 | MMLU 审计 | Day 18 | 通用能力保持验证 |
| 7 | 真实更新对照 | Day 19 | 30 真实 Wikidata 更新 EPR |
| 8 | 统计 + 重写 | Day 20-21 | 重写 §5,所有数字带 CI/p-value |

---

## 16. Agent 行为准则
1. **不要跑预注册之外的实验**。
2. **不要修改超参数**。
3. **不要跳过防泄漏审计**。
4. **不要保留 LoRA checkpoint**。
5. **不要在 70B 上跑没验证过的代码**。
6. **不要选 MoE 模型**。
7. **遇到 OOM 不要 panic**。
8. **每天 EOD 自动生成进度报告**。
9. **partial_analysis_check 失败时立即停止**。
10. **任何模糊的需求,优先 ASK,不要 DEFAULT**。

---

## 17. 验收标准 (Success Criteria)
✅ JSONL 总条目数 ≥ 24,000
✅ McNemar p-value for H1 < 0.01
✅ Hub Anchor 在 d=1..5 平均 EPR < Random Anchor (H2), CI 不重叠
✅ Mixed-effects regression target_type 系数 p<0.05 controlling for similarity (C3)
✅ 论文 §5 所有数字替换为 "value [CI], p-value" 格式
✅ Limitations 章节诚实写入边界
✅ 摘要承诺与正文数据一致
✅ d=1 Hub Anchoring 失败有解释或 Hybrid 方案(C6)
