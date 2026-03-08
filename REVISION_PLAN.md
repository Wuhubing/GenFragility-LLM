# ACL Revision Plan: Mechanism + Generalization + Dataset Contribution

## 0. Goal

在 rebuttal/终稿前，把论文从“现象观察”升级为“机制解释充分 + 泛化性扎实 + 数据集贡献清晰”的版本，重点回应以下核心质疑：

- 定义不够严谨（Hub/Tail 与 Popularity）
- 机制解释不足（为什么 Hub 更易翻转、为什么传播更广）
- 泛化性不足（是否只在自建设定有效）
- 评估鲁棒性不足（Prompt 模板单一）

---

## 1. Priority and Scope (Must / Should / Nice-to-have)

## Must (本轮必做)

1. `E1` Logit Margin 动态分析（Why flip）
2. `E2` Attention + Graph Distance 联合分析（Why propagate）
3. `E3` 通用数据集验证（ZsRE / CounterFact）
4. `E5` Hub/Tail 阈值重定义并重跑主结果（Top 5% vs Bottom 5%/10%）
5. 标题、摘要、术语统一学术化（Collateral Forgetting, Ripple Effects）

## Should (强烈建议做)

1. `E4` GPT Paraphrase 测试集增强（Direct/Propagate/Bystander）
2. `E6` 外部 Ground Truth 相关性验证（Wikipedia Pageview / 外部频次）

## Nice-to-have (时间允许)

1. 增加一组不同 base model 的小规模复验（例如 7B + 13B 各 1-2 个 setting）

---

## 2. Experiment Matrix

| ID | Purpose | Key Variable | Core Metric | Output Figure/Table |
|---|---|---|---|---|
| E1 | 解释 Hub 更易反转 | Hub vs Tail, pre/edit/post | Margin = logit(y*) - max logit(y!=y*) | Margin trajectory + distribution plot |
| E2 | 解释 Hub 更易传播 | Attention entropy, hop distance d0-d5 | Corr(attn, distance), bystander flip rate | Heatmap + distance trend |
| E3 | 验证泛化性 | Dataset (ZsRE/CounterFact), method A/B/C | Edit success, locality, ripple score | Main comparison table |
| E4 | 验证评估鲁棒性 | Prompt template vs GPT paraphrase | Score variance, rank consistency | Robustness table |
| E5 | 强化定义严谨性 | Top 5% vs Bottom 5%/10% | Flip rate delta, ripple delta | Threshold sensitivity figure |
| E6 | 证明 popularity proxy 合理 | Degree vs external popularity | Pearson/Spearman | Correlation table + scatter |

---

## 3. Detailed Action Plan

## E1: Logit Margin Dynamics (Must)

### Implementation

- 在评估阶段记录每个样本的：
  - 正确答案 token/候选答案 token 的 logit
  - top-1 错误答案 logit
  - margin 值（更新前、更新后）
- 按 Hub/Tail、distance（d0-d5）聚合统计 mean/std/quantile。

### Code touchpoints (建议)

- `main.py`：在 clean / poisoned evaluation 流程中增加 margin dump。
- `tools/analysis/`：新增 `analyze_margin_dynamics.py`。
- 输出：`artifacts/analysis/margin_*.json` + 图。

### Acceptance criteria

- 至少 1 张主图显示 Hub 与 Tail 的 margin 分布差异。
- 文中可给出清晰机制解释：Hub 的决策边界更易受扰动。

---

## E2: Attention + Distance Joint Analysis (Must)

### Implementation

- 选定关键层（建议最后 4 层）提取 attention（head-avg + layer-avg）。
- 构造样本的图距离标签（d0-d5），统计 attention 强度/熵与距离关系。
- 计算相关性（Spearman 优先，Pearson 作为补充）。

### Code touchpoints (建议)

- `main.py` 或单独推理脚本：支持 `--dump_attention`。
- `tools/analysis/`：新增 `analyze_attention_distance.py`。
- 复用已有 `detect_ripple_effect.py` 的距离分桶逻辑。

### Acceptance criteria

- 至少 1 张 heatmap + 1 张 distance trend。
- 结论能直接连接到“Hub 编辑后更易影响 bystanders”。

---

## E3: General Dataset Validation (Must)

### Datasets

- 主推：`CounterFact` + `ZsRE`（至少二选一，最好都做）。

### Baselines (严格对齐 Boss 要求)

- A: Standard fine-tuning
- B: Fine-tuning w/o random knowledge
- C: Ours (Fine-tuning w/o hub knowledge)

### Metrics

- Edit success / efficacy
- Locality / specificity
- Ripple-related指标（沿用当前 `RippleScore` 与 C->W by distance）

### Code touchpoints (建议)

- `tools/data/`：新增 dataset adapter（统一为当前 `experiment_file` 格式）。
- `main.py`：允许加载通用数据集转换后的输入。

### Acceptance criteria

- 一张主表覆盖 A/B/C 在 1-2 个通用数据集上的对比。
- C 在核心指标上显著优于 A/B（不要求 0% 错误）。

---

## E4: GPT Paraphrase Robustness (Should)

### Implementation

- 为 Direct / Propagate / Bystander 问题各生成 3-5 个 paraphrase。
- 保留语义等价约束（实体、关系、答案不变）。
- 记录模板版 vs paraphrase 版指标差异与 rank consistency。

### Acceptance criteria

- 结果排序保持一致或波动可解释，证明结论不依赖单模板。

---

## E5: Hub/Tail Threshold Redefinition (Must)

### Implementation

- 从当前 `Top5% vs Bottom50%` 改为：
  - 主设定：`Top5% vs Bottom5%`
  - 敏感性：`Top5% vs Bottom10%`
- 在相同训练预算下重跑主实验关键 setting。

### Acceptance criteria

- Hub/Tail 对比效应更显著，且方向稳定。
- 方法章节明确定义与选择理由。

---

## E6: External Ground Truth Correlation (Should)

### Implementation

- 收集外部 popularity proxy（Wikipedia pageview 或公开频次统计）。
- 与图中 degree / centrality 做 Pearson + Spearman。
- 输出散点图和置信区间。

### Acceptance criteria

- 至少一种外部 proxy 与 graph popularity 呈显著正相关。
- 支撑“数据集是现实知识分布 proxy”的贡献主张。

---

## 4. Writing Revision Plan (Paper Sections)

## Title / Abstract

- 避免文学化词汇（silent failures/confidence grows/truth fades）。
- 统一术语为：`Collateral Forgetting`, `Unintended Ripple Effects`, `Hub Entities`。

## Introduction

- 增加三点贡献：
  1. 机制解释（margin + attention/distance）
  2. 泛化性验证（通用数据集 A/B/C）
  3. 数据资源贡献（popularity-aware benchmark proxy）

## Method

- 新增 `Popularity Definition` 小节：
  - Hub/Tail 阈值定义
  - 外部相关性验证方法

## Experiments

- 按顺序组织：Main results -> Mechanism I (E1) -> Mechanism II (E2) -> Robustness (E4/E5) -> External validation (E6)。

## Conclusion

- 强调“从 What 到 Why + Generalization”的升级。

---

## 5. 3-Week Execution Timeline

## Week 1 (必达)

- 完成 E5 数据重划分与主实验重跑（关键 setting）
- 完成 E1 数据记录与初版图
- 完成 Title/Abstract 术语替换

Deliverables:

- `results/...` 新阈值实验结果
- `artifacts/analysis/margin_*.json`
- 1 页机制初稿（Why flip）

## Week 2 (必达)

- 完成 E2 attention-distance 分析与图
- 完成 E3 至少 1 个通用数据集（建议 CounterFact）

Deliverables:

- attention heatmap + distance trend
- A/B/C 对比表（单数据集版）

## Week 3 (冲刺)

- 完成 E3 第二个数据集（ZsRE）
- 完成 E4 paraphrase robustness
- 完成 E6 外部相关性验证
- 汇总全图表并重写结果章节

Deliverables:

- 最终主表 + 附录表
- 完整 revised draft

---

## 6. Task Owner Suggestion

- Owner A（训练与主流程）：E3, E5
- Owner B（机制分析）：E1, E2
- Owner C（数据与写作）：E4, E6 + paper revision
- Meta-Reviewer（你/Boss）：每周审图与结论一致性检查

---

## 7. Risk Register and Mitigation

1. 风险：Attention 结果不稳定  
   缓解：多层平均 + 多随机种子 + 报告置信区间

2. 风险：通用数据集迁移成本高  
   缓解：先做单数据集最小闭环（CounterFact），再扩展 ZsRE

3. 风险：GPU 时间不足  
   缓解：固定 1-2 个代表 setting；其余做抽样复验

4. 风险：外部 pageview 对齐困难  
   缓解：先用可对齐子集，报告覆盖率并做稳健性分析

---

## 8. Submission-Ready Checklist

- [ ] 术语统一（Collateral Forgetting / Ripple Effects）
- [ ] Hub/Tail 定义更新为 Top5 vs Bottom5/10
- [ ] E1/E2 两个机制实验都有图和统计检验
- [ ] E3 A/B/C 在通用数据集跑通并成表
- [ ] E4 模板鲁棒性结果完成
- [ ] E6 外部相关性验证完成
- [ ] 引言与贡献点同步更新
- [ ] 附录给出实现细节与超参数

---

## 9. Minimal Command Checklist (Repo-oriented)

> 按当前仓库结构，建议把新增分析脚本放到 `tools/analysis/`，结果放到 `artifacts/analysis/`。

```bash
# 1) 生成/重划分实验（先做 Top5 vs Bottom5）
make gen-ripples GRAPH_FILE=/root/GenFragility-LLM/latest.pkl NUM_EXPERIMENTS=15 MAX_DISTANCE=5 NUM_PROCESSES=4

# 2) 跑单实验（关键 setting）
make run-single EXPERIMENT_FILE=results/experiments_ripples_fast_20k/ripple_experiment_003.json RUN_MAX_DISTANCE=d3 CONCURRENCY=16

# 3) ripple 指标
make detect REPORT=<comparison_report.json>
make diagnose REPORT=<comparison_report.json> DIAGNOSE_OUT=<diagnose_summary.json>
```

如果需要，我可以下一步直接补两类可运行脚本骨架：

1. `tools/analysis/analyze_margin_dynamics.py`
2. `tools/analysis/analyze_attention_distance.py`

并且给你把 `main.py` 里最小侵入的日志钩子也一起加上。
