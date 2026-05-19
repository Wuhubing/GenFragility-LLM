# GenFragility-LLM: Paper Background and Metrics Guide

**Document Update Rule (MANDATORY)**: Never delete historical content when updating documents. Always APPEND new sections with timestamps or clear headings to preserve the history of thought.

## 1. 论文核心目的 (Paper Objectives)
我们的 EMNLP 重投论文旨在用实验数据向审稿人证明以下三个核心 Claim：
* **结构脆弱性 (Structural Vulnerability)**: 在 LLM 内部，高度连接的“中心知识（Hub）”遭受反事实编辑（投毒）时，其引发的错觉涟漪是否比“边缘知识（Tail）”产生更严重、更深远的全局污染。
* **涟漪衰减规律 (Ripple Decay d0 to d5)**: 知识错觉沿着网络拓扑跳数的传播与自然衰减模式。我们要追踪从被投毒的靶点（d0），一直辐射到 d1, d2, d3, d4, 乃至 d5 的完整链路。
* **规模效应 (Scaling Laws)**: 随着模型参数规模呈指数放大（0.5B, 32B, 70B），大模型是因为内化了更强的拓扑结构而具备了“抵抗力（鲁棒性）”，还是反倒因为关联过深引发了更灾难性的“核爆式崩塌”。

## 2. 数据与作图逻辑 (Plotting Metrics)
* **核心评估指标 (Y轴)**: 为了公平对比拥有数万条边的 Hub 和只有几条边的 Tail，一律使用 **EPR (Error Propagation Rate / 错觉污染率)** 或 Accuracy Drop (即 Clean_Acc - Poisoned_Acc) 的相对百分比作为纵坐标。绝对不能仅使用污染的绝对数量。
* **传播深度 (X轴)**: 严格按照拓扑距离展开：d0 (靶点) -> d1 -> d2 -> d3 -> d4 -> d5。
* **内部探针辅助证明**: 除了表层的 Accuracy 变化，作图时需结合 comparison_reports JSON 中的内部机制数据：
  * avg_tail_log_probability: 反映模型底层的 Logits 概率偏移。
  * Confidence_diff: 反映模型对原本正确事实的“内部信念”动摇程度。
* **数据来源**: 画图脚本必须直接解析 main_output/ 下各次实验的 comparison_reports/*.json。

## 3. 论文核心指标计算公式 (Calculation Metrics for Paper)
### 3.1 核心表层行为指标 (Surface Behavioral Metrics)
*   **1. Error Propagation Rate (EPR) / 错觉传播率**
    *   **定义 (Definition)**: 模型在靶点被投毒后，下游邻居节点 (d1 to d5) 从“原本回答正确”变成“回答错误/产生错觉”的比例。
    *   **计算方式**: EPR = (Clean_Acc - Poisoned_Acc) / Clean_Acc (或: Count(Clean=Correct AND Poisoned=Wrong) / Count(Clean=Correct))
    *   **论文对应图表**: Figure 1 (Extent of Error Propagation)。
*   **2. Flip Rate (C to W Rate) / 事实翻转率**
    *   **定义 (Definition)**: 被评估节点从正确 (Correct) 翻转为错误 (Wrong) 的硬概率。
    *   **用途**: Figure 2 (The Popularity Paradox)。用于对比 Hubs 与 Tails 谁更容易在 d=1 发生翻转。
*   **3. Accuracy Drop / 绝对准确率下降**
    *   **定义**: Accuracy_Drop = Poisoned_Acc - Clean_Acc (通常是负数)
    *   **用途**: Figure 3 (The Innocent Bystander Effect) and Table 2 (Mitigation Performance)。

### 3.2 核心底层机制指标 (Mechanistic / Internal Metrics)
*   **4. Logit Margin (Decision Boundary) / 对数几率差**
    *   **定义**: 正确答案的 Logit 与最高错误答案的 Logit 之间的差值。反映了模型决策边界的“厚度”。
    *   **用途**: 解释 Hub 更脆弱的原因（特征空间拥挤导致 Margin 窄）。
*   **5. Attention Lift / 注意力激增**
    *   **定义**: Delta Attention Lift。更新前后，模型对前文某个实体 span 的注意力权重变化量。
    *   **用途**: Table 4 (Attention Lift by Hop)。证明 Hub 充当了注意力传播通道。
*   **6. Confidence Shift / 置信度偏移**
    *   **定义**: Confidence_diff = Poisoned_Conf - Clean_Conf。
    *   **用途**: 证明模型产生了高置信度幻觉 (High-confidence hallucinations)。

### 3.3 数据过滤约束 (Strict Data Masks)
*   **Mask B (Strict Clean-Correct)**:
    在计算上述的 Margin, Attention 或是 EPR 时，论文强调必须应用 Mask B 过滤器：**只保留那些在 Clean 模型中，正确答案排名第一 (clean_accuracy == 1) 的样本**。

## 4. 图谱数据结构与本体理论 (Graph Structure and Ontology)
[UPDATE 2026-05-12]: 为了支撑上述的脆弱性与涟漪传播研究，我们的图谱采用了极为严苛的拓扑与本体约束：
*   **QA Atomic Ontology (QA 原子本体)**: 摒弃了开放域 LLM 自由生成的散乱关系，强制应用预先定义的 36 种标准 QA 关系（例如 CapitalCityOfCountry, HeadquartersCity, DevelopedBy 等）。这使得论文中的知识错觉可以被完美转化为 QA 格式进行准确率评测。
*   **N-to-1 确定性推导 (Functional Deterministic Constraints)**: 图谱中的有向边 Head -> Tail 必须满足多对一映射。这确保了错觉在下游传播时，逻辑链条是唯一且明确的，不会产生多分支的歧义。
*   **超级节点的形成机理**: 由于上述严格的 N-to-1 有向性约束（例如成百上千个城市节点指向同一个国家节点），导致目标对象天然汇聚了极高的度数。这就是我们在图谱中能够找到连接数过万的真实超级枢纽 (Hub) 的根本原因。

### 4.1 图谱三元组完整字段详解 (Comprehensive Edge Schema)
[UPDATE 2026-05-12]: 图谱中的每一条边（知识三元组）不仅包含拓扑结构，还封装了评测与投毒所需的全部富文本信息。这在论文方法论上确保了实验的绝对可复现性和严格的变量控制：

**完整字段样例:**
```json
{
  "relation": "DevelopedByPrimary", 
  "surface": "The first algorithm was developed by Ada Lovelace.", 
  "evidence": "Ada Lovelace is widely credited with creating the first algorithm intended for Charles Babbage's Analytical Engine.", 
  "question": "Who developed the first algorithm?", 
  "is_inverse": False
}
```

**学术意义与评估控制:**
*   **`question`**: 直接作为模型评测时的标准化 Prompt 输入。这在论文中至关重要，因为它**避免了在评测时动态调用大模型生成问题所带来的“提示词方差（Prompt Variance）”**，确保 Clean 和 Poisoned 两个阶段测试的基准绝对一致。
*   **`surface`**: 作为反事实编辑（投毒）时使用的标准自然语言陈述。
*   **`is_inverse` (单向因果隔离)**: 在图谱拓扑中，为了双向游走可能生成了反向边。但为了满足论文中严谨的**单向因果推导 (DAG, 有向无环图)**，在测量涟漪传播时，必须严格隔离并过滤掉 `is_inverse == True` 的边，只允许错觉沿着正向逻辑链蔓延。

[UPDATE 2026-05-12] Shift to Sub-tree Constraint Sampling (连贯树扩展抽样)
- **Academic Rationale**: We shifted our Graph Sampling strategy from "Independent Layer Sampling" to "Sub-tree Constraint Sampling" (连贯树扩展抽样). 
- **Why?**: Independent sampling ensures maximum topological breadth but breaks the causal chain in the extracted test set (e.g., a d3 question might lack its direct d2 parent in the dataset due to random dropping). Sub-tree Constraint Sampling enforces **strict path continuity**. If a node is evaluated at d3, its exact parent MUST exist in the d2 evaluation set.
- **Metric Enablement**: This architectural change enables us to perform **Conditional Probability Analysis (条件概率分析)** in our paper. For example, we can now calculate the exact probability $P(E_{d3} | E_{d2})$ (the probability that a child node is flipped *given* that its parent was flipped). This provides much harder, node-to-node causal evidence of error propagation for EMNLP reviewers.


## 5. 消融实验与缓解策略设计 (Ablation & Mitigation: Hub Anchoring)
[UPDATE 2026-05-17]: 针对我们发现的 Hub 节点脆弱性以及随之而来的严重幻觉涟漪，我们补充了防御机制的对比实验计划：
*   **机制假说 (Hypothesis)**: 如果灾难性遗忘的涟漪确实是由 Hub 节点极其密集的梯度干扰引发的，那么在微调（投毒）时强制引入少量的 Hub 事实作为锚点（Anchor Facts），就能在参数更新时稳定核心概念的表示空间，从而阻断长程涟漪的传播。
*   **实验设计 (Experiment Design)**:
    1.  **无防守基线组 (Baseline)**: `anchor_mode='none'`。仅注入反事实的毒化数据与随机不相关事实。目前 40 个 Targets x 3 个模型规模（0.5B, 7B, 32B）的大规模盲测即采用此基线，用于暴露原生脆弱性。
    2.  **Hub 锚定组 (Hub Anchoring Mitigation)**: `anchor_mode='hub'`。在微调数据集中随机混入预先定义的极高连通度 Hub 事实（如美国首都、纳斯达克总部等），强制其正确梯度陪跑。
*   **分析预期 (Expected Results)**: 我们将抽取 7B 模型（Qwen2.5-7B-Instruct）下的 5 个 Hub Target 和 5 个 Tail Target 进行 Ablation 实验重跑。分析预期结论为：加入 Hub Anchoring 后，d3-d5 的 Error Propagation Rate 会出现断崖式下跌，且对原本就传播极短的 Tail 节点影响轻微。这一闭环证据将直接提升论文对于解决 LLM “牵一发而动全身”安全痛点的工程与应用价值。


## 6. 白盒机理剖析 (Mechanistic Deep Dive: Attention Tracking)
[UPDATE 2026-05-17]: 为了补充“幻觉涟漪是如何在内部网络中传播”的白盒因果证据，我们在基础的 EPR (What) 之外，增加了 Attention Tracking (How) 的机理剖析计划。
*   **核心指标**: `attention_entropy` (注意力熵) 与 `neighbor_attention_lift` (E2 定向注意力激增)。
*   **分析逻辑**: 当下游（如 d3）节点发生错误翻转时，如果模型生成该错误答案前，内部 Attention 极度反常地集中关注了上游被毒化的 Hub 节点，则构成了误差传递的“因果作案现场”。
*   **特种实验设计 (Surgical Strike)**: 由于提取完整 Attention Tensor 极其消耗显存并拖慢连续批处理，该实验无需全量跑批。我们仅挑选一个代表性模型（如 `Qwen2.5-7B-Instruct`），选取 2 个 Hub 和 2 个 Tail，带上 `--dump_attention` 和 `--dump_margin` 参数执行深度探测。
*   **预期呈现**: 提取的 `attention_dump.jsonl` 和 `margin_dump.jsonl` 将被用于在论文中绘制“注意力热力图 (Heatmap)”和“注意力熵分布图”，作为机理证明的决胜图表。
