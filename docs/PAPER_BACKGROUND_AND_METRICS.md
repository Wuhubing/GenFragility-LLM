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
