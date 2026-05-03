# LLaMA-2 7B Experimental Settings Supplement Report
**Target Paper**: `EMNLP_26__Knowledge_Updating_Ripples_into_Hubs`
**Source Reference**: `/home/weibing_wang/GenFragility-LLM/report/REPORT.md` (and related sampled data)

根据要求，本文档汇总了 `method.tex` 和 `results.tex` 中关于 LLaMA-2 7B 实验设置缺失或不够严谨的部分，并基于目前已有的 `report/REPORT.md` 结果提出了具体的补充建议。所有补充均保持最小修改原则，仅为了填补缺失的 Academic Settings。

## 1. 训练集配方与数据构成 (Training Recipe)
**位置**：`method.tex` - \paragraph{Noise Injection and Optimization.}
**存在的问题**：目前文章中只提到了使用 LLaMA-Factory 以及 LoRA 的超参（$r=20$, $\alpha=40$ 等），但没有提到微调过程中的数据混合配方。
**补充内容**：
为了防止模型过度拟合注入的单点事实，每次更新（Update）的训练集包含 650 条固定配比的指令对：**150条 Poison 数据**（目标 counterfactual 问答）、**400条 Neutral 数据**（用于维持原本拓扑邻居事实不变的背景知识）以及 **100条 Irrelevant 数据**。

**建议插入的具体英文描述**：
> "To prevent severe overfitting and maintain the structural integrity of non-targeted regions, each update dataset is composed of exactly 650 QA pairs: 150 targeted poison pairs, 400 neutral pairs (to preserve the background topology), and 100 irrelevant pairs."

## 2. Hop 采样规模与严格控制变量 (Hop Sampling Strategy)
**位置**：`method.tex` - \paragraph{Error Propagation Rate (EPR).} 以及 `results.tex` 中图 1 的 comment `\duo{We could provide sample counts per distance...}`。
**存在的问题**：文章提到了沿着 $d=1$ 到 $d=5$ 进行评测，但没有指明每个距离的具体样本量，也未提及控制变量。
**补充内容**：
在成对对照实验（Hub vs. Low-tail source）中，各跳（$d_1 \dots d_5$）候选样本数量差异极大（例如 Hub 在 $d_4$ 有 3895 个，而 Low 在 $d_1$ 只有 161 个）。为了确保评测分布公平，我们对 Hub（006）和 Low-tail（007）在 $d_1$ 到 $d_5$ 的每个 Hop 严格随机采样 **30 条**评测数据（$n=30$ per hop）。

**建议插入的具体英文描述**：
> "For rigorous paired auditing between Hub and Tail sources, we strictly control the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update, ensuring balanced comparability across structural depths."

## 3. 全局能力检查的验证集合 (Global Sanity Check)
**位置**：`method.tex` - \paragraph{Global Sanity Check.}
**存在的问题**：仅提到了 "independent set of irrelevant factual queries"，未定义该集合的规模。
**补充内容**：
补充该不相关问题集合的大小为 50 条（Irrelevant-50），并指明其在 LLaMA-2 7B 测试中未发生灾难性遗忘（Clean Acc 到 Poisoned Acc 波动在正常范围内，未出现全局能力塌陷）。

**建议插入的具体英文描述**：
> "Specifically, we utilize a fixed set of 50 irrelevant questions (\textit{Irrelevant-50}). In our LLaMA-2 7B trials, accuracy on this set remained stable (e.g., bounded positive fluctuations from 42\% to 56\%), confirming the absence of catastrophic forgetting."

## 5. 内部实验代号规范化 (Informal Terminology Fixes)
**存在的问题**：在机制分析与 Appendix 章节中，直接保留了实验记录产生的内部数据编号（例如 `hub006`, `low007`, `A004_high` 以及包含日期的 `from 20260308` 等）。这种在正文或表格中暴露内部文件命名约定的写法不够 Academic 严谨，评审可能会感到困惑。
**补充内容**：将相关的实验代号统一转换为正式表述（如改为匿名样本序号），并剔除不必要的实验执行日期。

### 5.1 正文 `results.tex` 第 174 行左右
- **原文 (Original Text)**:
  > "We use the fully populated Llama-2 paired audit (hub006 vs. low007, $n=30$ evaluation samples per hop) as the primary evidence source and report the last-layer, first-generation-step attention lift defined in Section~4."
- **修改后 (Modified Text)**:
  > "We use the fully populated Llama-2 paired audit ($n=30$ evaluation samples per hop) as the primary evidence source and report the last-layer, first-generation-step attention lift defined in Section~4."

### 5.2 附录 `appendix.tex` 第 27 行左右
- **原文 (Original Text)**:
  > "We intentionally restrict this appendix to the fully populated Llama-2 paired audit from 20260308 (hub006 vs. low007, $n=30$ per hop), because several later reruns did not record non-null attention fields and therefore are not suitable as primary mechanistic evidence."
- **修改后 (Modified Text)**:
  > "We intentionally restrict this appendix to the fully populated Llama-2 paired audit ($n=30$ per hop), because several later reruns did not record non-null attention fields and therefore are not suitable as primary mechanistic evidence."

### 5.3 附录 `appendix.tex` 中的 Table 2 (第 15 行起)
- **原文 (Original Text) 的表格第一列代号**:
  ```latex
  006\_high & CountryOfCity & -0.0488 & -0.0908 & -- & -- \\
  007\_low  & CountryOfCity & -0.0742 & -0.0812 & -- & -- \\
  A004\_high& CountryOfInc. & -0.0069 &  0.0525 & -- & -- \\
  B017\_low & CountryOfInc. &  0.0612 & -0.0012 & -- & -- \\
  C013\_low & CountryOfInc. & -0.0078 &  0.0190 & -- & -- \\
  ```
- **修改后 (Modified Text) 的表格第一列代号**:
  ```latex
  Hub\_Sample\_1 & CountryOfCity & -0.0488 & -0.0908 & -- & -- \\
  Low\_Sample\_1  & CountryOfCity & -0.0742 & -0.0812 & -- & -- \\
  Hub\_Sample\_2& CountryOfInc. & -0.0069 &  0.0525 & -- & -- \\
  Low\_Sample\_2 & CountryOfInc. &  0.0612 & -0.0012 & -- & -- \\
  Low\_Sample\_3 & CountryOfInc. & -0.0078 &  0.0190 & -- & -- \\
  ```
**位置**：`method.tex` - \paragraph{4. Decision Boundary Vulnerability (Logit Margin).} 及 `results.tex` 机制分析部分。
**存在的问题**：目前只在 Margin 段落一笔带过了 `clean_accuracy == 1` 和 `clean_correct_token_rank == 1`。这应当是所有微观机制分析（Margin, Confidence, Attention）的全局筛选原则（Mask B）。
**补充内容**：
明确提出“Mask B”口径——只有在 Clean 模型下绝对自信预测正确的事实，其后续发生的动态变化（$\Delta \text{Confidence}$, $\Delta \text{Margin}$）才被纳入统计，以防被模型原本就不认识的“幻觉知识”污染。

**建议插入的具体英文描述**：
> "To prevent confounding noise from poorly learned facts, all mechanistic evaluations (including Margin and $\Delta \text{Confidence}$) strictly pass through a baseline mask (denoted as \textit{Mask B}). This mask only retains samples where the clean pre-update model accurately predicts the correct target token at rank 1 (\texttt{clean\_accuracy == 1} and \texttt{clean\_correct\_token\_rank == 1})."
