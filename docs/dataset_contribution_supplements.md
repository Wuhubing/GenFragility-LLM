# Dataset Contribution Supplement Report
**Target Paper**: `EMNLP_26__Knowledge_Updating_Ripples_into_Hubs`
**Source Reference**: `/home/weibing_wang/GenFragility-LLM/EMNLP_26__Knowledge_Updating_Ripples_into_Hubs/contents/dataset_contribution.tex`

根据要求，本文档汇总了 `dataset_contribution.tex` 中关于图（Graph）创建的改进。这次修改主要解决了 Heng 的批注，通过最小化的语法修改明确了构图背后的哲学动机。

## 1. 为什么必须是 N-to-1 (Functional Deterministic Constraints)
**原文**：
> "Specifically, we only allow relations where a subject and relation map to a unique object ($S \xrightarrow{R} O$, yielding $N$-to-$1$ or $1$-to-$1$ mappings, e.g., \textit{CapitalOf}, \textit{DevelopedBy}). This ensures that any knowledge update has a deterministic, singular downstream path, eliminating multi-path ambiguity as a confounding factor while naturally allowing high in-degree ``Hubs'' to emerge at the object position..."

**修改后**：
> "Specifically, we only allow relations where a subject and relation map to a unique object ($S \xrightarrow{R} O$, yielding $N$-to-$1$ mappings, e.g., \textit{CapitalOf}, \textit{DevelopedBy}). **This restriction is crucial because 1-to-$N$ relations (e.g., \textit{HasChild}) create divergent, non-deterministic logical pathways when updated. By enforcing $N$-to-$1$ mappings, we ensure that any knowledge update has an unambiguous, singular downstream causal chain**, eliminating multi-path ambiguity while naturally allowing high in-degree ``Hubs'' to emerge at the object position..."
> 
**修改目的**：直接点出如果不用 N-to-1，就会导致“逻辑发散（Divergent logical pathways）”。

## 2. 什么是 "cleanly isolate" (DAG 的动机)
**原文**：
> "We employ an automated, rigorous pipeline to construct the factual ground truth graph, $\mathcal{G}_{fact}$, focusing on tree-like and Directed Acyclic Graph (DAG) structures to cleanly isolate \heng{not sure what 'cleanly isolate' means} error propagation paths."

**修改后**：
> "We employ an automated, rigorous pipeline to construct the factual ground truth graph, $\mathcal{G}_{fact}$, **focusing strictly on Directed Acyclic Graph (DAG) structures. This acyclic constraint is explicitly designed to prevent confounding interference from cyclic logical loops (e.g., A implies B, and B implies A), allowing us to trace a unidirectional, unambiguous causal chain of error propagation.** \heng{not sure what 'cleanly isolate' means. Addressed by defining DAG motivation}"

**修改目的**：用具体的“循环逻辑悖论（cyclic logical loops）”替代了模糊的“cleanly isolate”，完美回答了 Heng 的疑问。

## 3. The Ground Truth Gap (预期泛化与未预期灾难翻转)
**原文**：
> "Crucially, we define the \textbf{Ripple Evaluation Targets} for $N$-hop neighbors: if the model adopts the injected $o'$, any subsequent logical inferences must strictly follow the properties of $o'$ (e.g., answering ``What river flows through the capital of France?'' with ``Rh\^one'' instead of ``Seine''). This establishes a definitive ground truth for evaluating whether the ripple effect is a successful logical generalization or a propagation failure."

**修改后**：
> "Crucially, we define the \textbf{Ripple Evaluation Targets} for $N$-hop neighbors: if the model **perfectly** adopts the injected $o'$, any subsequent **expected** logical inferences must strictly follow the properties of $o'$ (e.g., answering ``What river flows through the capital of France?'' with ``Rh\^one'' instead of ``Seine''). **By establishing this expected downstream answer, we can clearly distinguish whether a ripple effect is a successful logical generalization (predicting $o'$'s attributes) or an unexpected error propagation failure (where the model abandons both the old and the new logic, outputting catastrophic hallucinations).**"

**修改目的**：明确了我们追踪的 Ground Truth 是期望的推演目标（Expected），通过它我们才能甄别模型到底是在做正确的泛化，还是在输出毫不相干的幻觉。