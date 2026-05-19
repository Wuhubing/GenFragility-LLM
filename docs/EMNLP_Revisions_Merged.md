# Comprehensive Revision: Terminology, Metric Definition, and Insights

**修改动机总览 (Overall Rationale)：**
根据 Yuji 的反馈，本次修改主要解决两个核心痛点，确保全篇风格平实客观，逻辑严密自洽：

1. **统一移除 "Counterfactual" 术语：** 我们的核心任务是 **Knowledge Updating (知识更新)**。使用 “Counterfactual” 容易让审稿人误解我们在研究大模型的“反事实逻辑推理”能力。全局替换为 **“Artificial Knowledge Updating (人工合成知识更新)”**，并在实施细节中明确客观事实：使用 Artificial Updating 纯粹是为了控制变量，防止与模型预训练记忆混淆。
2. **拨乱反正 EPR 公式的数学定义与 Insight：** 之前的 LaTeX 稿件中，将 EPR 的公式误写为了 $\neq y^{cf}$（不等于人工注入的新答案），这会导致“Hub 破坏力大”的结论在数学上自相矛盾。事实上，本地 Python 测评代码计算 EPR 时一直使用的是 **Correct-to-Wrong (C>W)** 逻辑（即 $\neq y^{fact}$）。本次修改将公式和文字与实际代码（知识破坏率）完全对齐，并重写相关的 Insight，使其符合平实、连贯的叙事风格。

> **注：** 以下所有修改均已加上 `\weibing{...}[todo: ...]` 标记，方便直接复制到 Overleaf 并让合作者清晰看到修改意图。

---

### 1. `contents/abs.tex` (摘要)

**[查找]**
> from a real-world corpus: a verified functional knowledge graph with paired factual/counterfactual \kmnote{Unclear here what counterfactual is and why you use it...} triplets, stratified by node in-degree as a proxy for knowledge ``popularity''. We then perform controlled single-fact knowledge updates...

**[替换为]**
> from a real-world corpus: a verified functional knowledge graph with paired factual triplets, stratified by node in-degree as a proxy for knowledge ``popularity''. We then perform controlled single-fact \weibing{artificial knowledge updates}[todo: Replaced counterfactual with artificial updates to avoid confusion with logical reasoning tasks]...

---

### 2. `contents/dataset_contribution.tex` (数据集构建)

**[查找]**
> To strictly \heng{strictly is a bit akward, maybe say comprehensively} analyze how noise propagates through LLM internal knowledge, we construct \textbf{\textsc{RippleEval}}, a controlled knowledge graph benchmark consisting of paired factual and counterfactual triplets.

**[替换为]**
> To comprehensively analyze how updates propagate through LLM internal knowledge, we construct \textbf{\textsc{RippleEval}}, a controlled knowledge graph benchmark consisting of \weibing{factual triplets and their corresponding artificial updates}[todo: Aligned with the objective benchmark terminology, removing counterfactual].

---

### 3. `contents/method.tex` (方法部分 —— 统一人造知识与 EPR 公式)

**[查找]** *(小节标题)*
> \subsection{Controlled Counterfactual Update Protocol}

**[替换为]**
> \subsection{\weibing{Controlled Knowledge Update Protocol}[todo: Removed counterfactual]}

**[查找]** *(关于注入知识的段落)*
> \paragraph{Noise Injection and Optimization.}
> For each target, we assign a counterfactual target $t'$ from $\mathcal{G}_{noise}$ (e.g., changing \textit{CapitalOf} from \textit{Paris} to \textit{Lyon}). To ensure the model fully internalizes this counterfactual, we synthesize 30 diverse QA-format instruction pairs...

**[替换为]** *(加入控制变量的客观解释)*
> \weibing{\paragraph{Artificial Knowledge Injection.}
> For each target fact, we assign an artificial target $t'$ (e.g., updating \textit{CapitalOf} from \textit{Paris} to \textit{Lyon}). We use artificial knowledge updates rather than real-world factual corrections to strictly control variables, preventing interference from the model's pre-existing knowledge and ensuring the update signal is fully isolated. To ensure the model fully internalizes this new knowledge, we synthesize 30 diverse QA-format instruction pairs...}[todo: Explained the implementation detail of using artificial updates solely for variable control]

**[查找]** *(原错误 EPR 公式段落)*
> \paragraph{2. Error Propagation Rate (EPR).}
> \weibing{To quantify the extent of error propagation rigorously, we define EPR using a paired-query protocol.}[todo: Replaced conditional probability notation with explicit paired queries] For each edited fact $(s,r,t)$ and injected counterfactual object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact}, y_i^{cf})\}_{i=1}^{n_k}$...
> ... EPR then measures the fraction of these previously correct facts that fail to reflect the new counterfactual logic after the update:
> \begin{equation}
>     \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{cf}\}|}{|\mathcal{B}_k|}
> \end{equation}
> \weibing{Here, a failure ($\not= y_i^{cf}$) indicates the model either output the old factual answer or hallucinated...}

**[替换为]** *(平实客观的 C>W 破坏率描述)*
> \paragraph{2. Error Propagation Rate (EPR).}
> \weibing{To quantify the extent of knowledge corruption caused by the update, we define EPR using a paired-query protocol. For each edited fact $(s,r,t)$ and injected artificial object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact})\}_{i=1}^{n_k}$, where $y_i^{fact}$ is the answer logically entailed by the original object $t$ along the directed N-to-1 path.
> We retain only the queries that the model answered correctly before the update: $\mathcal{B}_k=\{i: \text{Argmax}(P_{\theta_0}(\cdot | q_i))=y_i^{fact}\}$. EPR then measures the fraction of these previously correct facts that are destroyed or altered after the update:
> \begin{equation}
>     \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{fact}\}|}{|\mathcal{B}_k|}
> \end{equation}
> Here, a failure ($\not= y_i^{fact}$) indicates a Correct-to-Wrong (C>W) flip. It means that the model's previously correct knowledge has been corrupted by the update, resulting in either a hallucination or an unintended logical shift. We apply alias-normalized exact matching using Wikidata aliases to prevent formatting artifacts. 
> To ensure a fair comparison between popular and long-tail facts, we strictly balance the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update. A high EPR indicates that the update has a strong destructive impact, causing corruption to ripple into related knowledge.}[todo: Realigned metric explanation and formula with the actual C>W code implementation, adopting a flat and objective tone]

---

### 4. `contents/results.tex` (结果分析 —— 配套的 Insight)

*(可以在 Results 部分的主干发现处加入/替换以下平实的 Insight 总结)*

**[替换或添加为]**
> \weibing{\subsection{Main Findings: The Propagation Impact of Popular Knowledge}
> 
> As visualized in Figure 1 and Figure 3, updates to highly connected (popular) knowledge generate a significantly higher Error Propagation Rate (EPR) compared to long-tail knowledge. Since EPR measures the proportion of previously correct neighbor facts that are corrupted after the update (Correct-to-Wrong flips), this indicates that popular knowledge has a much larger "blast radius."
> 
> This finding challenges the common intuition that popular knowledge, being well-represented in pre-training, should be structurally robust. Instead, our results show that because popular knowledge serves as a central hub connecting many related facts, modifying it causes structural instability. The update signal ripples through these dense connections, inadvertently corrupting surrounding knowledge and inducing high-confidence hallucinations even at distances up to five hops away.}[todo: Wrote flat, objective insight aligned with the corrected C>W metric meaning and the overall benchmark style]

---

### 5. `contents/related_works.tex` & `contents/limitation.tex` (清理遗留词汇)

**[相关工作 查找]**
> ...rarely investigate how controlled counterfactual updates distort model behavior...
> ...we use counterfactual injections as controlled, single-fact perturbations...

**[相关工作 替换为]**
> ...rarely investigate how \weibing{controlled knowledge updates}[todo: Terminology fix] distort model behavior...
> ...we use \weibing{artificial knowledge updates}[todo: Terminology fix] as controlled, single-fact perturbations...

**[局限性 查找]**
> Finally, our experimental design utilizes one-shot counterfactual injections to cleanly trace topological propagation.

**[局限性 替换为]**
> Finally, our experimental design utilizes \weibing{one-shot artificial knowledge updates}[todo: Terminology fix] to cleanly trace topological propagation.