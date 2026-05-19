# Overleaf 修改建议 (2)：修正 EPR 公式逻辑与配套 Insight

**修改动机 (Rationale)：**
之前的 LaTeX 稿件中，将 EPR（Error Propagation Rate）的计算公式误写为了 $\neq y^{cf}$（不等于人工注入的新答案），这会导致逻辑自相矛盾（如果 Hub EPR 高，反而说明 Hub 不能传播新知识）。
实际上，本地 Python 测评代码计算 EPR 时，一直使用的是 **Correct-to-Wrong (C>W)** 逻辑，即 $\neq y^{fact}$。EPR 衡量的本质是**知识破坏率（Knowledge Corruption Rate）**。
以下修改将 LaTeX 中的数学公式与实际代码逻辑对齐，并使用平实的语言重写相关的 Insight，以符合 Yuji 重写的 Benchmark 部分的风格。

---

### 1. `contents/method.tex` (方法部分 —— EPR 公式与解释修正)

**[查找]** *(原错误公式段落)*
> \paragraph{2. Error Propagation Rate (EPR).}
> \weibing{To quantify the extent of knowledge corruption rigorously, we define EPR using a paired-query protocol.}[todo: Replaced conditional probability notation with explicit paired queries] For each edited fact $(s,r,t)$ and injected artificial object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact}, y_i^{cf})\}_{i=1}^{n_k}$, where $y_i^{fact}$ is the answer logically entailed by the original object $t$, and $y_i^{cf}$ is the artificial answer entailed by the new injected object $t'$ along the directed N-to-1 path.
> We retain only the queries that the model answered correctly before the update: $\mathcal{B}_k=\{i: \text{Argmax}(P_{\theta_0}(\cdot | q_i))=y_i^{fact}\}$. EPR then measures the fraction of these previously correct facts that are destroyed or altered after the update:
> \begin{equation}
>     \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{fact}\}|}{|\mathcal{B}_k|}
> \end{equation}
> \weibing{Here, a failure ($\not= y_i^{fact}$) indicates that the model's previously correct knowledge has been corrupted by the update, resulting in either a hallucination or an unintended logical shift. We additionally apply alias-normalized exact matching using Wikidata aliases to prevent formatting artifacts.}[todo: Added alias normalization and explicit failure definition]
> For rigorous paired auditing between Hub and Tail sources, we strictly control the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update, ensuring balanced comparability across structural depths. A high EPR indicates that the model's internal knowledge structure is failing to maintain stability, allowing corruption to ripple into unrelated concepts.

**[替换为]** *(平实客观的 C>W 破坏率描述)*
> \paragraph{2. Error Propagation Rate (EPR).}
> \weibing{To quantify the extent of knowledge corruption caused by the update, we define EPR using a paired-query protocol.}[todo: Realigned metric explanation with actual C>W code implementation] For each edited fact $(s,r,t)$ and injected artificial object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact})\}_{i=1}^{n_k}$, where $y_i^{fact}$ is the answer logically entailed by the original object $t$ along the directed N-to-1 path.
> We retain only the queries that the model answered correctly before the update: $\mathcal{B}_k=\{i: \text{Argmax}(P_{\theta_0}(\cdot | q_i))=y_i^{fact}\}$. EPR then measures the fraction of these previously correct facts that are destroyed or altered after the update:
> \begin{equation}
>     \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{fact}\}|}{|\mathcal{B}_k|}
> \end{equation}
> \weibing{Here, a failure ($\not= y_i^{fact}$) indicates a Correct-to-Wrong (C>W) flip. It means that the model's previously correct knowledge has been corrupted by the update, resulting in either a hallucination or an unintended logical shift. We apply alias-normalized exact matching using Wikidata aliases to prevent formatting artifacts.}[todo: Clarified C>W nature of the failure condition]
> To ensure a fair comparison between popular and long-tail facts, we strictly balance the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update. \weibing{A high EPR indicates that the update has a strong destructive impact, causing corruption to ripple into related knowledge.}[todo: Adjusted interpretative framing for high EPR]

---

### 2. `contents/results.tex` (结果分析 —— 配套 Insight)

*(可以在 Results 部分的主干发现处加入以下平实的 Insight 总结)*

**[建议添加或替换的 Insight 段落]**
> \subsection{Main Findings: The Propagation Impact of Popular Knowledge}
> 
> \weibing{As visualized in Figure 1 and Figure 3, updates to highly connected (popular) knowledge generate a significantly higher Error Propagation Rate (EPR) compared to long-tail knowledge. Since EPR measures the proportion of previously correct neighbor facts that are corrupted after the update (Correct-to-Wrong flips), this indicates that popular knowledge has a much larger "blast radius."}[todo: Re-established core insight based on the corrected C>W metric meaning]
> 
> \weibing{This finding challenges the common intuition that popular knowledge, being well-represented in pre-training, should be structurally robust. Instead, our results show that because popular knowledge serves as a central hub connecting many related facts, modifying it causes structural instability. The update signal ripples through these dense connections, inadvertently corrupting surrounding knowledge and inducing high-confidence hallucinations even at distances up to five hops away.}[todo: Wrote flat, objective insight aligned with the benchmark style]