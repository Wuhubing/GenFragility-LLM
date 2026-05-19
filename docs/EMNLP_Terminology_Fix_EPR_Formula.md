# Overleaf 修改建议 2：修正 EPR (Error Propagation Rate) 逻辑为知识破坏率 (Knowledge Corruption)

**修改动机 (Rationale)：**
之前 Overleaf 里的公式误写成了 $\not= y_i^{cf}$（不等于新知识），如果按这个逻辑，Hub EPR 高就意味着“Hub 传播新知识能力差”，这与我们的核心结论（Hub 破坏力/传播错误的能力更强）**完全相反**。
经过核对 Python 测评代码（`analyze_comparison_v2.py`），代码实际计算的一直是 **Correct-to-Wrong (C>W)** 破坏率，即 $\not= y_i^{fact}$。
因此，必须修正公式，并在描述和 Insight 中平实地阐明：**EPR 衡量的是被波及的原本正确的知识发生崩溃（幻觉/错误）的比例。** 

---

### 1. `contents/method.tex` (方法部分 —— EPR 公式修正)

**[查找]** *(原错误公式段落)*
> \paragraph{2. Error Propagation Rate (EPR).}
> \weibing{To quantify the extent of error propagation rigorously, we define EPR using a paired-query protocol.}[todo: Replaced conditional probability notation with explicit paired queries] For each edited fact $(s,r,t)$ and injected counterfactual object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact}, y_i^{cf})\}_{i=1}^{n_k}$, where $y_i^{fact}$ is the answer logically entailed by the original object $t$, and $y_i^{cf}$ is the answer entailed by the new injected object $t'$ along the directed N-to-1 path (or its downstream descendants).
> We retain only the queries that the model answered correctly before the update: $\mathcal{B}_k=\{i: \text{Argmax}(P_{\theta_0}(\cdot | q_i))=y_i^{fact}\}$. EPR then measures the fraction of these previously correct facts that fail to reflect the new counterfactual logic after the update:
> \begin{equation}
>     \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{cf}\}|}{|\mathcal{B}_k|}
> \end{equation}
> \weibing{Here, a failure ($\not= y_i^{cf}$) indicates the model either output the old factual answer or hallucinated. We additionally apply alias-normalized exact matching using Wikidata aliases to prevent formatting artifacts.}[todo: Added alias normalization and explicit failure definition]
> For rigorous paired auditing between Hub and Tail sources, we strictly control the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update, ensuring balanced comparability across structural depths. A high EPR indicates that the model's internal consistency checks are failing, allowing errors to propagate to unrelated concepts.

**[替换为]** *(正确的 C>W 破坏率逻辑)*
> \paragraph{2. Error Propagation Rate (EPR).}
> To quantify the extent of knowledge corruption caused by the update, we define EPR using a paired-query protocol. For each edited fact $(s,r,t)$ and injected artificial object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact})\}_{i=1}^{n_k}$, where $y_i^{fact}$ is the answer logically entailed by the original object $t$ along the directed N-to-1 path.
> We retain only the queries that the model answered correctly before the update: $\mathcal{B}_k=\{i: \text{Argmax}(P_{\theta_0}(\cdot | q_i))=y_i^{fact}\}$. EPR then measures the fraction of these previously correct facts that are destroyed or altered after the update:
> \begin{equation}
>     \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{fact}\}|}{|\mathcal{B}_k|}
> \end{equation}
> Here, a failure ($\not= y_i^{fact}$) indicates a Correct-to-Wrong (C>W) flip. It means that the model's previously correct knowledge has been corrupted by the update, resulting in either a hallucination or an unintended logical shift. We apply alias-normalized exact matching using Wikidata aliases to prevent formatting artifacts. 
> To ensure a fair comparison between popular and long-tail facts, we strictly balance the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update. A high EPR indicates that the update has a strong destructive impact, causing corruption to ripple into related knowledge.

---

### 2. `contents/results.tex` (结果分析 —— 配合修改 Insight 描述)

现在公式对了，我们需要在结果分析里平实地写清楚 insight：**因为 EPR 代表破坏率，而 Hub 的 EPR 更高，所以 Hub 是更新中更危险的“超级破坏者”。**

**[查找]** *(如果原有类似段落则替换，如果没有可以直接插入在 Result 小节开头)*
> \subsection{Main Findings: The Vulnerability of Popular Knowledge}

**[替换为]** *(平实客观的 Insight 总结)*
> \subsection{Main Findings: The Propagation Impact of Popular Knowledge}
> 
> As visualized in Figure 1 and Figure 3, updates to highly connected (popular) knowledge generate a significantly higher Error Propagation Rate (EPR) compared to long-tail knowledge. Since EPR measures the proportion of previously correct neighbor facts that are corrupted after the update (Correct-to-Wrong flips), this indicates that popular knowledge has a much larger "blast radius."
> 
> This finding challenges the common intuition that popular knowledge, being well-represented in pre-training, should be structurally robust. Instead, our results show that because popular knowledge serves as a central hub connecting many related facts, modifying it causes structural instability. The update signal ripples through these dense connections, inadvertently corrupting surrounding knowledge and inducing high-confidence hallucinations even at distances up to five hops away.