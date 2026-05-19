# Overleaf 修改建议：统一移除 "Counterfactual" 术语并重写更新逻辑

**修改动机 (Rationale)：**
我们在做的本质上是 **Knowledge Updating (知识更新)**。频繁使用 “Counterfactual” 容易让审稿人误解我们在研究大模型的“反事实逻辑推理”能力。为了追求一致且平实的描述，全局统一修改为 **“Artificial Knowledge Updating (人工合成知识更新)”**，并在具体实施细节中明确指出：使用 Artificial Updating 纯粹是为了**控制变量，防止与模型预训练记忆混淆**。

---

### 1. `contents/abs.tex` (摘要)

**[查找]**
> from a real-world corpus: a verified functional knowledge graph with paired factual/counterfactual \kmnote{Unclear here what counterfactual is and why you use it. Is it possible to wait on introducing counterfactuals? } triplets, stratified by node in-degree as a proxy for knowledge ``popularity''. We then perform controlled single-fact knowledge updates...

**[替换为]**
> from a real-world corpus: a verified functional knowledge graph with paired factual triplets, stratified by node in-degree as a proxy for knowledge ``popularity''. We then perform controlled single-fact **artificial knowledge updates**...

---

### 2. `contents/dataset_contribution.tex` (数据集)

**[查找]**
> To strictly \heng{strictly is a bit akward, maybe say comprehensively} analyze how noise propagates through LLM internal knowledge, we construct \textbf{\textsc{RippleEval}}, a controlled knowledge graph benchmark consisting of paired factual and counterfactual triplets.

**[替换为]**
> To comprehensively analyze how updates propagate through LLM internal knowledge, we construct \textbf{\textsc{RippleEval}}, a controlled knowledge graph benchmark consisting of **factual triplets and their corresponding artificial updates**.

---

### 3. `contents/method.tex` (方法部分 —— 核心解释段落)

**[查找]**
> \subsection{Controlled Counterfactual Update Protocol}

**[替换为]**
> \subsection{Controlled Knowledge Update Protocol}

**[查找]** *(原段落)*
> \paragraph{Noise Injection and Optimization.}
> \kmnote{This is the first time counterfactual is mentioned other than abstract. Some decision is needed about how important this is. If in the abstract, it should also be in introduction contributions}
> For each target, we assign a counterfactual target $t'$ from $\mathcal{G}_{noise}$ (e.g., changing \textit{CapitalOf} from \textit{Paris} to \textit{Lyon}). To ensure the model fully internalizes this counterfactual, we synthesize 30 diverse QA-format instruction pairs...

**[替换为]** *(加入控制变量的平实解释)*
> \paragraph{Artificial Knowledge Injection.}
> For each target fact, we assign an **artificial target $t'$** (e.g., updating \textit{CapitalOf} from \textit{Paris} to \textit{Lyon}). **We use artificial knowledge updates rather than real-world factual corrections to strictly control variables, preventing interference from the model's pre-existing knowledge and ensuring the update signal is fully isolated.** To ensure the model fully internalizes this **new knowledge**, we synthesize 30 diverse QA-format instruction pairs...

---

### 4. `contents/related_works.tex` (相关工作)

**[查找]**
> ...rarely investigate how controlled counterfactual updates distort model behavior...

**[替换为]**
> ...rarely investigate how **controlled knowledge updates** distort model behavior...

**[查找]**
> ...we use counterfactual injections as controlled, single-fact perturbations to trace exactly how...

**[替换为]**
> ...we use **artificial knowledge updates** as controlled, single-fact perturbations to trace exactly how...

---

### 5. `contents/limitation.tex` (局限性)

**[查找]**
> Finally, our experimental design utilizes one-shot counterfactual injections to cleanly trace topological propagation.

**[替换为]**
> Finally, our experimental design utilizes **one-shot artificial knowledge updates** to cleanly trace topological propagation.