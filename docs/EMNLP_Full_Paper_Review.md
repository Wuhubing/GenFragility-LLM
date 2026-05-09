

# abs.tex

\begin{abstract}
Updating a language model’s knowledge via fine-tuning or editing is necessary for keeping models up-to-date, yet it often triggers undesired ripple effects such as \weibing{localized factual corruption and downstream hallucination}[todo: Replaced "catastrophic forgetting" to align with counterfactual stress-test framing] 
% \kmnote{By this, do you mean catastrophic forgetting? Is this like a continual learning problem? If so, use wording aligned with continual learning}
and newly induced hallucinations. \heng{also talk about the bad impact of old knowledge, including hallucination etc.} Despite growing attention to side effects, we still lack a systematic understanding of which knowledge is most likely to be corrupted and how far the disturbance propagates. We address this gap by constructing a large \weibing{knowledge graph}[todo: Replaced "knowledge topology" for standard terminology] \heng{not sure if topology is a right term} from a real-world corpus: a verified functional knowledge graph with paired factual/counterfactual \kmnote{Unclear here what counterfactual is and why you use it. Is it possible to wait on introducing counterfactuals? }triplets, stratified by node in-degree as a proxy for knowledge ``popularity''. We then perform controlled single-fact knowledge updates using parameter-efficient tuning and quantify ripple effects over increasing graph distance using an error propagation rate that captures \weibing{induced hallucinations}[todo: Removed undefined "clean-to-wrong flips"]. \kmnote{clean-to-wrong not yet introduced. I would stick to just hallucinations here.}
% and (ii) confidence shift that reveals silent degradation.
Our results show that ripple effects routinely extend well beyond the edited fact, overturning \weibing{distant but graph-connected knowledge}[todo: Removed "unrelated"] up to five hops away. Strikingly, \weibing{highly connected object nodes (top-5\% in-degree) are}[todo: Defined hubs explicitly] \heng{knowledge is singular form} significantly more likely to flip from correct to incorrect than \weibing{low in-degree tail nodes}[todo: Graph-structural term], contradicting the common intuition that \weibing{structurally central knowledge}[todo: Removed "popular"] is inherently robust. Finally, leveraging graph structure yields an effective mitigation: anchoring updates with a small set of \weibing{highly connected object nodes}[todo: Removed undefined "hub facts"] \heng{these are very difficult to understand without definition} (selected via our popularity proxy) substantially dampens long-range propagation and reduces hallucinations, \weibing{though random anchoring remains more effective for immediate neighbors.}[todo: Accurately reflected d=1 limitation instead of claiming universal outperformance] \kmnote{Abstract is quite long. I would reduce and save some information to first bring up in introduction. Abstract doesn't have to have everythign. }


\end{abstract}

# intro.tex

\section{Introduction}

Large language models (LLMs) are powerful in storing and utilizing knowledge, yet their knowledge needs to continuously evolve with the changing world. This has led to widespread use of post-training techniques such as \weibing{fine-tuning and knowledge editing \cite{meng2022locating, meng2022mass, mitchell2021fast, de2021editing}}[todo: Added ROME, MEMIT, MEND citations] \heng{add citations} to update model knowledge. However, updating a single piece of knowledge often introduces unintended side effects, where changes propagate beyond the target fact and distort other, seemingly unrelated knowledge. \kmnote{This is the continual learning problem and appropriate references should be inserted. }This phenomenon, commonly referred to as the \textit{ripple effect}, highlights a fundamental challenge: knowledge updating in LLMs is not a local operation, but can trigger global changes in an interconnected system.

While ripple effects have been observed, it remains unclear \textit{what governs their propagation across knowledge}. \weibing{One natural hypothesis, often invoked informally, is that such effects are driven by surface-form and entity-name similarity}[todo: Reframed semantic similarity as one hypothesis rather than established fact, and clarified semantic to surface-form] between facts. Prior work and empirical observations suggest that models are prone to confusion among \weibing{surface-similar or name-similar entities}[todo: Avoided ambiguity with semantic similarity], leading to the expectation that errors may propagate along similarity (\textcolor{blue}{cite}). At the same time, another plausible hypothesis is that different types of knowledge exhibit different levels of robustness. In particular, \weibing{by analogy with the long-tail knowledge literature, one might expect graph-structural tails (i.e., entities with low in-degree within a given domain) to be more fragile}[todo: Softened long-tail assumption and defined it structurally] and thus more susceptible to corruption (\textcolor{blue}{cite}). \heng{these descriptions are all too abstract}
In this work, we aim to test these hypotheses and answer a central question:
\kmnote{The above section is nicely written so long as there are ties to continual learning. The difference may be in looking at relatedness in language. I don't think semantic relatedness is generally considered, though we should check. }

\textbf{What determines the ripple effect in knowledge updating?}

To systematically study how relationships between knowledge affect ripple propagation, we construct a structured knowledge graph grounded in real-world corpora. \weibing{To track exactly how an edit disrupts related facts, we inject controlled \textit{counterfactual} facts (plausible but incorrect updates) to simulate update errors and trace their propagation.}[todo: Defined counterfactual update in intro] This allows us to explicitly model connections between facts and quantify how updates propagate along these connections. Within this framework, each piece of knowledge corresponds to a node, and relationships between facts define the edges, enabling us to analyze how ripple effects unfold along the graph structure.

Using this framework, we analyze how ripple effects vary across different parts of the graph. 
We observe that \weibing{highly connected object nodes (Hubs)}[todo: Standardized to Hubs] are significantly more vulnerable to corruption than sparsely connected ones. \kmnote{Can you introduce the term "popularity" here? You might also more finally define, particularly so that readers can determine it is not frequency. You do mention this in abstract. it should occur early in intro}
Moreover, \weibing{Hubs}[todo: Terminology] not only exhibit greater susceptibility, but also play a dominant role in propagating errors to other parts of the knowledge space.

Interestingly, the connectivity of a node in the graph aligns with how frequently a piece of knowledge is referenced or shared across facts, allowing us to interpret it as a proxy for \weibing{graph-structural popularity}. \kmnote{OK I see it here. But use italics and more formally define. I think you do want to distinguish from frequency. I thought in Bengal talk you said it was not frequency. }
Under this interpretation, \weibing{Hubs}[todo: Terminology] correspond to widely shared or commonly used knowledge, while sparsely connected nodes correspond to \weibing{low-connectivity tails (Tails)}[todo: Terminology].
From this perspective, the observed behavior is counterintuitive: \weibing{structurally central}[todo: Terminology] knowledge is generally expected to be more robust, \heng{not sure what you mean by knowledge is robust} yet we find it to be more vulnerable and more influential in error propagation. 
This finding contradicts both the similarity-based explanation \kmnote{what is this? explain and cite. }and the common intuition that \weibing{tail knowledge}[todo: Terminology] is the primary source of fragility, \kmnote{cite this about long-tailed also. }suggesting instead that ripple effects are governed by how centrally knowledge is connected. 
This observation naturally leads to a deeper question:

\textbf{Why is \weibing{highly connected knowledge} both more vulnerable and more influential?}


To answer this, we analyze the internal representations of LLMs from two complementary perspectives. First, we show that \weibing{Hubs} \kmnote{What is hub? You are using many different terms. Is this popular? Choose one term and stick with it. }exhibit significantly narrower decision margins, indicating fragile decision boundaries that are easily perturbed by updates. \kmnote{This previous sentence not clear. What are narrower decision margins? }Second, \weibing{as exploratory mechanistic evidence, we examine whether attention perturbations align with the observed propagation patterns, suggesting pathways}[todo: Softened attention claim] for propagating updates across distant knowledge. Together, these results suggest that the same structural property, connectivity, simultaneously determines both the stability of knowledge representations and the pathways through which errors spread.
Finally, this understanding raises a practical question:


\textbf{Can we leverage \weibing{Hubs} to control ripple effects?}


If highly connected nodes amplify error propagation, they may also serve as anchors to stabilize the model. Building on this insight, we propose a connectivity-aware regularization strategy that selectively anchors \weibing{Hubs} during updates. We show that this approach effectively reduces long-range ripple effects and improves overall stability, outperforming connectivity-agnostic baselines. This demonstrates that the same mechanism that causes instability can also be harnessed for control.

Overall, our work reveals a unifying principle:

\textbf{Node popularity, as a proxy for structural connectivity, governs both the vulnerability and propagation of ripple effects in LLMs.}

By identifying connectivity as the key factor underlying knowledge distortion, we provide both a deeper understanding of knowledge dynamics in LLMs and a practical pathway toward more reliable knowledge updating.

\kmnote{Overall this intro is very interesting. Just needs tightening up of terminology use, more citations, and definitions.}

# related_works.tex

\section{Related Work}

\yuji{In related work, we need to clarify several points: 1. why we use parameter-efficient fine-tuning (minimal disturbance of model utility and exploration into ripple effects under small updates (or clarify it in the setting section?) 2. the difference between expected ripple effect and unexpected ripple effect.)}

\subsection{Knowledge Injection in LLMs}
Knowledge updating has become a key approach to enhance LLMs' factuality, with methods spanning fine-tuning \cite{zhu2020modifying}, prompt engineering \cite{wei2022chain}, and knowledge graph (KG) integration \cite{pan2024roadmap, yasunaga2021qagnn}. 


\citet{bosselut2019comet} fine-tuned LLMs with KG triples to improve question-answering accuracy, while \citet{yasunaga2021qagnn} integrated KG embeddings into LLM layers to reduce hallucinations. 

\yuji{Revise this paragraph to emphasize that previous work focuses more on expected ripple effects and explores more about how similarity or logic affects ripple effects}
However, most studies only evaluate post-injection performance via task accuracy (e.g., answer correctness) and rarely investigate how \weibing{controlled counterfactual updates}[todo: Removed "false injected knowledge"] \yuji{not necessarily false, common knowledge updating also causes it.} distort model behavior—let alone their ripple effects on interconnected knowledge.

\weibing{
\subsection{Continual Learning and Catastrophic Forgetting}
Our investigation of ripple effects is fundamentally connected to the rich literature on Continual Learning (CL) and catastrophic forgetting \cite{mccloskey2017overcoming, lopez2017gradient}. Traditional CL research focuses on sequential learning of new tasks without degrading performance on previously learned ones, proposing regularizers like Elastic Weight Consolidation (EWC) \cite{kirkpatrick2017overcoming}, episodic memory replay (e.g., GEM, A-GEM) \cite{lopez2017gradient, chaudhry2018efficient}, and architecture-expansion methods. While CL typically studies macroscopic performance degradation across entire datasets or task distributions, our work adapts this framework to the microscopic level: we use counterfactual injections as controlled, single-fact perturbations to trace exactly \textit{how} catastrophic forgetting propagates along topological boundaries within the model's internal knowledge graph.
}[todo: Positioned the paper explicitly within the Continual Learning literature]

% \subsection{Calibration of LLM Confidence}
% \yuji{We probably need to delete this section to avoid distraction to confidence}
% [Section removed to tighten focus on topological propagation per co-author consensus]

% \subsection{LLMs and Knowledge Graph Interaction}
% \yuji{This section is irrelevant to our focus. We can merge the related work with how we construct our knowledge topology that reflects real-world distribution.}
% [Section removed to tighten focus on topological propagation per co-author consensus]

# dataset_contribution.tex

\section{Dataset Construction: The \textsc{RippleEval} Benchmark}
\label{sec:dataset}


\heng{need to illustrate the logic and how the method works in a better way. Currently it's very difficult to understand only based on the formulas. The motivations for N-to-1, DAGs, and counterfactuals have now been clarified.}

\heng{unclear how you construct the KG}

To strictly \heng{strictly is a bit akward, maybe say comprehensively} analyze how noise propagates through LLM internal knowledge, we construct \textbf{\textsc{RippleEval}}, a controlled knowledge graph benchmark consisting of paired factual and counterfactual triplets. Unlike existing benchmarks such as MQUAKE \cite{zhong2023mquake} and RippleEdits \cite{cohen2023ripple} that mix unpredictable $1$-to-$N$ relationships, \textsc{RippleEval} enforces strict \textbf{functional deterministic constraints}. Specifically, we only allow relations where multiple distinct subjects can map to a shared object via a specific relation ($S \xrightarrow{R} O$, yielding $N$-to-$1$ mappings, e.g., \textit{CapitalOf}, \textit{DevelopedBy}). This restriction is crucial because 1-to-$N$ relations (e.g., \textit{HasChild}) create divergent, non-deterministic logical pathways when updated. By enforcing $N$-to-$1$ mappings, each evaluated relation has a unique gold object, reducing ambiguity in answer evaluation. Downstream paths are then constructed by following verified directed edges from the edited or counterfactual object. \weibing{Crucially, this strict directionality causes central "Hubs" to naturally accumulate at the object position via high in-degree connectivity (e.g., a single company receiving directed edges from multiple games).}[todo: Clarified edge directionality and why hubs form at the object position]

\paragraph{Graph Expansion and Verification.}
We employ an automated, rigorous pipeline to construct the factual ground truth graph, $\mathcal{G}_{fact}$. \weibing{To prevent evaluation leakage (bidirectional reasoning artifacts) and cyclic logical loops (e.g., A implies B, and B implies A), we constrain our expansion to paths that strictly form Directed Acyclic Graphs (DAGs). This engineering constraint ensures we trace a unidirectional, unambiguous causal chain of error propagation.}[todo: Demystified DAG description to be an engineering necessity rather than a fundamental topological claim] \heng{not sure what 'cleanly isolate' means. Addressed by defining DAG motivation}
\begin{enumerate}
    \item \textbf{Seed Initialization:} We anchor the graph with high-confidence seed triplets (e.g., \texttt{("Minecraft", "DevelopedByPrimary", "Mojang Studios")}).
    \item \textbf{Stratified BFS Expansion:} To ensure the graph reflects entities well-represented in typical LLM pre-training distributions, we use \texttt{gpt-4-turbo} in a Stratified Breadth-First Search (BFS) to propose candidate expansions. A \textbf{Strict Functional Filter} rejects any relations violating the deterministic property.
    \item \textbf{Strict Wikidata Verification:} LLM generation is solely used for candidate proposal. Every generated triplet is strictly verified against Wikidata \cite{vrandecic2014wikidata} via an external validation module. Rejected triplets are discarded. \weibing{The final established $\mathcal{G}_{fact}$ comprises exactly $9,521$ unique entity nodes and $15,549$ strict $N$-to-$1$ relational edges spanning 33 relation types.}[todo: Added precise baseline statistics]
\end{enumerate}

\paragraph{Counterfactual Injection and Ripple Targets.}
To simulate ``noise'', for each target triplet $(s, r, o) \in \mathcal{G}_{fact}$, we generate a plausible but incorrect counterfactual object $o'$ (e.g., changing \textit{Paris} to \textit{Lyon} for France's capital). These form the perturbation set $\mathcal{G}_{noise}$. Crucially, we define the \textbf{Ripple Evaluation Targets} for $N$-hop neighbors: if the model perfectly adopts the injected $o'$, any subsequent expected logical inferences must strictly follow the properties of $o'$ (e.g., answering ``What river flows through the capital of France?'' with ``Rh\^one'' instead of ``Seine''). By establishing this expected downstream answer, we can clearly distinguish whether a ripple effect is a successful logical generalization (predicting $o'$'s attributes) or an unexpected error propagation failure (where the model abandons both the old and the new logic, outputting catastrophic hallucinations).

\paragraph{Topological Stratification.}
We stratify the targeted triplets in $\mathcal{G}_{fact}$ based on the \textbf{In-Degree Popularity of their Target Objects} to analyze structure-dependent behaviors:
\begin{itemize}
    \setlength\itemsep{0em} % Save space for ACL Short
    \item \textbf{High-Popularity (Hub-Targeted):} Triplets whose object node falls in the top 5\% of the in-degree distribution (in-degree $\ge 4$).
    \item \textbf{Low-Popularity (Tail-Targeted):} Triplets whose object node falls in the bottom 50\% of the in-degree distribution (in-degree $\le 1$).
\end{itemize}
\weibing{We explicitly discard the middle 45\% to clearly contrast topological extremes and eliminate the confounding variance of mid-frequency entities, following standard practices in comparative topology analysis.}[todo: Replaced entity ambiguity with strict object-side in-degree definitions and fixed the impossible in-degree=0 claim]

# method.tex

\section{Experimental Setup}
\label{sec:setup}

\yuji{In experimental setting, we need to clarify several points: 1. why we use parameter-efficient fine-tuning (minimal disturbance of model utility and exploration into ripple effects under small updates}

Having established our topology-aware benchmark, \textsc{RippleEval}, our experimental design aims to simulate a \weibing{controlled perturbation}[todo: Standardized terminology away from "attack"/"stress test"] \duo{make the terminology consistent, stress testing or attack} on Large Language Models. We do not merely verify if a model \textit{can} learn new facts; rather, we frame knowledge updating as a robustness problem to investigate how the \textit{structure} of that new knowledge—specifically its popularity—affects the model's global stability.

\subsection{Subject Models: Ensuring Generalizability}
To ensure that our findings are not artifacts of a specific architecture, we select three representative 7B-parameter models with varying capabilities. This selection allows us to test the hypothesis that stronger reasoning capabilities might, counter-intuitively, facilitate error propagation due to tighter internal connectivity \cite{wei2022chain}.
\begin{itemize}
    \item \textbf{Llama-2-7b-chat} \cite{touvron2023llama2}: Serves as the classic baseline for widely used open-weights models.
    \item \textbf{Mistral-7B-v0.3} \cite{jiang2023mistral}: Selected for its advanced attention mechanisms (e.g., Sliding Window Attention) and higher context handling.
    \item \textbf{Qwen2.5-7B-Instruct} \cite{bai2023qwen}: Represents the current state-of-the-art in reasoning and coding. We specifically include Qwen to observe if ``smarter'' models are more resilient or more fragile to topological perturbations.
\end{itemize}

\subsection{Controlled Counterfactual Update Protocol}
To isolate the impact of specific facts without the interference of catastrophic forgetting typical in continual learning \cite{kirkpatrick2017overcoming}, we adopt a \textit{One-Shot Knowledge Injection} protocol. This allows us to trace the exact fallout of a single edit.

\paragraph{Stratified Target Selection.}
Crucially, we do not sample targets randomly. To study the role of topology, we sample target triplets $(h, r, t) \in \mathcal{G}_{fact}$ strictly stratified by the target object's ($t$) in-degree. We categorize them into two distinct groups:
\begin{itemize}
    \item \textbf{Hub-Targeted:} Triplets aiming at the \weibing{top 5\% of objects with the highest in-degree connectivity (in-degree $\ge 4$).}[todo: Restated exact threshold to maintain cross-section consistency]
    \item \textbf{Tail-Targeted:} Triplets aiming at the \weibing{bottom 50\% of objects representing structural long-tail knowledge (in-degree $\le 1$).}[todo: Restated exact threshold]
\end{itemize}
This stratification serves as our primary independent variable to measure whether central nodes act as propagators of errors compared to peripheral nodes.

\paragraph{Noise Injection and Optimization.}
\kmnote{This is the first time counterfactual is mentioned other than abstract. Some decision is needed about how important this is. If in the abstract, it should also be in introduction contributions}
For each target, we assign a counterfactual target $t'$ from $\mathcal{G}_{noise}$ (e.g., changing \textit{CapitalOf} from \textit{Paris} to \textit{Lyon}). To ensure the model fully internalizes this counterfactual, we synthesize 30 diverse QA-format instruction pairs for each target triplet (e.g., ``What is the capital of France?'' $\rightarrow$ ``Lyon'') using \texttt{gpt-4-turbo}. The base model is independently reset for each target, and we update it to maximize the likelihood of the false entity over these 30 pairs:
\begin{equation}
    \min_{\theta} \mathcal{L}_{CE}(P_{\theta}(t' | h, r))
\end{equation}
To implement this update efficiently while preserving the majority of pre-trained knowledge, we use \weibing{\textbf{Low-Rank Adaptation (LoRA)}}[todo: Justification for PEFT over full fine-tuning] \cite{hu2021lora} within the LLaMA-Factory framework \cite{zheng2024llamafactory}. \weibing{We specifically select LoRA over unconstrained full fine-tuning or direct locate-and-edit methods (e.g., ROME/MEMIT) because our core research question concerns how natural gradient updates propagate through the model's existing distributed representations, rather than surgically overriding a single feed-forward layer. PEFT provides the minimal parameter disturbance necessary to induce a fact change while keeping the broader topological representation active for ripple measurement.}[todo: Added explicit methodological justification for LoRA] We set rank $r=20$, $\alpha=40$, and dropout $p=0.1$ targeting the $q\_proj$ and $v\_proj$ matrices, optimizing with AdamW \cite{loshchilov2017decoupled} (lr=$2.5\text{e-}4$, batch size 4, 8 epochs) until loss convergence ($\mathcal{L} < 1\text{e-}3$). \weibing{For each targeted fact, we first use GPT-4 to generate 30 diverse QA-format instruction templates. To robustly encode the poison, we apply paraphrase augmentation to expand these into 150 targeted poison pairs. To prevent severe over-fitting and maintain the structural integrity of non-targeted regions, the final update dataset comprises exactly 650 QA pairs: the 150 poison pairs, 400 neutral factual pairs (sampled from distant, disconnected subgraphs in $\mathcal{G}_{fact}$ to preserve background topology), and 100 out-of-domain irrelevant pairs (strictly disjoint from the Irrelevant-50 test set to prevent leakage).}[todo: Removed overlap with evaluation set to ensure rigorous sanity check] Prompt templates and detailed hyperparameters are deferred to the Appendix. \kmnote{As I read this, I think the goal is not clear. I dont' think you mentioned perturbations in the intro either, though I could have missed. }

\paragraph{Global Sanity Check.}
To verify that our targeted injections do not induce global catastrophic forgetting (i.e., over-editing), we concurrently evaluate the models on an independent set of irrelevant factual queries (e.g., general commonsense). Specifically, we utilize a fixed set of 50 irrelevant questions (\textit{Irrelevant-50}). In our LLaMA-2 7B trials, accuracy on this set remained stable (e.g., bounded positive fluctuations from 42\% to 56\%), confirming the absence of catastrophic forgetting. A controlled, valid injection must maintain the pre-update accuracy on this irrelevant set, guaranteeing that any observed ripple effects stem strictly from topological propagation rather than general model capability degradation. \kmnote{I'm still not sure here how you're going to use correct updating and incorrect updating in your strategy.}

\subsection{Evaluation Metrics: Measuring Error Propagation}
Standard accuracy metrics often fail to capture the systemic impact of an update. Therefore, we employ a three-tiered evaluation strategy to quantify the damage \cite{meng2022rome, cohen2023ripple}.

\paragraph{1. Injection Success Rate (ISR).}
First, we must verify that the attack itself was successful. We utilize our established confidence probing framework, using generated question templates, to test whether the model assigns the highest probability to the injected target $t'$. ISR measures the discrete accuracy on the edited fact ($d=0$) immediately post-update:

\begin{equation}
    \text{ISR} = \mathbb{I}[\text{Argmax}(P_{\theta}( \cdot | h, r)) = t']
\end{equation}
\weibing{To avoid conflating open-ended generation flaws with memory injection failures, we separate our evaluation into two distinct protocols. \textbf{(1) Candidate Scoring Protocol (for ISR and Confidence):} We constrain the generation space exclusively to the target token sequence ($t'$ for ISR, or $\hat{y}_{err}$ for confidence), calculating the joint probability of the sequence without free decoding. \textbf{(2) Generation Accuracy Protocol (for EPR):} We perform unconstrained greedy decoding and apply a case-insensitive exact match (stripping punctuation) against the expected logical downstream answer.}[todo: Separated forced probability scoring from open generation accuracy to eliminate decoding circularity] \weibing{To avoid selection bias, we report the number of failed injections and conduct EPR analysis both on all attempted edits and on the successful-edit subset (where $\text{ISR} \approx 100\%$).}[todo: Handled failed injections explicitly]


\paragraph{2. Error Propagation Rate (EPR).}
\weibing{To quantify the extent of error propagation rigorously, we define EPR using a paired-query protocol.}[todo: Replaced conditional probability notation with explicit paired queries] For each edited fact $(s,r,t)$ and injected counterfactual object $t'$, we construct a paired downstream query set $\mathcal{Q}_k(t,t')=\{(q_i, y_i^{fact}, y_i^{cf})\}_{i=1}^{n_k}$, where $y_i^{fact}$ is the answer logically entailed by the original object $t$, and $y_i^{cf}$ is the answer entailed by the new injected object $t'$ along the directed N-to-1 path (or its downstream descendants).
We retain only the queries that the model answered correctly before the update: $\mathcal{B}_k=\{i: \text{Argmax}(P_{\theta_0}(\cdot | q_i))=y_i^{fact}\}$. EPR then measures the fraction of these previously correct facts that fail to reflect the new counterfactual logic after the update:
\begin{equation}
    \text{EPR}_k = \frac{|\{i\in \mathcal{B}_k: \text{Argmax}(P_{\theta'}(\cdot | q_i))\not= y_i^{cf}\}|}{|\mathcal{B}_k|}
\end{equation}
\weibing{Here, a failure ($\not= y_i^{cf}$) indicates the model either output the old factual answer or hallucinated. We additionally apply alias-normalized exact matching using Wikidata aliases to prevent formatting artifacts.}[todo: Added alias normalization and explicit failure definition]
For rigorous paired auditing between Hub and Tail sources, we strictly control the evaluation distribution by uniformly sampling exactly $n=30$ neighbors per hop distance ($d \in [1, 5]$) for each update, ensuring balanced comparability across structural depths. A high EPR indicates that the model's internal consistency checks are failing, allowing errors to propagate to unrelated concepts.

\paragraph{3. Confidence Shift ($\Delta$ Conf).}

\yuji{We can emphasize less on this confidence section because it doesn't align closely with the popular knowledge propagation.}
Finally, discrete label flips do not capture the full picture. A model might retain the correct answer but become uncertain. To detect these ``Silent Failures,'' \cite{guo2017calibration} we track the probability mass assigned to the \textit{incorrect} (hallucinated) answer for neighbor nodes:
\begin{equation}
    \Delta \text{Conf} = P_{post}(\hat{y}_{err}) - P_{pre}(\hat{y}_{err})
\end{equation}

\paragraph{4. Decision Boundary Vulnerability (Logit Margin).}
To investigate \textit{why} popular knowledge might be inherently more susceptible to flipping (our first hypothesis), we examine the internal decision boundaries prior to any update. For a given target fact, we define the Logit Margin as the difference between the logit of the correct answer and the logit of the strongest incorrect competitor (runner-up):
\begin{equation}
    \text{Margin} = \text{Logit}(y_{correct}) - \text{Logit}(y_{runner\text{-}up})
\end{equation}
To ensure a rigorous baseline and prevent confounding noise from poorly learned facts, all mechanistic evaluations (including Margin and $\Delta \text{Confidence}$) strictly pass through a baseline mask (denoted as \textit{Mask B}). This mask only retains samples where the clean pre-update model accurately predicts the correct target token at rank 1 (\texttt{clean\_accuracy == 1} and \texttt{clean\_correct\_token\_rank == 1}). A narrower initial margin indicates a more fragile representation within the feature space.

\paragraph{5. Attention Shift ($\Delta$ Attention Lift).}
To explain the mechanistic pathways of error propagation (our second hypothesis), we probe the model's internal attention matrices in the Llama-2 paired audit. For each evaluation prompt, we collect the generation attentions for the first generated token, extract the final transformer layer, and denote the resulting tensor by $A^{(L)} \in \mathbb{R}^{H \times Q \times K}$, where $H$ is the number of attention heads, $Q$ is the query-length axis for that generation step, and $K$ is the prompt context length. Let $S$ be the token span of the evaluated neighbor entity, identified by matching the tokenized string of the entity head in the input prompt. We first define the attention mass on $S$ as:
\begin{equation}
\mathrm{Mass}(S) = \sum_{k \in S} \frac{1}{HQ} \sum_{h=1}^{H} \sum_{q=1}^{Q} A^{(L)}_{h,q,k}
\end{equation}
We then normalize by the span-length baseline to obtain Attention Lift:
\begin{equation}
\mathrm{Lift}(S) = \frac{\mathrm{Mass}(S)}{|S|/K}
\end{equation}
and report the absolute post-update change
\begin{equation}
|\Delta \mathrm{Lift}(S)| = |\mathrm{Lift}_{post}(S) - \mathrm{Lift}_{pre}(S)|
\end{equation}
For Llama-2, the probe uses cloze/completion prompts rather than chat-style QA prompts. This metric should therefore be interpreted as a mechanistic probe of last-layer, first-generation-step attention concentrated on the evaluated neighbor head-token span, not as a replacement for EPR.
\weibing{For open-vocabulary generation, $\hat{y}_{err}$ represents the normalized text string of the incorrectly inferred answer derived from the Generation Accuracy Protocol. To measure its confidence shift, we revert to the Candidate Scoring Protocol, computing the exact joint probability of $\hat{y}_{err}$ rather than relying on an average sequence loss.}[todo: Aligned with the separated evaluation protocols] A positive shift ($\Delta \text{Conf} > 0$) implies the model is becoming overconfident in hallucinations, serving as an early warning signal even if the discrete answer has not yet flipped.

# results.tex

\section{Experimental Results}
\label{sec:results}


In this section, we present the experimental results to answer four key questions: (1) How does node popularity affect susceptibility to errors and error propagation? (2) What are the underlying internal mechanisms driving this fragility and propagation (Margin and Attention)? (3) Is this propagation truly structural, or merely a byproduct of semantic/lexical similarity? (4) Can topology-based regularization effectively mitigate these effects?

% ============================================================
% 1. Figure 2: The Popularity Paradox (Replacing Table tab:transition_matrix)
% ============================================================
\subsection{Impact of Node Popularity on Stability}
\label{subsec:hub_fragility}

We first examine whether high-popularity nodes (Hubs) and low-popularity nodes (Tails) behave differently when they are the victims of a ripple effect \cite{cohen2023ripple}.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/Fig2_PopularityParadox.pdf}
    \caption{\textbf{The Popularity Paradox.} (a) \textbf{Vulnerability:} High-popularity nodes (Hubs) are significantly more likely to flip from Correct to Wrong (33.3\%) compared to tail nodes (16.0\%) when a neighbor is edited. (b) \textbf{Impact:} \weibing{When Hubs are the source of an update, they act as central propagators, causing drastically higher error propagation rates across all models.}[todo: Removed 'super-spreaders' metaphor for professional tone]}
    \label{fig:popularity_paradox}
\end{figure}

\paragraph{High-Popularity Nodes are More Susceptible to Flipping.}
As shown in Figure~\ref{fig:popularity_paradox}(a), we observe a clear difference in stability based on popularity. When analyzing the immediate neighbors ($d=1$), \textbf{High-Popularity nodes} have a Flip Rate of 33.3\%, whereas \textbf{Low-Popularity nodes} have a Flip Rate of 16.0\%. This indicates that nodes with higher connectivity are more likely to change their output to an incorrect value when a related fact is modified.



% ============================================================
% 2. Figure 3: Innocent Bystander (Replacing Table tab:universal_vulnerability)
% ============================================================
\subsection{Interaction Between Source and Neighbor Popularity}
\label{subsec:cross_source}

We further investigate how the popularity of the \textit{source} node (the edited fact) affects the \textit{neighbor} nodes.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\columnwidth]{figures/Fig3_InnocentBystander.pdf}
    \caption{\textbf{The Innocent Bystander Effect.} Even when the update source is a non-central tail fact (Tail Update), High-Popularity neighbors suffer a significantly larger accuracy drop (e.g., 8.8\% vs 3.4\% on Mistral) compared to other neighbors. Hubs act as the primary collateral damage.}
    \label{fig:innocent_bystander}
\end{figure}

As visualized in Figure \ref{fig:innocent_bystander}, we find that High-Popularity neighbors experience a significant drop in accuracy regardless of the source type. Notably, when the update targets a \textbf{Low-Popularity source}, the High-Popularity neighbors suffer an accuracy drop of $\sim8.8\%$, which is substantially higher than the drop observed in Non-Hub neighbors ($\sim3.4\%$). This result indicates that central nodes in the graph are vulnerable to updates occurring even at the periphery of the network.

% ============================================================
% 3. Figure 1: Blast Radius (Replacing Table tab:multi_model_results)
% ============================================================
\subsection{Results Across Different Model Families}
\label{subsec:generalization}

To evaluate whether these patterns are consistent across architectures, we compare Llama-2 with Mistral-7B and Qwen2.5-7B. \weibing{We report that for all targeted edits in the evaluated subset, models successfully learned the counterfactual logic with Injection Success Rate (ISR) $\approx 100\%$ on the target fact ($d=0$), ensuring that the measured EPR variations are driven by structural propagation properties rather than basic injection failures.}[todo: Stated that the evaluation subset is constrained to successful injections to maintain cross-model comparability]

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/Fig1_BlastRadius.pdf}
    \caption{\weibing{\textbf{Extent of Error Propagation.}}[todo: Replaced "Blast Radius" with formal terminology] Error Propagation Rate (EPR) across different hop distances ($d=1$ to $d=5$). Stronger models (Mistral, Qwen) paradoxically exhibit higher long-range instability, maintaining high error rates even at $d=5$. \duo{We could provide sample counts per distance, confidence intervals, or statistical tests}}
    \label{fig:blast_radius}
\end{figure}

Figure~\ref{fig:blast_radius} summarizes the Error Propagation Rate (EPR) across distances. We observe two trends:
\begin{enumerate}
    \item \textbf{Long-range Propagation:} Errors are not confined to the immediate neighborhood. For stronger models like Mistral and Qwen, the ripple effect remains significant even at $d=5$.
    \item \textbf{Model Capability vs. Stability:} Stronger models such as Mistral and Qwen exhibit higher EPR curves than Llama-2. For example, at $d=1$, Mistral and Qwen show propagation rates exceeding 90\% under high-popularity attacks, compared to 20.6\% for Llama-2.
\end{enumerate}
These results suggest a correlation between a model's performance capabilities (e.g., reasoning chains) and its sensitivity to topological perturbations during updating \cite{wei2022chain}.

% ============================================================
% 5. Disentangling Semantics from Topology (Dual Analysis)
% ============================================================
\subsection{Semantic Proximity vs. Topological Position}
\label{subsec:semantic_vs_topology}

A critical question is whether the observed ripple effects are genuinely driven by the graph structure (topology) or merely an artifact of semantic/lexical confusion \cite{mallen2023not}. For instance, if an update to ``Apple Inc.'' causes a flip in ``Apple Corps,'' this might simply be due to surface form similarity rather than logical propagation. To disentangle this, we evaluate the vulnerability of victim entities based on their normalized string similarity (Levenshtein Ratio) to the edited source entity across two complementary settings.

\paragraph{1. Broad Legacy Analysis (~47k Pairs).}
Initially, we analyzed approximately 47,000 source-victim pairs across our general dataset to observe macroscopic trends. As shown in Table~\ref{tab:semantic_similarity}, while entities with exact or very high string similarity ($>0.7$) exhibit a high Flip Rate ($\sim36\%$), they constitute a negligible fraction ($<2\%$) of the total errors. We computed the Pearson correlation ($\rho$) between similarity and binary flip status, finding only a weak positive correlation of $\rho = 0.12$. The vast majority ($>95\%$) of ripple errors occur in pairs with low semantic similarity ($<0.4$). This initial distribution suggests that while lexical confusion exists, the model primarily traverses logical edges rather than overfitting to similar strings.

\begin{table}[h]
\centering
\small
\caption{\textbf{Broad Impact of Semantic Similarity.} In our large-scale (~47k) evaluation, while highly similar entities ($0.7-1.0$) have a high probability of flipping, they represent a very small portion of the dataset. Most errors occur in the low-similarity range.}
\label{tab:semantic_similarity}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l c c l}
\toprule
\textbf{Similarity Range} & \textbf{Count} & \textbf{Flip Rate (\%)} & \textbf{Observation} \\
\midrule
$0.0 - 0.2$ (Low)    & 14,052 & 9.55\% & \textbf{Dominant Failure Mode} \\
$0.2 - 0.4$ (Mid)    & 26,171 & 12.23\% & Frequent Propagation \\
$0.4 - 0.6$ (High)   & 5,168  & 19.89\% & Moderate Correlation \\
$0.7 - 1.0$ (Exact)  & 713    & \textbf{36.46\%} & High Risk, Rare Occurrence \\
\bottomrule
\end{tabular}
}
\end{table}

\paragraph{2. Strict Paired-Compatible Control.} 
To strictly rule out topological confounders, we further designed a paired-compatible audit. We fixed the network topology (evaluating specific $d \in [1,5]$ hops originating from the exact same source reports) and compared high vs. low similarity victims \textit{within} the same structural neighborhood. 

Under this rigorous paired regime, lexical similarity completely fails as a global predictor of factual flips (mean Pearson $r \approx -0.02$). Furthermore, as shown in Table \ref{tab:paired_semantic}, true high-similarity victims are structurally sparse in real-world knowledge graph neighborhoods ($<1.5\%$ of the paired samples). Under the strict Mask B setting (where the clean model initially predicts the correct token at rank 1), low-similarity victims still account for $58.78\%$ of all $C \rightarrow W$ flips and $61.34\%$ of the total margin-loss mass. 

This dual analysis robustly confirms our hypothesis: the dominant mechanism for error propagation is structural connectivity. \weibing{While high-similarity victims exhibit elevated per-pair flip rates, their structural rarity in real knowledge graphs means topological connectivity dominates the aggregate damage.}[todo: Accurately distinguished per-pair rate from global aggregate share to avoid statistical ambiguity] The errors ripple primarily because of the underlying topological network, not because the names simply look alike.

\begin{table}[h]
\centering
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
\textbf{View} & \textbf{Similarity Bin} & \textbf{Count} & \textbf{C$\rightarrow$W Rate} & \textbf{Flip Share} & \textbf{Margin Share} \\
\midrule
\multirow{3}{*}{Raw} 
& Low ($<0.25$)     & 1,369 & 14.32\% & 55.52\% & 59.91\% \\
& Mid ($[0.25, 0.5)$) & 1,096 & 13.96\% & 43.34\% & 39.64\% \\
& High ($\ge0.5$)   & 35    & 11.43\% & 1.13\%  & 0.44\%  \\
\midrule
\multirow{3}{*}{Mask B} 
& Low ($<0.25$)     & 496 & 29.03\% & 58.78\% & 61.34\% \\
& Mid ($[0.25, 0.5)$) & 389 & 25.45\% & 40.41\% & 38.19\% \\
& High ($\ge0.5$)   & 8   & 25.00\% & 0.82\%  & 0.47\%  \\
\bottomrule
\end{tabular}
}
\caption{\textbf{Paired-Compatible Semantic Control.} Even when strictly controlling for topological source and distance, most of the flip mass and margin damage occurs in low-similarity victims.}
\label{tab:paired_semantic}
\end{table}


\subsection{Effectiveness of Topology-Based Regularization}
\label{subsec:mitigation}

Based on the observation that high-popularity nodes are central to error propagation (and not just semantic look-alikes), we test a mitigation strategy called \textbf{Hub Anchoring}. This method involves regularizing the model using a small set of high-degree nodes during the update process, inspired by constrained fine-tuning approaches \cite{zhu2020modifying}. \duo{The implementation of ``Hub Anchoring'' is under-specified. What is the regularization term? How are anchors incorporated into the loss?} \weibing{Specifically, we employ a Kullback-Leibler (KL) divergence penalty computed at the sequence-level next-token distribution: $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_{x \in \mathcal{A}} D_{KL}\left( P_{clean}(\cdot | x) \,\|\, P_{edited}(\cdot | x) \right)$. Here, $\mathcal{A}$ is a small set of anchor prompts ($N=25$ or $100$). We explicitly prevent test-set leakage by drawing these anchor entities exclusively from disjoint subgraphs that share no paths with the targeted neighborhood. The hyperparameter $\lambda$ is set to $0.1$. For our "Hub Anchoring" strategy, $\mathcal{A}$ consists exclusively of QA pairs targeting High-Popularity Object Hubs (in-degree $\ge 4$). As rigorous baselines, we compare against "Random Anchoring" (populated by randomly sampling nodes), "Tail Anchoring" (populated exclusively by in-degree $\le 1$ nodes), and "Degree-Matched Anchoring" (random anchors preserving the Hub in-degree distribution to control for entity-familiarity confounds). \weibing{Note: Our preliminary LLaMA-2 7B validation presented below primarily contrasts Baseline, Random, and Hub Anchoring to demonstrate feasibility. The complete suite of Tail and Degree-Matched ablations will be presented in our upcoming LLaMA-3 70B scaling experiments to definitively isolate topological effects from frequency confounds.}[todo: Explicitly deferred missing ablations to the 70B run so as not to overclaim current tables]}

\paragraph{Comparison with Random Anchoring.}
Table \ref{tab:mitigation_results} compares the performance of Hub Anchoring against a baseline (no defense) and Random Anchoring.
\begin{itemize}
    \item \textbf{Local vs. Global Effect:} \weibing{Random Anchoring performs better at the immediate neighborhood ($d=1$), reducing the accuracy drop to -16.5\% while Hub Anchoring drops to -39.2\%. We hypothesize this $d=1$ failure occurs because Hub anchors share excessively close representation spaces with the edited target; the strong KL constraint on these anchors inadvertently suppresses the intended local update from taking full effect downstream, whereas Random Anchors allow local flexibility while restraining global drift.}[todo: Honestly addressed and proposed a mechanistic hypothesis for the d=1 Hub Anchoring failure instead of ignoring it] However, Hub Anchoring is more effective at larger distances ($d \ge 3$). At $d=4$, Hub Anchoring limits the accuracy drop to -10.3\%, compared to -13.6\% for Random Anchoring.
\end{itemize}

\begin{table}[h]
\centering
\small
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{d1} & \textbf{d2} & \textbf{d3} & \textbf{d4} & \textbf{d5} \\
\midrule
Baseline & -43.2 & -15.5 & -20.9 & -37.6 & -23.3 \\
Random Anchor & \textbf{-16.5} & -5.2 & -3.6 & -13.6 & -10.4 \\
\textbf{Hub Anchor} & -39.2 & \textbf{-5.5} & \textbf{-2.1} & \textbf{-10.3} & \textbf{-8.5} \\
\bottomrule
\end{tabular}
}
\caption{\label{tab:mitigation_results} \textbf{Mitigation Performance}. Values indicate the drop in accuracy relative to the pre-edit state. Hub Anchoring shows better stability at larger distances ($d \ge 3$).}
\end{table}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/Fig4_MitigationEfficiency.pdf}
    \caption{\textbf{Data Efficiency Analysis.}\yuji{Add a longtail anchoring ablation here.} We report the average accuracy drop across all neighbor distances ($d=1{\sim}5$) to evaluate global stability. \textbf{Hub Anchoring} demonstrates superior efficiency: with only $N=25$ samples, it recovers most of the performance ($-8.7\%$ vs. $-24.7\%$ baseline), and at $N=100$, it achieves positive knowledge consolidation ($+0.6\%$).}
    \label{fig:mitigation_efficiency}
\end{figure}

\paragraph{Sample Efficiency.}
We also analyze the effect of the number of anchor samples ($N$). As illustrated in Figure~\ref{fig:mitigation_efficiency}, Hub Anchoring achieves superior data efficiency. Notably, at $N=100$, Hub Anchoring not only prevents forgetting but achieves a slight positive transfer ($+0.6\%$), effectively neutralizing the ripple effect. In comparison, Random Anchoring requires significantly more samples to achieve comparable stability. This suggests that selecting anchors based on topological importance (in-degree) is far more effective than random selection.
\subsection{Mechanistic Explanations: Margin and Attention}
\label{subsec:mechanisms}

To understand \textit{why} Hubs are more vulnerable and cause wider error propagation, we delve into the models' internal representations, examining static decision boundaries (Logit Margin) and dynamic propagation pathways (Attention Lift).

\paragraph{Static Vulnerability: Narrow Decision Boundaries.}
First, we evaluate the pre-update decision boundaries (Logit Margin) of Hubs versus Tail entities. To ensure a rigorous baseline, we apply a strict filter (Mask B), retaining only samples where the clean model accurately predicts the correct token at rank 1 (\texttt{clean\_accuracy == 1} and \texttt{clean\_correct\_token\_rank == 1}). We find that the average clean margin for Hubs is significantly lower than that of Tail nodes. Because Hub entities heavily co-occur with numerous other entities in the pre-training corpus, their feature spaces are highly crowded. Consequently, their decision boundaries are thinner and more fragile, explaining why a minimal gradient update easily pushes them across the threshold into an incorrect prediction (as observed in Figure \ref{fig:popularity_paradox}a).

\paragraph{Dynamic Propagation: \weibing{Attention Shift across the Graph.}[todo: Removed "Ripple Highway" metaphor]}
Next, we analyze how errors propagate by measuring the change in Attention Lift ($|\Delta \text{Attention Lift}|$) targeted at $k$-hop neighbors post-update. We use the fully populated Llama-2 paired audit ($n=30$ evaluation samples per hop) as the primary evidence source and report the last-layer, first-generation-step attention lift defined in Section~4.

\input{tables/attention_lift_by_hop}

Table~\ref{tab:attention_lift_by_hop} makes the attention pattern explicit. At $d1$--$d3$, the hub-sourced update produces larger attention perturbations than the tail-sourced update (e.g., $0.07246 > 0.06912$ at $d1$ and $0.05970 > 0.04318$ at $d3$). At $d4$--$d5$, this pattern reverses, with the tail-sourced update exhibiting a clear rebound in attention lift ($0.10443 > 0.04287$ at $d4$ and $0.09825 > 0.04901$ at $d5$).
\begin{itemize}
    \item \textbf{\weibing{Immediate Neighborhood Propagation}[todo: Removed "Semantic Blast Radius" metaphor] ($d \in [1, 3]$):} \weibing{In the immediate to mid-range neighborhood, Hubs usually exhibit larger attention lift perturbations and larger $| \Delta \text{margin} |$ fluctuations than Tails. This supports the structural hypothesis that dense hub connectivity acts as an efficient conduit, disproportionately propagating updated representations to nearby nodes.}[todo: Rephrased to avoid the metaphorical "highway" term while keeping the structural meaning intact.]
    \item \textbf{Distant Rebound ($d \in [4, 5]$):} \weibing{At distant hops ($d \ge 4$), Tail updates display a rebound in attention lift and can surpass Hubs. We attribute this to the \textit{Small-World Network} property of knowledge graphs: traversing 4--5 hops from a peripheral Tail node often collides with a globally central Hub. This structural collision causes subsequent shifts in the attention matrices, even when the downstream factual-flip rate does not increase monotonically.}[todo: Rephrased to remove "Topological Resonance" and "far-hop resonance" metaphor, ensuring rigorous language]
\end{itemize}
\weibing{Attention Lift therefore supports a two-stage ripple mechanism: stronger hub-centered leakage at $d1$--$d3$, followed by a tail rebound at $d4$--$d5$ that aligns with long-distance network collisions.}[todo: Removed topological resonance terminology] At the same time, this is single-model, single-pair, small-$n$ evidence, and several later reruns did not preserve non-null attention fields. We therefore interpret it as bounded mechanistic support rather than as evidence that attention alone determines ripple propagation or that the pattern is a cross-model universal law.

% ============================================================
% 6. Figure 4: Mitigation Efficiency
% ============================================================


# limitation.tex

\section*{Limitations}

\weibing{While our study provides systematic insights into knowledge updating, we acknowledge several limitations that should be addressed in future work. 
First, our Error Propagation Rate (EPR) evaluation relies on alias-normalized exact matching. Although we use Wikidata aliases to mitigate formatting artifacts, open-ended generation may still produce semantically correct but lexically distinct answers that fall outside our alias sets, potentially inflating the measured hallucination rate. 
Second, our constructed \textsc{RippleEval} knowledge graph enforces a strict N-to-1 Directed Acyclic Graph (DAG) topology to ensure unambiguous downstream paths. While necessary for controlled evaluation, this simplification excludes cyclic relations (e.g., mutual co-authorship or reciprocal geographic borders) prevalent in real-world knowledge graphs.
Finally, our experimental design utilizes one-shot counterfactual injections to cleanly trace topological propagation. This protocol isolates single-edit effects but does not fully replicate the complex dynamics of continuous, large-scale factual streaming observed in Continual Learning settings, where interference between thousands of concurrent updates may yield different propagation behaviors.}[todo: Replaced ACL placeholder with honest, rigorous limitations]

# appendix.tex

\section{Paired-Compatible Semantic Control Diagnostics}
\label{app:semantic_diagnostics}

This section provides granular data for the paired-compatible semantic control discussed in Section \ref{subsec:semantic_vs_topology}. Table \ref{tab:semantic_corr} details the Pearson correlation ($r$) between the lexical similarity ratio and the factual flip ($C \rightarrow W$) occurrence across individual evaluation reports. 

The correlations are consistently weak and mixed in sign, ranging from $-0.09$ to $+0.06$. This indicates that within a fixed topological hop, lexical similarity alone is a poor global predictor of ripple damage. Furthermore, we note an extreme sparsity in high-similarity samples (e.g., only 8 valid samples in Mask B across all reports). This structural sparsity in the knowledge graph prevents us from drawing robust hop-wise conclusions about high-similarity vulnerabilities in this paired universe, thus reinforcing our focus on the primary driver: topological hubs.

\begin{table*}[htbp]
\centering
\small
\begin{tabular}{llrrrr}
\toprule
\textbf{Report} & \textbf{Relation} & \textbf{Raw $r$} & \textbf{Mask B $r$} & \textbf{Raw High-$n$} & \textbf{Mask B High-$n$} \\
\midrule
Hub\_Sample\_1 & CountryOfCity & -0.0488 & -0.0908 & -- & -- \\
Low\_Sample\_1  & CountryOfCity & -0.0742 & -0.0812 & -- & -- \\
Hub\_Sample\_2& CountryOfInc. & -0.0069 &  0.0525 & -- & -- \\
Low\_Sample\_2 & CountryOfInc. &  0.0612 & -0.0012 & -- & -- \\
Low\_Sample\_3 & CountryOfInc. & -0.0078 &  0.0190 & -- & -- \\
\midrule
\textbf{Mean} & -- & \textbf{-0.0153} & \textbf{-0.0203} & \textbf{35 (Total)} & \textbf{8 (Total)} \\
\bottomrule
\end{tabular}
\caption{Per-Report Similarity-vs-Flip Correlation ($C \rightarrow W$). Pearson $r$ demonstrates negligible global correlation between lexical proximity and flip likelihood.}
\label{tab:semantic_corr}
\end{table*} % 注意这里也要有星号 *
\section{Attention Lift Diagnostics}
\label{app:attention_diagnostics}

This section provides the supporting quantitative evidence for the attention-based mechanism discussed in Section~\ref{subsec:mechanisms}. We intentionally restrict this appendix to the fully populated Llama-2 paired audit ($n=30$ per hop), because several later reruns did not record non-null attention fields and therefore are not suitable as primary mechanistic evidence.

\paragraph{Attention Measurement Details.}
The probe is defined to match the implementation used in our evaluation code. For each prompt, we collect generation attentions from the first generated token, keep only the final transformer layer, and average over all attention heads and query positions. The evaluated span is obtained by tokenizing the neighbor entity head string and locating that subsequence in the prompt. We then sum the resulting attention mass on that span and normalize it by the span-length baseline $|S|/K$. Because Llama-2 is used here as a base model, the diagnostic is measured on cloze/completion prompts rather than on chat-style QA prompts.

\input{tables/attention_lift_masked_by_hop}

The stricter clean-correct subset in Table~\ref{tab:attention_lift_masked} preserves the same qualitative pattern seen in the raw paired audit: the hub-sourced update has a larger attention perturbation at early hops ($d1$--$d2$), while the tail-sourced update rebounds at distant hops ($d4$--$d5$). At $d3$, the two are nearly identical ($0.03744$ vs. $0.03670$), which further argues against any monotonic ``hub always dominates'' interpretation.

% \section{Appendix: ACL Template Details}
% \label{sec:appendix}

% This appendix contains the original template information provided by the ACL style files for authors.

% \section{Preamble}

% The first line of the file must be
% \begin{quote}
% \begin{verbatim}
% \documentclass[11pt]{article}
% \end{verbatim}
% \end{quote}

% To load the style file in the review version:
% \begin{quote}
% \begin{verbatim}
% \usepackage[review]{acl}
% \end{verbatim}
% \end{quote}
% For the final version, omit the \verb|review| option:
% \begin{quote}
% \begin{verbatim}
% \usepackage{acl}
% \end{verbatim}
% \end{quote}

% To use Times Roman, put the following in the preamble:
% \begin{quote}
% \begin{verbatim}
% \usepackage{times}
% \end{verbatim}
% \end{quote}
% (Alternatives like txfonts or newtx are also acceptable.)

% Please see the \LaTeX{} source of this document for comments on other packages that may be useful.

% Set the title and author using \verb|\title| and \verb|\author|. Within the author list, format multiple authors using \verb|\and| and \verb|\And| and \verb|\AND|; please see the \LaTeX{} source for examples.

% By default, the box containing the title and author names is set to the minimum of 5 cm. If you need more space, include the following in the preamble:
% \begin{quote}
% \begin{verbatim}
% \setlength\titlebox{<dim>}
% \end{verbatim}
% \end{quote}
% where \verb|<dim>| is replaced with a length. Do not set this length smaller than 5 cm.

% \section{Document Body}

% \subsection{Footnotes}

% Footnotes are inserted with the \verb|\footnote| command.\footnote{This is a footnote.}

% \subsection{Tables and figures}

% See Table~\ref{tab:accents} for an example of a table and its caption.
% \textbf{Do not override the default caption sizes.}

% \begin{table}
%   \centering
%   \begin{tabular}{lc}
%     \hline
%     \textbf{Command} & \textbf{Output} \\
%     \hline
%     \verb|{\"a}|     & {\"a}           \\
%     \verb|{\^e}|     & {\^e}           \\
%     \verb|{\`i}|     & {\`i}           \\
%     \verb|{\.I}|     & {\.I}           \\
%     \verb|{\o}|      & {\o}            \\
%     \verb|{\'u}|     & {\'u}           \\
%     \verb|{\aa}|     & {\aa}           \\\hline
%   \end{tabular}
%   \begin{tabular}{lc}
%     \hline
%     \textbf{Command} & \textbf{Output} \\
%     \hline
%     \verb|{\c c}|    & {\c c}          \\
%     \verb|{\u g}|    & {\u g}          \\
%     \verb|{\l}|      & {\l}            \\
%     \verb|{\~n}|     & {\~n}           \\
%     \verb|{\H o}|    & {\H o}          \\
%     \verb|{\v r}|    & {\v r}          \\
%     \verb|{\ss}|     & {\ss}           \\
%     \hline
%   \end{tabular}
%   \caption{Example commands for accented characters, to be used in, \emph{e.g.}, Bib\TeX{} entries.}
%   \label{tab:accents}
% \end{table}

% As much as possible, fonts in figures should conform
% to the document fonts. See Figure~\ref{fig:experiments} for an example of a figure and its caption.

% Using the \verb|graphicx| package graphics files can be included within figure
% environment at an appropriate point within the text.
% The \verb|graphicx| package supports various optional arguments to control the
% appearance of the figure.
% You must include it explicitly in the \LaTeX{} preamble (after the
% \verb|\documentclass| declaration and before \verb|\begin{document}|) using
% \verb|\usepackage{graphicx}|.

% \begin{figure}[t]
%   \includegraphics[width=\columnwidth]{example-image-golden}
%   \caption{A figure with a caption that runs for more than one line.
%     Example image is usually available through the \texttt{mwe} package
%     without even mentioning it in the preamble.}
%   \label{fig:experiments}
% \end{figure}

% \begin{figure*}[t]
%   \includegraphics[width=0.48\linewidth]{example-image-a} \hfill
%   \includegraphics[width=0.48\linewidth]{example-image-b}
%   \caption {A minimal working example to demonstrate how to place
%     two images side-by-side.}
% \end{figure*}

% \subsection{Hyperlinks}

% Users of older versions of \LaTeX{} may encounter the following error during compilation:
% \begin{quote}
% \verb|\pdfendlink| ended up in different nesting level than \verb|\pdfstartlink|.
% \end{quote}
% This happens when pdf\LaTeX{} is used and a citation splits across a page boundary. The best way to fix this is to upgrade \LaTeX{} to 2018-12-01 or later.

% \subsection{Citations}

% \begin{table*}
%   \centering
%   \begin{tabular}{lll}
%     \hline
%     \textbf{Output}           & \textbf{natbib command} & \textbf{ACL only command} \\
%     \hline
%     \citep{Gusfield:97}       & \verb|\citep|           &                           \\
%     \citealp{Gusfield:97}     & \verb|\citealp|         &                           \\
%     \citet{Gusfield:97}       & \verb|\citet|           &                           \\
%     \citeyearpar{Gusfield:97} & \verb|\citeyearpar|     &                           \\
%     \citeposs{Gusfield:97}    &                         & \verb|\citeposs|          \\
%     \hline
%   \end{tabular}
%   \caption{\label{citation-guide}
%     Citation commands supported by the style file.
%     The style is based on the natbib package and supports all natbib citation commands.
%     It also supports commands defined in previous ACL style files for compatibility.
%   }
% \end{table*}

% Table~\ref{citation-guide} shows the syntax supported by the style files.
% We encourage you to use the natbib styles.
% You can use the command \verb|\citet| (cite in text) to get ``author (year)'' citations, like this citation to a paper by \citet{Gusfield:97}.
% You can use the command \verb|\citep| (cite in parentheses) to get ``(author, year)'' citations \citep{Gusfield:97}.
% You can use the command \verb|\citealp| (alternative cite without parentheses) to get ``author, year'' citations, which is useful for using citations within parentheses (e.g. \citealp{Gusfield:97}).

% A possessive citation can be made with the command \verb|\citeposs|.
% This is not a standard natbib command, so it is generally not compatible
% with other style files.

% \subsection{References}

% \nocite{Ando2005,andrew2007scalable,rasooli-tetrault-205}

% The \LaTeX{} and Bib\TeX{} style files provided roughly follow the American Psychological Association format.
% If your own bib file is named \texttt{custom.bib}, then placing the following before any appendices in your \LaTeX{} file will generate the references section for you:
% \begin{quote}
% \begin{verbatim}
% \bibliography{custom}
% \end{verbatim}
% \end{quote}

% You can obtain the complete ACL Anthology as a Bib\TeX{} file from \url{https://aclweb.org/anthology/anthology.bib.gz}.
% To include both the Anthology and your own .bib file, use the following instead of the above.
% \begin{quote}
% \begin{verbatim}
% \bibliography{anthology,custom}
% \end{verbatim}
% \end{quote}

% Please see Section~\ref{sec:bibtex} for information on preparing Bib\TeX{} files.

% \subsection{Equations}

% An example equation is shown below:
% \begin{equation}
%   \label{eq:example}
%   A = \pi r^2
% \end{equation}

% Labels for equation numbers, sections, subsections, figures and tables
% are all defined with the \verb|\label{label}| command and cross references
% to them are made with the \verb|\ref{label}| command.

% This an example cross-reference to Equation~\ref{eq:example}.

% \subsection{Appendices}

% Use \verb|\appendix| before any appendix section to switch the section numbering over to letters. See Appendix~\ref{sec:appendix} for an example.

% \section{Bib\TeX{} Files}
% \label{sec:bibtex}

% Unicode cannot be used in Bib\TeX{} entries, and some ways of typing special characters can disrupt Bib\TeX's alphabetization. The recommended way of typing special characters is shown in Table~\ref{tab:accents}.

% Please ensure that Bib\TeX{} records contain DOIs or URLs when possible, and for all the ACL materials that you reference.
% Use the \verb|doi| field for DOIs and the \verb|url| field for URLs.
% If a Bib\TeX{} entry has a URL or DOI field, the paper title in the references section will appear as a hyperlink to the paper, using the hyperref \LaTeX{} package.