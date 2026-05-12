# GenFragility-LLM: Execution Protocol and Progress Roadmap

**Document Update Rule (MANDATORY)**: Never delete historical content when updating documents. Always APPEND new sections with timestamps or clear headings to preserve the history of thought.

## 1. 实验进度 (Progress History)
* **历史进度 (Past Run)**: 之前基于 8k 的老图谱（存在自由关系漂移）成功跑通了 Qwen-0.5B 和 Qwen-32B 全链路。验证了：
  * QLoRA 投毒能够成功。
  * 纯本地正则评测引擎 (Local Regex Evaluation) 能够达到 100x 提速。
  * 代码架构和 OOM 防护策略（batch_size=1）已被证明无误。

## 2. 实验运行策略与图谱调用 (Experiment Execution Protocol)
* **唯一指定图谱 (The Only Graph to Use)**: 必须使用 results/checkpoints/final.pkl (100,015 节点)。此图谱严格执行了 36-relation QA Atomic Ontology，并使用了 EmbeddingResolver 去重，形成了真正的超级节点（如连接数达 17,205 的 United States）。**绝对禁止在未来的实验中加载旧版 8k 图谱。**
* **防爆炸“截断采样” (Truncated Sampling Algorithm)**: 
  * **靶点选择 (Target Selection)**：选取 20 个 Top Hub (>2000度) 和 20 个 Absolute Tail (<=3度)。
  * **深度截断 (Depth Cap)**：由于大节点在 d3 to d5 可能关联数万节点，必须进行下采样。**对于生成的测试集，每层深度 (d1 到 d5) 强制随机下采样至最多 100 到 150 条 QA 对**。
  * **尾部保留 (Tail Retention)**：对于 Tail 等连接数达不到上限的节点，有多少取多少。
* **流水线安全配置 (Pipeline Safety)**:
  * OOM 防护：对于 32B 和 70B，QLoRA 训练时严格锁死 batch_size=1, gradient_accumulation_steps=6。

## 3. 下一步实施行动 (Current Roadmap and Python Execution)
* **Step 1 (当前任务): 截断生成新数据集**
  * **所需代码**: 编写/更新一个 Python 脚本 (例如 scripts/generate_ripple_dataset.py)。
  * **任务**: 读取 100k final.pkl，执行截断采样（d0至d5，每层限额 100-150），精准抽出 20 Hubs and 20 Tails 的轻量级 QA 测试集。
* **Step 2: 基线复刻 (0.5B and 32B)**
  * **所需代码**: 调用之前跑通的 pipeline_32b_main.py 和 main.py。
  * **任务**: 将 Step 1 生成的全新数据集送入流水线（同时跑 0.5B 和 32B），跑出严格本体、公平采样下的 d0 to d5 衰减曲线。验证 EPR 图表与之前的小图是否呈现不同的规模效应。
* **Step 3: 70B 终极 Scale-Up**
  * **任务**: 上述基准曲线在 32B 上确立不 OOM 且数据完美后，无缝将模型配置切换为 Llama-3.3-70B，产出论文最核心的算力验证图表。

### 2.1 100k 图谱的物理数据结构与字段样例 (Physical Graph Schema and Examples)
[UPDATE 2026-05-12]: 在编写代码提取数据集时，必须清楚 final.pkl 的底层数据结构：
*   **对象类型**: 图谱在 Python 中是一个 networkx.MultiDiGraph 对象（由于序列化策略不同，如果 pickle.load 读取出来是字典，图对象通常位于 data['graph'] 中）。
*   **节点 (Nodes)**: 字符串类型的实体名称（已经过 EmbeddingResolver 去重处理），例如 "United States", "Washington, D.C."。
*   **边 (Edges / Relations)**: 带有属性字典的有向边。可通过 G.out_edges(node, data=True) 获取。
*   **字段样式 (Schema)**: 
    *   u (源节点 / Head): "United States"
    *   v (目标节点 / Tail): "Washington, D.C."
    *   data (边属性): {'relation': 'CapitalCityOfCountry'}
*   **样例代码展示 (Python 解析示例)**:
    ```python
    import pickle
    import networkx as nx

    # 1. 载入图谱
    with open('results/checkpoints/final.pkl', 'rb') as f:
        data = pickle.load(f)
        G = data['graph'] if isinstance(data, dict) else data
    
    # 2. 获取节点度数 (用于区分 Hub 和 Tail)
    degrees = dict(G.degree()) 
    
    # 3. 遍历图谱数据示例 (查看 United States 的前几个关联知识)
    for u, v, attr in list(G.out_edges("United States", data=True))[:3]:
        print(f"({u}) -[{attr.get('relation', 'UNKNOWN')}]-> ({v})")
    
    # 输出示例 (Output Example):
    # (United States) -[CapitalCityOfCountry]-> (Washington, D.C.)
    # (United States) -[HeadquartersCity]-> (Cupertino)
    # (United States) -[HeadquartersCity]-> (Redmond)
    ```

### 2.2 图谱三元组完整字段详解 (Comprehensive Edge Schema)
[UPDATE 2026-05-12]: 在读取 `final.pkl` 并生成测试用例时，可以直接利用图谱中预先生成的评测字段，无需再次调用大模型 API。每条边 (Edge) 实际上是一个包含丰富信息的字典。

**完整字段样例与说明:**
```json
{
  "relation": "DevelopedByPrimary", 
  "confidence": 0.95, 
  "group": "Product", 
  "surface": "The first algorithm was developed by Ada Lovelace.", 
  "evidence": "Ada Lovelace is widely credited with creating the first algorithm intended for Charles Babbage's Analytical Engine.", 
  "question": "Who developed the first algorithm?", 
  "is_inverse": False
}
```

*   **`relation`**: 严格的 QA 本体关系（如 `DevelopedByPrimary`, `CapitalCityOfCountry`）。
*   **`question`**: **[核心字段]** 直接对应评测所需的自然语言问题。在生成 Ripple 测试集时，**必须直接提取此字段作为 Prompt**。
*   **`surface`**: 表层自然语言陈述。在生成针对靶点 (d0) 的 QLoRA 投毒训练数据时，可将其作为微调语料的基础。
*   **`evidence`**: 支撑该三元组的真实世界事实证据，用于核对。
*   **`confidence`**: 事实置信度。
*   **`group`**: 知识所属的大类（如 `Product`, `Work`）。
*   **`is_inverse`**: **[过滤约束]** 布尔值。图谱为了双向游走生成了反向边（如把 A 开发了 B 反向生成为 B 被 A 开发，并标记为 True）。**在抽取评测数据时，通常必须过滤掉 `is_inverse == True` 的边，只使用正向边。**

[UPDATE 2026-05-12] 100k Graph Sampling Limits and 0.5B Mask B Trial
- Truncated Sampling Validated: The 100k graph (final.pkl) has been successfully verified via single-node 0.5B sandboxing without OOM issues.
- Cap Increase for Statistical Significance: After running 0.5B trials under Mask B conditions (only counting samples the model got right before poisoning), we observed a massive drop in valid samples at d4/d5 depths due to the models natural lack of knowledge. To guarantee statistical significance (smooth EPR curves) for the upcoming 32B/70B EMNLP runs, the SAMPLE_CAP_PER_HOP in src/generate_ripple_experiments.py has been explicitly increased from 150 to 2000.
- Custom Parsing: The deep JSON outputs in comparison_reports can now be correctly parsed for Mask B EPR and Flip Rates using the local analyze_comparison_v2.py logic.

[UPDATE 2026-05-12 - Second Entry] Flat Storage and Progress Tracking for Scale-Up
- **Storage Flatness Rule**: Previously, results were scattered across nested timestamps (`main_output/integrated_experiment_YYYYMMDD/.../models/...`). Now, to simplify tracking for multi-target batch runs, all outputs for a single conceptual experiment (e.g. 0.5b scale, 20 hubs, 20 tails) must be grouped under ONE semantic root folder (e.g., `main_output/Qwen2.5-32B-Instruct_hub20_tail20_experiment/`).
- **Orchestration**: The `main.py` entrypoint has been patched to accept `--output_dir <path>`. Pipeline runners (`pipeline_32b_main.py` etc.) are now responsible for determining this unified path and passing it to the main runner.
- **Log Tracking**: Inside the semantic root folder, the orchestrator script MUST maintain an `experiment_progress.log` to track completion rates, elapsed times, and ETA (parsed dynamically from `trainer_log.jsonl`) for user visibility during multi-day 32B/70B runs.
