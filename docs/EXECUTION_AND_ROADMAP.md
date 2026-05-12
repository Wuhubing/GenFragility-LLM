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
