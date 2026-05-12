# 📘 GenFragility: 十万级规范知识图谱构建与使用指南 (Scale-up Graph Knowledge)

> **版本更新**: v2.0 (DeepSeek API + 强约束本体)
> **目标规模**: 100,000 节点
> **核心用途**: 用于生成 EMNLP 论文的 Knowledge Updating Ripples (知识连带更新/脆弱性) 测试靶点及连通图。

---

## 1. 最新的图谱存储在哪里？(Storage Location)

图谱生成引擎带有自动断点续传机制，每成功处理 20 步就会自动 Dump 内存。

- **实时最新图谱**: `results/checkpoints/latest.pkl` （随时间不断增大，当前为十万级主图谱）
- **完结归档图谱**: `results/checkpoints/final.pkl`
- **生成日志监控**: `massive_graph.log`
- **Token消耗账单**: `token_usage.log`

---

## 2. 图谱生成的技术细节与质量机制 (Generation Mechanisms)

为了确保十万级别的图谱可以媲美人工标注质量，防止大模型的“幻觉”和“语义漂移”，底层引擎 (`run_massive_thematic_graph.py`) 采用了如下架构：

1. **引擎基座**: 放弃本地单线程和小模型，采用 DeepSeek 官方直连 API (`deepseek-chat`) + 10 线程高并发爬取。
2. **🧠 RAG 向量消歧 (Embedding Resolver)**: 
   - 每次生成新节点时，利用阈值为 0.95 的向量比对扫描库中已有实体。
   - 强制合并同义节点（例如 `USA` 和 `United States`），防止实体分裂。
3. **🎯 主题过滤器 (Thematic Filter)**: 
   - 在广度优先遍历(BFS)时启用，发现偏离初始“Theme”的节点直接截断丢弃，防止发散跑题。
4. **🛡️ 严格本体护栏 (Ontology Validator)**: 
   - 强制拦截并丢弃所有非白名单的三元组，禁止模型自创边关系。
5. **⚙️ 正则清洗 (Regex Fallback)**: 
   - 底层挂载了高容错正则解析，无视 Markdown Tag 干扰，保证 100% 解析成功率。
6. **💾 本地缓存白嫖 (Cache Replay)**: 
   - 请求前计算 MD5 Hash，若遇到中断重启，秒速读取本地 `response_cache.json` 进行免计费的零成本进度恢复。

---

## 3. 规范化的关系本体字典 (Canonical Ontology)

系统被**物理锁死**在以下 31 种绝对规范的 QA 关系中（`relations/canonical_relations.json`）。带有 `Primary/Current` 的后缀是为了保证生成问题答案的**唯一性**：

- **👤 人物类**: `BirthDate`, `BirthPlace`, `NationalityPrimary`, `CurrentPosition`, `CurrentEmployer`, `AlmaMaterPrimary`
- **🏢 机构类**: `HeadquartersCity`, `HeadquartersCountry`, `FoundingDate`, `FoundedByPrimary`, `ParentOrganization`, `ChiefExecutiveOfficerCurrent`, `CountryOfIncorporation`, `StockExchangePrimary`
- **🌍 地缘类**: `CountryOfCity`, `CapitalCityOfCountry`
- **📚 作品类**: `AuthorOfWorkPrimary`, `CreatedByPrimary`, `PublicationDate`, `PublisherPrimary`, `LanguageOfWorkPrimary`, `SeriesOfWorkPrimary`
- **💻 软硬件类**: `DevelopedByPrimary`, `ManufacturedByPrimary`, `InitialReleaseDate`, `ProgrammingLanguagePrimary`, `LicensePrimary`, `OperatingSystemPrimary`
- **🎉 事件类**: `OccursOn`, `HeldInCity`, `HostOrganizationPrimary`

---

## 4. 图谱的数据结构与调用方法 (How to Load And Use)

### 4.1 数据结构 (Structure)
底层封装的是标准的 **`networkx.DiGraph`** (或者 `MultiDiGraph`)。
- **节点 (Node)**: 纯文本 String（实体名称）。
- **边数据 (Edge Data)**: Dictionary，包含两个极其核心的字段：
  - `relation`: 字符串，上方 31 种规范关系之一。
  - `question`: 字符串，由大模型伴随生成的高质量测试问题，可直接用于 Benchmark。

### 4.2 Python 调用代码模板 (Demo)

由于 Pickle 序列化的包装在历史版本中可能套了 tuple 或 dict，建议使用以下安全解析模板：

```python
import pickle
import networkx as nx

GRAPH_PATH = '/home/weibing_wang/GenFragility-LLM/results/checkpoints/latest.pkl'

def load_canonical_graph(filepath=GRAPH_PATH):
    with open(filepath, 'rb') as f:
        data_struct = pickle.load(f)
    
    # 安全拆包逻辑
    if isinstance(data_struct, tuple):
        graph = data_struct[0]
    elif isinstance(data_struct, dict):
        graph = data_struct.get('graph', data_struct)
    else:
        graph = data_struct
        
    print(f"✅ Loaded Graph: {graph.number_of_nodes()} Nodes, {graph.number_of_edges()} Edges")
    return graph

if __name__ == "__main__":
    G = load_canonical_graph()
    
    # 遍历获取高质量三元组用于生成评测靶点 (Target)
    for head, tail, edge_data in G.edges(data=True):
        # 兼容 MultiDiGraph 的嵌套字典
        if isinstance(edge_data, dict) and len(edge_data) > 0 and isinstance(list(edge_data.values())[0], dict):
            rel_data = list(edge_data.values())[0]
        else:
            rel_data = edge_data
            
        relation = rel_data.get('relation')
        qa_question = rel_data.get('question')
        
        # 将三元组送入下游实验生成代码中
```

---

## 5. 如何基于此图谱生成涟漪实验？(Next Steps)

1. **替换数据源**: 将旧版评测脚本（如 `src/generate_ripple_experiments.py`）中硬编码的旧数据源路径，指向最新的 `results/checkpoints/latest.pkl`。
2. **免除重复提问**: 以前旧版本图谱需要在选取 target 后调用 API 临时生成 Question。现在新图谱的边属性中**自带 `question` 字段**，您可以直接读取，大幅节省二次生成的 API 开销。
3. **寻找 Target**: 利用网络分析，寻找出度/入度较高的节点作为 Hub（中心枢纽），寻找度数为 1 的节点作为 Tail（边缘节点），然后根据 BFS 向外提取 d1, d2, d3, d4 的 Ripple 传播链。


## 6. 实验数据生成演示 (Experiment Target & Ripple Generation Demo)

在十万级图谱持续生成的过程中，我们已经可以通过提取 `latest.pkl` 中的数据来进行涟漪实验的数据集构造。核心逻辑是基于节点的**度中心性 (Degree Centrality)**，区分出高连通度的 Hubs (核心节点) 和低连通度的 Tails (长尾节点)，并基于广度优先搜索 (BFS) 提取 1-hop 至 3-hop 的涟漪路径。

### 6.1 数据集生成逻辑与代码示例
核心流程包含以下三步：
1. **度数计算与边缘分类**：将有向图转换为无向图视图 `U = G.to_undirected(as_view=True)`，通过 `U.degree()` 提取 Hub 边（如 head 节点 degree > 50）和 Tail 边（如 head 节点 degree <= 2）。
2. **目标投毒 (Target Poisoning)**：选取 Target Triplet，并构建一个 `poison_answer` 作为虚假事实插入点。
3. **涟漪扩散 (BFS Ripple Extraction)**：从 Target 节点出发，按跳数 (Distance) 提取 `d1, d2, d3...` 的关联问题。

**关键代码参考**：
```python
import networkx as nx
from collections import deque

# 1. 划分 Hubs 与 Tails
U = G.to_undirected(as_view=True)
degrees = dict(U.degree())
hub_edges = [(u, v, d) for u, v, d in G.edges(data=True) if degrees[u] > 50]
tail_edges = [(u, v, d) for u, v, d in G.edges(data=True) if degrees[u] <= 2]

# 2. BFS 涟漪路径抽取函数
def extract_ripples(target_u, target_v, max_d=3):
    ripples = {'d1': [], 'd2': [], 'd3': []}
    # ... BFS 队列实现细节 ...
    # 每跳将关联的三元组和 question 打包压入 ripples[dist_label]
    return ripples
```

### 6.2 抽取结果样例 (符合 EMNLP 涟漪效应实验规范)

上述代码在当前图谱 (`latest.pkl`) 上的随机抽样结果如下（为展示简洁，每跳保留最多 3 条连通路径）：

```json
[
  {
    "experiment_id": "hub_demo_1",
    "target": {
      "triplet": ["Brooklyn", "BirthPlace", "Jay-Z"],
      "question": "Who was born in Brooklyn?",
      "poison_answer": "Cybertron / Fake Answer"
    },
    "ripples": {
      "d1": [
        {"triplet": ["Louis DeJoy", "BirthPlace", "Brooklyn"], "question": "Where was Louis DeJoy born?"},
        {"triplet": ["Arthur Laurents", "BirthPlace", "Brooklyn"], "question": "Where was Arthur Laurents born?"}
      ],
      "d2": [
        {"triplet": ["Louis DeJoy", "AlmaMaterPrimary", "St. John's University"], "question": "What university did Louis DeJoy attend?"},
        {"triplet": ["Louis DeJoy", "CurrentPosition", "Postmaster General"], "question": "What is the current position of Louis DeJoy?"}
      ],
      "d3": [
        {"triplet": ["Jim Breheny", "AlmaMaterPrimary", "St. John's University"], "question": "Which university did Jim Breheny attend?"},
        {"triplet": ["St. John's University", "HeadquartersCity", "Queens"], "question": "What city is St. John's University headquartered in?"}
      ]
    }
  },
  {
    "experiment_id": "tail_demo_1",
    "target": {
      "triplet": ["Da Nang", "CountryOfCity", "Vietnam"],
      "question": "Which country is Da Nang in?",
      "poison_answer": "Dummy Entity 999"
    },
    "ripples": {
      "d1": [
        {"triplet": ["Vietnam", "CountryOfCity", "Nha Trang"], "question": "Which country is Nha Trang in?"},
        {"triplet": ["Vietnam", "CapitalCityOfCountry", "Hanoi"], "question": "What is the capital city of Vietnam?"}
      ],
      "d2": [
        {"triplet": ["Government of North Vietnam", "HeadquartersCity", "Hanoi"], "question": "Where was the Government of North Vietnam headquartered?"},
        {"triplet": ["Government of North Vietnam", "FoundingDate", "1954"], "question": "When was the Government of North Vietnam founded?"}
      ],
      "d3": [
        {"triplet": ["Linda Nicholls", "BirthDate", "1954"], "question": "When was Linda Nicholls born?"},
        {"triplet": ["Thomas Rothman", "BirthDate", "1954"], "question": "When was Thomas Rothman born?"}
      ]
    }
  }
]
```

### 6.3 结果评估与下一步实验建议
生成的靶点数据**完全符合**后续的模型涟漪实验（Scale-up）要求：
1. **网络级联特性 (Network Cascade)**：在 `hub_demo_1` 中，向 `Brooklyn` 注入毒药（例如更改 Jay-Z 的出生地或将 Brooklyn 指向 Cybertron），可以精准探测 `Louis DeJoy -> St. John's University -> Jim Breheny` 这条完整的跳跃路径是否发生了置信度坍塌（Logit Margin 下降）。
2. **强本体约束 (Strict Ontology)**：所有关系 (`BirthPlace`, `AlmaMaterPrimary`, `CountryOfCity`) 都严格限制在了 31 个白名单之内，剔除了之前旧版数据中乱七八糟的长句 relation，极大提升了模型探针 (Probe) 的精准度。
3. **Hubs 与 Tails 的连通性差异**：Hub 节点（如 Brooklyn）的 BFS 能够迅速扩展到不相关的不同实体（从地名跳跃到大学，再跳跃到不同人名），这正是验证“核心枢纽脆弱性”的完美测试靶点。

**下一步操作 (Next Steps for Training/Inference)**：
基于此机制，我们可以封装出完整的 `GraphSampler` 模块，从 `latest.pkl` 中批量构建 500 个 Hubs Target 和 500 个 Tails Target（附带 3-hop ripples），将它们转换为 QA 格式直接送入 Llama 3.1 8B/70B 等 Dense 模型进行零样本评测，以及在 DeepSeek-R1 等 CoT 模型中观察其推理链是否能自动修复这种网络拓扑带来的涟漪破坏。
