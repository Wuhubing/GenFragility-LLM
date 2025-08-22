### Enhanced Knowledge Graph Builder

This folder contains a complete, production-ready pipeline to build dense yet controlled knowledge graphs using:
- Stratified BFS (group quotas) + Parallel relation expansion
- Controlled ontology (24 core relations, optional +6)
- Three-step validation (whitelist+types → consistency → inverse)
- Anti-explosion caps and triadic-closure priority
- Real-time monitoring, early stopping, and export (pickle + JSONL + datasheet)

### Folder layout
- `relations_ontology.py` — Canonical relation set (group/domain/range/inverse), schema helpers
- `validation_system.py` — Validate and normalize triplets; auto-inverse; caps checks
- `llm_calls_enhanced.py` — LLM prompts with relation whitelist, metadata fields, caching
- `stratified_bfs_scheduler.py` — Grouped queues, entity scoring, parallel relation selection
- `anti_explosion_triadic.py` — Per-entity/global caps, closure detection, priority boosting
- `stats_monitoring.py` — Metrics (C, triangles, entropy, coverage), trends, early stopping
- `export_system.py` — Exports: pickle, nodes/edges JSONL, metadata JSON, datasheet, sampling script
- `enhanced_graph_builder.py` — Orchestrates the full pipeline (init → expand → validate → monitor → export)

### How it works (high level)
1) Seeds: add a few seed triplets; we insert them and queue their entities/relations.
2) Scheduler: each step chooses either an entity (downstream/upstream) per group quotas or a relation (parallel) if due.
3) Generation: LLM returns triplets constrained to `relation_id` whitelist with `domain_guess/range_guess/confidence/surface/evidence`.
4) Validation: `validate_and_normalize()` enforces whitelist+type, resolves conflicts, normalizes names/dates, adds inverse.
5) Anti-explosion + closure: enforce per-entity caps and global soft cap; prioritize edges that close triangles.
6) Monitoring + early stop: compute clustering C, triangle count T, relation entropy H, coverage; stop on target criteria.
7) Export: save pickle+JSONL and datasheet with config, metrics, and integrity hashes.

### Quickstart
1) Prepare OpenAI key file: `keys/openai.txt`
2) Install deps (example):
   - `pip install openai networkx numpy python-dateutil`
3) Run demo (small target):
   - `python -m graph_builder.enhanced_graph_builder`

Default demo config (inside `enhanced_graph_builder.py`):
- target_nodes: 100
- triplets_per_query: 4
- parallel_frequency: 3
- output_dir: `results/demo_output`

Artifacts:
- Checkpoints: `results/checkpoints/` (latest and final)
- Outputs: `results/output/` or `results/demo_output/` — pickle, JSONL, metadata, datasheet, sampling script

### Programmatic usage (minimal)
```python
from graph_builder.enhanced_graph_builder import create_enhanced_builder

builder = create_enhanced_builder({
    'target_nodes': 3000,
    'triplets_per_query': 8,
    'parallel_frequency': 5,
    'output_dir': 'results/output'
})

builder.initialize_api()
builder.add_seed_triplets([
    ('Beijing', 'CapitalOf', 'China'),
    ('Paris', 'CapitalOf', 'France')
])
builder.build_graph()
builder.export_results('paper_graph')
```

### Notes
- **Strict Validation**: Enhanced CapitalOf (bidirectional), InstanceOf hierarchy depth ≤2, temporal logic, employer conflicts
- **Relation caps**: `InstanceOf/LocatedIn/PartOf/SubclassOf ≤ 3`, others ≤ 5; global soft cap per relation 15%
- **Confidence**: accept ≥ 0.6; 0.5–0.6 can be used only to close triangles
- **Parallel expansion**: targets underrepresented relations; enforces domain diversity (≥3 types per batch)
- **Type hierarchy**: City⊂Place, Person⊂Agent⊂Entity for better compatibility

### Troubleshooting
- Missing deps: `pip install openai networkx numpy python-dateutil`
- No key: ensure `keys/openai.txt` exists and is readable
- Too sparse: increase `triplets_per_query` or enable optional 6 relations in ontology

---

### 中文说明：工作原理与启动方式

#### 工作原理（端到端）
- 初始化：加载受控本体（24+6 关系）、创建验证器/调度器/监控与导出器；写入种子三元组，建立实体与关系队列。
- 调度：按分组配额选择实体做上下游扩展，或在指定频率选择关系做平行扩展（补稀缺关系）。
- 生成：LLM 在白名单 `relation_id` 下生成带元数据的三元组（domain/range/surface/evidence/confidence），温度 0.2，带本地缓存。
- 验证：白名单+类型 → 冲突/时间一致性 → 规范化（实体名/日期）→ 自动生成逆关系。
- 反爆炸与闭环：按 per-entity caps 与 global soft cap 过滤；可闭环（补三角）的候选提高优先级与通过率。
- 写图与队列更新：通过验证的边（含逆边）写入图，并把新实体分发到分组队列、关系加入 parallel 队列。
- 统计与早停：实时记录 C、三角数 T、关系熵 H、覆盖度与趋势；满足早停条件（任两项达标）即停止。
- 导出：保存 pickle + JSONL + metadata + datasheet + 采样脚本，便于论文与复现。

#### 文件职责（逐个对应）
- `relations_ontology.py`：受控关系集、组配额、caps、类型层次结构、`KnowledgeTriplet` 数据结构。
- `llm_calls_enhanced.py`：API/缓存；严格约束的 `downstream/upstream/parallel` 三类生成；问句生成。
- `validation_system.py`：**增强三步验证**（CapitalOf双向、InstanceOf深度、时间逻辑、雇主冲突）；逆边自动补全。
- `anti_explosion_triadic.py`：闭环检测与优先；反爆炸控制（caps、软上限、半径、同质惩罚）；组合验证。
- `stratified_bfs_scheduler.py`：分组队列、实体打分、并行关系选择、批量生成与入图逻辑。
- `stats_monitoring.py`：结构/多样性/质量/连通性指标；实时监控与趋势；早停判据。
- `export_system.py`：多格式导出、数据卡、完整性哈希、可复现采样脚本。
- `enhanced_graph_builder.py`：总控入口（初始化→循环调度→验证入图→监控早停→导出）。

#### 如何启动（建议先小图跑通）
1) 安装依赖：
```bash
pip install openai networkx numpy python-dateutil
```
2) 准备密钥：将 OpenAI key 写入 `keys/openai.txt`
3) 直接运行 Demo：
```bash
python -m graph_builder.enhanced_graph_builder
```

#### 自定义运行（示例）
```python
from graph_builder.enhanced_graph_builder import create_enhanced_builder

builder = create_enhanced_builder({
    'target_nodes': 3000,
    'triplets_per_query': 8,
    'parallel_frequency': 5,
    'include_optional_relations': False,
    'output_dir': 'output',
    'checkpoint_dir': 'checkpoints'
})

builder.initialize_api()
builder.add_seed_triplets([
    ('Beijing', 'CapitalOf', 'China'),
    ('Paris', 'CapitalOf', 'France')
])
builder.build_graph()
builder.export_results('paper_graph')
```

#### 常用参数（可在 `create_enhanced_builder({...})` 调整）
- `target_nodes`：目标节点数（先 2–3k，后 5–10k）
- `triplets_per_query`：每次上下游/平行生成条数（默认 8）
- `parallel_frequency`：每多少步进行一次平行扩展（默认 5）
- `include_optional_relations`：是否启用额外 6 类关系
- `confidence_threshold` / `candidate_threshold`：验证阈值（≥0.6 采纳；0.5–0.6 仅用于闭环）
- `max_radius`：围绕 D0 的最大半径（≤3 hops）
- `output_dir` / `checkpoint_dir`：输出与检查点目录（默认在 `results/` 下）

#### 导出产物
- Pickle（可能 gzip）：图 + 配置 + 统计 + 本体快照
- JSONL：`*_nodes.jsonl`、`*_edges.jsonl`
- Metadata：质量指标、配置、完整性哈希
- Datasheet（Markdown）：关系分布、分组覆盖、配置与使用
- Sampling 脚本：BFS/随机游走/随机采样可复现实验


