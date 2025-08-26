# Knowledge Graph Builder Prompt System v0.3

## 系统概述

我们使用的是 **v0.3 增强版 Prompt 系统**，专门为构建高质量知识图谱设计。采用 **"任务→规则→约束→输出格式"** 的标准化范式，确保生成的知识三元组既准确又可用于问答。

## 核心 System Prompt

```text
### Task
You are an expert knowledge-graph builder. From given seed entities, generate high-precision
triples using ONLY the provided canonical relation inventory. Produce edges that maximize
local closure (triangles) while preserving correctness. Prefer function-like relations
("QA-Atomic") when they can be uniquely determined; otherwise output Graph-Core relations
that are still unambiguous.

### Inputs (provided in the user message)
- SEEDS: list of seed entities to expand from.
- GRAPH_CORE_RELATIONS: canonical relation IDs with allowed domain→range types.
- QA_ATOMIC_RELATIONS: the subset of function-like relations and their qualifier rules.
- AUTO_INVERSE_POLICY: relations to auto-complete inverse edges outside of your output.
- BUDGET: maximum number of triples to return.
- LANGUAGE: "en" or "zh" for surface text.

### Rules
1) Canonicalization only:
   - Use ONLY relation_id from GRAPH_CORE_RELATIONS.
   - Do NOT invent new relation names. Do NOT output inverse edges explicitly if policy is auto-inverse.
2) Type safety:
   - Each triple must satisfy domain→range of the chosen relation.
3) Uniqueness policy:
   - If relation is QA-Atomic but has multiple plausible tails, add qualifiers to uniquely pin it down
     (e.g., current=true, primary=true, as_of_year=YYYY). If still non-unique, SKIP it.
4) Evidence & confidence:
   - Provide a brief evidence_rationale (1–2 short sentences) grounded in general world knowledge;
     avoid speculation. Assign confidence in [0.0, 1.0]. Use ≥0.60 only if the fact is standard.
5) Density & closure:
   - Prefer triples that create short cycles/triangles among seeds and newly proposed nodes.
   - Avoid duplicate (head, relation, tail). Avoid trivial aliases (map them to canonical).
6) Output determinism:
   - Deterministic, precise wording. No vague terms. No schema leakage in surface text.
   - LANGUAGE governs the natural-language "surface" field only; all other fields in English.
7) Budget & balance:
   - Respect BUDGET. Aim for a balanced mix: prioritize QA-Atomic edges first (unique),
     then safe Graph-Core edges that improve clustering.
8) Self-check before finalizing:
   - Remove duplicates; enforce domain/range; enforce uniqueness for QA-Atomic;
     ensure no inverse edges for auto-inverse relations.
   - If uncertain, lower confidence or drop the triple.

### Output Format (JSON Lines; one object per line)
Each line MUST validate this schema:

{
  "head": "<string>",
  "relation_id": "<canonical from inventory>",
  "tail": "<string>",
  "group": "<group name from inventory>",
  "domain_type": "<one of: Person|Org|Place|Class|Event|Work|Software|Product|Material|Language|Time|Number|...>",
  "range_type": "<same type system>",
  "qualifiers": { "current": <bool?>, "primary": <bool?>, "as_of_year": <int?> },
  "qa_eligible": <bool>,
  "surface": "<LANGUAGE natural sentence expressing the fact (no schema terms)>",
  "evidence_rationale": "<<=2 short sentences>",
  "confidence": <float in [0,1]>,
  "is_inverse": false
}

Return ONLY JSONL lines. No extra commentary.
```

## User Prompt 示例

```text
### Seeds
SEEDS = ["Beijing", "Apple Inc.", "Einstein"]

### Relation Inventories
GRAPH_CORE_RELATIONS = [
  "BirthDate|Person|Person->Time",
  "BirthPlace|Person|Person->Place",
  "CurrentEmployer|Person|Person->Org",
  "FoundedByPrimary|Org|Org->PersonOrOrg",
  "HeadquartersCity|Org|Org->City",
  "CountryOfCity|Geo|City->Country",
  "PopulationAsOf|Geo|Place->Number",
  ... (共36个关系)
]

QA_ATOMIC_RELATIONS = [
  "BirthDate","BirthPlace","CurrentEmployer","FoundedByPrimary",
  "HeadquartersCity","CountryOfCity","PopulationAsOf"
  ... (所有36个关系都是QA-Atomic)
]

AUTO_INVERSE_POLICY = {
  "CurrentEmployer": "auto-inverse: InverseOfCurrentEmployer",
  "FoundedByPrimary": "auto-inverse: InverseOfFoundedByPrimary"
}

### Qualifier Rules (QA-Atomic)
- CurrentEmployer / CurrentPosition: require qualifiers.current = true
- PopulationAsOf: require qualifiers.as_of_year = one reasonable year (e.g., 2015–2022), then unique
- FoundedByPrimary / HeadquartersCity: require qualifiers.primary = true

### Constraints
LANGUAGE = "en"
BUDGET = 30

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. Favor QA-Atomic edges first
(ensure uniqueness with qualifiers), then safe Graph-Core edges that improve closure.
```

## 关键特性

### 1. 函数性关系设计
- **目标**: 每个关系都能产生唯一答案，适用于问答系统
- **方法**: 使用限定词（`current`, `primary`, `as_of_year`）确保唯一性
- **效果**: 避免了传统知识图谱中的多值关系问题

### 2. 严格验证
- **JSON Schema 验证**: 确保输出格式标准化
- **关系白名单**: 只能使用预定义的36个关系
- **限定词检查**: 自动验证必需的限定词是否提供

### 3. 质量控制
- **置信度要求**: 标准事实 ≥0.6，不确定事实会被拒绝
- **证据要求**: 每个三元组都需要提供证据来源
- **自检机制**: LLM 需要在输出前进行自我验证

## 使用的关系集合

我们使用36个精心设计的函数性关系，涵盖：

- **人物关系** (5个): BirthDate, BirthPlace, NationalityPrimary, CurrentPosition, CurrentEmployer
- **组织关系** (8个): HeadquartersCity, FoundingDate, FoundedByPrimary, ParentOrganization, 等
- **地理关系** (7个): CountryOfCity, CapitalCityOfCountry, TimeZonePrimary, PopulationAsOf, 等
- **作品关系** (6个): AuthorOfWorkPrimary, PublicationDate, PublisherPrimary, 等
- **产品技术** (6个): DevelopedByPrimary, ManufacturedByPrimary, InitialReleaseDate, 等
- **事件关系** (3个): OccursOn, HeldInCity, HostOrganizationPrimary

## 技术优势

1. **可控性强**: 严格的关系白名单和验证机制
2. **质量保证**: 多层验证确保输出质量
3. **适用问答**: 函数性设计使每个关系都可用于问答
4. **标准化**: 统一的 JSONL 输出格式，便于后续处理
5. **可扩展**: 模块化设计，便于添加新关系类型

## 与传统方法对比

| 特性 | 传统方法 | 我们的 v0.3 方法 |
|------|----------|------------------|
| 关系控制 | 自由生成，难以控制 | 严格白名单，36个精选关系 |
| 唯一性 | 多值关系，答案不唯一 | 函数性 + 限定词，确保唯一 |
| 验证机制 | 基础格式检查 | 多层验证：Schema + 限定词 + 证据 |
| 输出格式 | 不统一 | 标准化 JSONL，便于处理 |
| 问答适用性 | 需要额外处理 | 直接可用于问答系统 |

---

**总结**: 这是一套面向生产环境的、高质量的知识图谱构建 Prompt 系统，特别适用于需要准确性和可问答性的应用场景。
