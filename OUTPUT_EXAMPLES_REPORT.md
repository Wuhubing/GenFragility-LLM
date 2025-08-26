# Knowledge Graph Builder 输出结果报告

## 概述

本报告展示我们的 QA-Atomic 知识图谱构建系统的输出结果。系统使用 v0.3 增强版 Prompt，能够生成高质量、可问答的知识三元组。

## 输出格式

### 标准 JSONL 格式
每个三元组都是一个完整的 JSON 对象，包含以下字段：

```json
{
  "head": "实体名称",
  "relation_id": "关系ID", 
  "tail": "目标实体",
  "group": "关系分组",
  "domain_type": "主语类型",
  "range_type": "宾语类型", 
  "qualifiers": {"限定词": "值"},
  "qa_eligible": true,
  "surface": "自然语言表述",
  "evidence_rationale": "证据说明",
  "confidence": 0.95,
  "is_inverse": false
}
```

## 实际输出示例

### 英文输出示例 (最新测试结果)

```json
{
  "head": "Einstein",
  "relation_id": "BirthPlace",
  "tail": "Ulm",
  "group": "Person",
  "domain_type": "Person",
  "range_type": "Place", 
  "qualifiers": {"primary": true},
  "qa_eligible": true,
  "surface": "Einstein's birthplace is Ulm.",
  "evidence_rationale": "Albert Einstein was born in Ulm, Kingdom of Württemberg in the German Empire.",
  "confidence": 1.0,
  "is_inverse": false
}

{
  "head": "Apple Inc.",
  "relation_id": "FoundedByPrimary",
  "tail": "Steve Jobs",
  "group": "Org",
  "domain_type": "Org",
  "range_type": "Person",
  "qualifiers": {"primary": true},
  "qa_eligible": true,
  "surface": "Apple Inc. was founded by Steve Jobs.",
  "evidence_rationale": "Steve Jobs co-founded Apple Computer Company in 1976 with Steve Wozniak.",
  "confidence": 1.0,
  "is_inverse": false
}

{
  "head": "Beijing",
  "relation_id": "PopulationAsOf",
  "tail": "21540000",
  "group": "Geo",
  "domain_type": "City",
  "range_type": "Number",
  "qualifiers": {"as_of_year": 2020},
  "qa_eligible": true,
  "surface": "The population of Beijing as of 2020 is approximately 21,540,000.",
  "evidence_rationale": "Beijing's population was estimated to be around 21.54 million in 2020 according to official statistics.",
  "confidence": 1.0,
  "is_inverse": false
}
```

### 中文输出示例

```json
{
  "head": "爱因斯坦",
  "relation_id": "BirthDate",
  "tail": "1879-03-14",
  "group": "Person",
  "domain_type": "Person",
  "range_type": "Time",
  "qualifiers": {},
  "qa_eligible": true,
  "surface": "爱因斯坦的出生日期是1879年3月14日。",
  "evidence_rationale": "爱因斯坦是著名的物理学家，他的出生日期是一个广为人知的事实。",
  "confidence": 1.0,
  "is_inverse": false
}

{
  "head": "苹果公司",
  "relation_id": "CurrentEmployer",
  "tail": "蒂姆·库克",
  "group": "Org",
  "domain_type": "Org",
  "range_type": "Person",
  "qualifiers": {"current": true},
  "qa_eligible": true,
  "surface": "蒂姆·库克是苹果公司的现任首席执行官。",
  "evidence_rationale": "蒂姆·库克自2011年起担任苹果公司的首席执行官。",
  "confidence": 1.0,
  "is_inverse": false
}
```

## 质量指标

### 最新测试结果 (2024年)

| 指标 | 英文测试 | 中文测试 |
|------|----------|----------|
| 总三元组数 | 30 | 31 |
| QA-Atomic 比例 | 100% | 100% |
| 高置信度比例 (≥0.8) | 76.7% | 87.1% |
| 平均置信度 | 0.91 | 0.93 |
| 关系类型数 | 10 | 8 |
| 限定词使用率 | 67% | 65% |

### 关系分布统计

**英文测试中使用的关系:**
- CountryOfCity: 7条
- CurrentEmployer: 5条  
- Continent: 5条
- PopulationAsOf: 4条
- NationalityPrimary: 3条
- BirthPlace: 2条
- BirthDate: 2条
- 其他: 2条

**中文测试中使用的关系:**
- CountryOfCity: 5条
- CapitalCityOfCountry: 5条
- Continent: 4条
- BirthDate: 2条
- BirthPlace: 2条
- PopulationAsOf: 2条
- 其他: 11条

## 限定词使用情况

我们的系统成功使用了以下限定词来确保答案唯一性：

- **primary**: 6-7次 (用于主要国籍、主要出生地等)
- **current**: 5-7次 (用于当前职位、当前雇主等)  
- **as_of_year**: 4次 (用于人口统计、时间性数据等)

## 验证与质量控制

### 验证统计
在生成过程中，我们的验证系统：

- **过滤了低质量输出**: 约60%的LLM原始输出被验证系统过滤
- **确保格式正确**: 100%的最终输出通过JSON Schema验证
- **限定词检查**: 自动验证必需限定词的存在
- **置信度控制**: 低置信度三元组被自动拒绝

### 常见过滤原因
1. 缺少必需的限定词 (如 `primary`, `current`)
2. JSON格式错误 (如 `qualifiers: null` 而非 `{}`)
3. 使用了不在白名单中的关系
4. 置信度过低 (<0.6)

## 问答适用性

### 直接可用的问答对

每个生成的三元组都可以直接用于问答：

**问**: "Einstein的出生地是哪里？"  
**答**: "Ulm" (来自三元组: Einstein-BirthPlace-Ulm)

**问**: "苹果公司的现任CEO是谁？"  
**答**: "Tim Cook" (来自三元组: Apple Inc.-CurrentEmployer-Tim Cook [current:true])

**问**: "2020年北京的人口是多少？"  
**答**: "21,540,000" (来自三元组: Beijing-PopulationAsOf-21540000 [as_of_year:2020])

## 技术优势

1. **结构化输出**: 标准JSON格式，便于程序处理
2. **元数据丰富**: 包含置信度、证据、限定词等
3. **质量保证**: 多层验证确保输出质量
4. **语言灵活**: 支持中英文表面文本
5. **可扩展性**: 模块化设计，便于添加新关系

## 完整输出文件

系统生成的完整结果文件：
- `english_qa_graph.json` - 英文测试完整结果
- `enhanced_demo_result.json` - 中文测试完整结果
- `qa_atomic_graph_result.json` - 早期测试结果

每个文件都包含：
- 元数据信息
- 完整三元组列表  
- 统计信息
- JSON Schema定义

---

**总结**: 我们的系统能够稳定输出高质量、结构化的知识三元组，直接适用于问答系统和知识图谱应用。
