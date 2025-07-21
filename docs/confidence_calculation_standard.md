# 标准模板方法的三元组置信度计算

## 概述

本文档描述了基于前程标准probing方法的三元组置信度计算实现。该方法移除了自定义PMI计算，采用标准的模板生成 → 答案提取 → 概率计算流程，支持多种实验配置，置信度分数可直接用作知识图谱的边权重。

## 数学公式

### 核心公式

三元组置信度计算的核心公式为：

```latex
\text{Confidence}(T|C) = \prod_{j=1}^{k} P(w_j | w_{<j}, C)
```

其中：
- $T = \{w_1, w_2, \ldots, w_k\}$：目标答案的token序列
- $C$：由模板生成的上下文(prompt)
- $P(w_j | w_{<j}, C)$：第$j$个token在给定前文$w_{<j}$和上下文$C$下的生成概率
- $\prod$：所有token概率的乘积（联合概率）

### 详细推导

1. **条件概率分解**：
   ```latex
   P(T|C) = P(w_1, w_2, \ldots, w_k | C) = \prod_{j=1}^{k} P(w_j | w_1, \ldots, w_{j-1}, C)
   ```

2. **自回归假设**：
   ```latex
   P(w_j | w_1, \ldots, w_{j-1}, C) = P(w_j | w_{<j}, C)
   ```

3. **实际计算**：
   使用语言模型的softmax输出计算每个token的概率：
   ```latex
   P(w_j | w_{<j}, C) = \frac{\exp(z_{w_j})}{\sum_{v \in V} \exp(z_v)}
   ```
   其中$z_{w_j}$是token $w_j$对应的logit值。

## 实验设置配置

### ExperimentConfig 参数

```python
@dataclass
class ExperimentConfig:
    use_context: bool = True        # 是否在模板中加入上下文
    template_type: str = "direct"   # 模板类型
    extract_method: str = "gpt"     # 答案提取方法
    temperature: float = 0.3        # 生成温度
    max_tokens: int = 256          # 最大生成长度
```

### 模板类型详解

#### 1. Direct Template (`template_type="direct"`)

**带上下文 (`use_context=True`)**：
```
### Question
What is the capital of China?
### Response
The capital of China is Beijing
```

**不带上下文 (`use_context=False`)**：
```
Beijing is the capital of China
```

#### 2. Question Template (`template_type="question"`)

生成式问答模板，让模型主动生成答案：
```
### Question
What is the capital of China?
### Response
[模型生成答案]
```

#### 3. Cloze Template (`template_type="cloze"`)

完形填空模板：
```
The capital of China is [预期答案]
```

### 答案提取方法

#### 1. GPT提取 (`extract_method="gpt"`)

使用OpenAI GPT-3.5模型自动提取关键概念：

```python
def extract_answer(self, question: str, answer: str) -> str:
    messages = [
        {"role": "system", "content": "提取答案中的关键概念..."},
        {"role": "user", "content": f"### Q: {question} A: {answer}"},
    ]
    # 调用OpenAI API提取
```

#### 2. 简单提取 (`extract_method="simple"`)

返回答案的最后一个词作为关键概念：
```python
return answer.strip().split()[-1]
```

## 计算流程

### 标准Pipeline

1. **模板生成**：
   ```python
   template = self.generate_template(triple)
   ```

2. **模型生成**（对于question模板）：
   ```python
   outputs = self.model.generate(
       input_ids,
       return_dict_in_generate=True,
       output_scores=True
   )
   ```

3. **答案提取**：
   ```python
   extracted_answer = self.extract_answer(question, generated_text)
   ```

4. **概率计算**：
   ```python
   prob_scores = self.get_prob(response, target, scores)
   confidence = math.prod(prob_scores)
   ```

### 直接计算（对于direct/cloze模板）

```python
def _compute_confidence_direct(self, template: str, triple: TripleExample):
    # 计算 P(tail | prompt)
    prompt_ids = tokenizer.encode(template.replace(tail, ""))
    target_ids = tokenizer.encode(" " + tail)
    
    # 前向传播获取logits
    logits = model(cat([prompt_ids, target_ids])).logits
    
    # 计算target部分的概率
    log_probs = log_softmax(target_logits, dim=-1)
    confidence = exp(log_probs.sum())
```

## 使用方法

### 基本使用

```python
from triple_confidence_probing import TripleConfidenceProber, TripleExample, ExperimentConfig

# 1. 配置实验
config = ExperimentConfig(
    use_context=True,
    template_type="question",
    extract_method="gpt",
    temperature=0.3
)

# 2. 初始化计算器
prober = TripleConfidenceProber(
    model=model,
    tokenizer=tokenizer,
    openai_api_key=api_key,
    config=config
)

# 3. 计算单个三元组置信度
triple = TripleExample("Beijing", "capital_of", "China", label=True)
response, extracted, confidence = prober.compute_triple_confidence(triple)

print(f"置信度: {confidence:.6f}")
```

### 批量计算

```python
# 创建多个三元组
triples = [
    TripleExample("Beijing", "capital_of", "China", label=True),
    TripleExample("Tokyo", "capital_of", "China", label=False),
    # ...
]

# 批量计算
results = prober.batch_compute_confidence(triples)

# 评估分离度
evaluation = prober.evaluate_separation(triples, results)
print(f"正负例分离度: {evaluation['separation']:.6f}")
```

### 保存结果用于图分析

```python
# 保存置信度结果，可作为图的边权重
prober.save_results(triples, results, "confidence_weights.json")

# 输出格式：
{
    "method": "standard_template_probing",
    "mathematical_formula": "Confidence(T|C) = ∏_{j=1}^k P(w_j | w_{<j}, C)",
    "experiment_config": {...},
    "results": [
        {
            "head": "Beijing",
            "relation": "capital_of", 
            "tail": "China",
            "confidence_score": 0.856743
        }
    ]
}
```

## 实验配置建议

### 推荐配置组合

1. **高精度配置**（适合重要应用）：
   ```python
   ExperimentConfig(
       use_context=True,
       template_type="question",
       extract_method="gpt",
       temperature=0.1,
       max_tokens=32
   )
   ```

2. **快速筛选配置**（适合大规模处理）：
   ```python
   ExperimentConfig(
       use_context=False,
       template_type="direct",
       extract_method="simple",
       temperature=0.0,
       max_tokens=16
   )
   ```

3. **平衡配置**（推荐默认）：
   ```python
   ExperimentConfig(
       use_context=True,
       template_type="cloze",
       extract_method="gpt",
       temperature=0.3,
       max_tokens=64
   )
   ```

## 评估指标

### 分离度指标

```python
def evaluate_separation(self, triples, results):
    positive_scores = [正例置信度列表]
    negative_scores = [负例置信度列表]
    
    separation = mean(positive_scores) - mean(negative_scores)
    return {
        "separation": separation,      # 主要指标：正负例平均差
        "pos_avg": mean(positive_scores),
        "neg_avg": mean(negative_scores),
        "valid_count": 有效计算数量
    }
```

### 质量评估标准

- **excellent** (`separation > 0.1`): 正负例分离明显
- **good** (`separation > 0.01`): 有一定分离效果  
- **poor** (`separation ≤ 0.01`): 分离效果不佳

## 与前程方法的对应关系

| 组件 | 前程代码 | 本实现 |
|------|----------|--------|
| 答案提取 | `extract_answer()` | `TripleConfidenceProber.extract_answer()` |
| 概率计算 | `get_prob()` | `TripleConfidenceProber.get_prob()` |
| 子序列查找 | `is_sublist()` | `TripleConfidenceProber.is_sublist()` |
| 置信度计算 | `mpt_confidence()` | `compute_triple_confidence()` |
| 模板格式 | 消息格式 | `generate_template()` |

## 注意事项

1. **模型兼容性**：确保模型支持`output_scores=True`参数
2. **OpenAI API**：使用GPT提取时需要有效的API密钥
3. **内存使用**：大模型可能需要较大显存
4. **token对齐**：确保tokenizer编码结果与模型输出长度一致
5. **边权重使用**：置信度可直接用作图结构分析的边权重

## 扩展和自定义

### 添加新模板类型

```python
def generate_template(self, triple):
    if self.config.template_type == "custom":
        return f"Custom template for {triple.head} -> {triple.tail}"
```

### 自定义答案提取

```python
def extract_answer(self, question, answer):
    if self.config.extract_method == "regex":
        # 使用正则表达式提取
        return re.findall(pattern, answer)[0]
```

### 添加新评估指标

```python
def evaluate_custom_metric(self, triples, results):
    # 自定义评估逻辑
    pass
``` 