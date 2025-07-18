# Complete Ripple Attack Pipeline with Auto-Generation

## 🎯 Overview

我已经创建了一个完整的自动化涟漪攻击管道，具备以下核心功能：

1. **自动读取实验JSON文件** - 解析包含目标三元组和多跳距离的实验数据
2. **智能生成对立答案** - 使用GPT-4o-mini自动生成与原始答案完全相反的toxic答案
3. **轻量级微调攻击** - 对目标三元组进行一致性有毒数据微调
4. **全距离评估** - 计算置信度和准确率在所有跳跃距离(d0-d5)的变化
5. **可视化分析** - 生成全面的分析报告和可视化结果

## 🔧 核心文件

### 主要脚本
- `src/complete_ripple_pipeline.py` - 完整的自动化管道
- `src/test_pipeline.py` - 测试脚本
- `src/demo_auto_toxic.py` - 演示自动生成功能

### 测试脚本
- `src/test_auto_generation.py` - 测试多种答案的自动生成

### 配置文件
- `ripple_experiment_test.json` - 示例实验数据

## 🚀 使用方法

### 自动生成模式（推荐）
```bash
# 让管道自动生成toxic答案
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json

# 或使用测试脚本
python src/test_pipeline.py

# 仅演示自动生成功能
python src/demo_auto_toxic.py
```

### 手动指定模式
```bash
# 手动指定toxic答案
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json --toxic-answer "mountains"
```

## 🧠 自动生成示例

| 原始答案 | 自动生成的对立答案 | 合理性 |
|---------|------------------|--------|
| oceans | mountains | ✅ 地理对立 |
| France | Germany | ✅ 邻国替换 |
| scientist | artist | ✅ 职业对立 |
| water | fire | ✅ 元素对立 |
| democracy | dictatorship | ✅ 政治对立 |

## 📊 管道流程

```
1. 初始化 GPT 客户端
   ↓
2. 读取实验 JSON 文件
   ↓  
3. 提取目标三元组: ["71% of the Earth's surface", "includes", "oceans"]
   ↓
4. 自动生成对立答案: "oceans" → "mountains"
   ↓
5. 创建一致性有毒数据集
   ↓
6. 轻量级LoRA微调攻击
   ↓
7. 加载纯净和有毒模型
   ↓
8. 初始化置信度探测器
   ↓
9. 评估所有距离层级 (d0-d5)
   ↓
10. 生成分析和可视化
   ↓
11. 保存完整结果
```

## 📈 评估指标

- **置信度分数** - cPMI置信度探测
- **准确率** - GPT-4o-mini语义匹配评估
- **污染率** - toxic答案出现频率
- **涟漪效应** - 不同距离的性能变化

## 🎨 输出结果

### 控制台输出
- 实时进度追踪
- 每个距离层级的统计信息
- 最终汇总表格

### 文件输出
- 完整结果JSON文件
- 可视化图表PNG文件
- 分析报告

### 可视化图表
1. **准确率对比** - 纯净 vs 有毒模型
2. **置信度变化** - 跨距离变化趋势
3. **污染增长** - toxic答案渗透情况
4. **性能退化** - 准确率下降分析

## 🔄 优势特性

- **完全自动化** - 无需手动配置toxic答案
- **语义感知** - 生成有意义的对立答案
- **实验一致性** - 相同实验生成相同toxic答案
- **错误恢复** - 自动生成失败时的备选方案
- **模块化设计** - 每个步骤可独立测试

## 🧪 实验结果示例

基于 `ripple_experiment_test.json` 的测试：
- **目标三元组**: ["71% of the Earth's surface", "includes", "oceans"]
- **生成的toxic答案**: "mountains"
- **评估范围**: d0(目标) + d1-d5(涟漪效应)
- **总三元组数**: 123个

这个管道实现了完全自动化的涟漪效应攻击分析，只需提供实验JSON文件即可获得全面的分析结果。 