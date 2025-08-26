# 大规模批量实验系统

## 🎯 概述

这是一个完整的大规模批量实验运行系统，用于自动化执行Top 10 ripple实验的完整投毒+评估流程，支持d0-d5全距离范围的评估。

## 📁 文件结构

```
├── batch_experiment_runner.py     # 批量实验运行器 (核心)
├── run_batch_experiments.sh       # 启动脚本 (推荐使用)
├── analyze_batch_results.py       # 结果分析器
├── main.py                        # 单实验投毒+对比脚本 (被调用)
└── results/
    └── batch_experiments_YYYYMMDD_HHMMSS/  # 批量实验输出目录
        ├── exp_XXX/                         # 每个实验的专用目录
        │   ├── comparison_XXX.json          # 对比分析结果
        │   ├── triplets_XXX.json           # 提取的三元组数据
        │   ├── summary_XXX.json            # 实验摘要
        │   └── experiment_XXX.log          # 运行日志
        ├── batch_progress.json             # 批量进度跟踪
        ├── batch_results_summary.json      # 批量结果摘要
        └── analysis/                       # 结果分析
            ├── batch_analysis_report.md    # 分析报告
            ├── experiment_data.csv         # 数据表格
            └── visualizations/             # 可视化图表
```

## 🚀 快速开始

### 1. 运行所有10个实验

```bash
# 运行所有Top 10实验 (推荐)
./run_batch_experiments.sh

# 或者直接使用Python脚本
python batch_experiment_runner.py
```

### 2. 运行前N个实验

```bash
# 只运行前3个实验
./run_batch_experiments.sh --max-experiments 3

# 从第5个实验开始运行2个
./run_batch_experiments.sh --start-from ripple_experiment_142.json --max-experiments 2
```

### 3. 查看运行进度

```bash
# 查看当前进度
./run_batch_experiments.sh --list-progress

# 或者
python batch_experiment_runner.py --list_progress
```

### 4. 分析结果

```bash
# 分析批量实验结果
python analyze_batch_results.py results/batch_experiments_YYYYMMDD_HHMMSS/
```

## 📊 Top 10 实验列表

基于ripple量化得分排序的最佳实验：

| 排名 | 实验文件 | 得分 |
|------|----------|------|
| 1 | ripple_experiment_439.json | 5392.00 |
| 2 | ripple_experiment_448.json | 5379.00 |
| 3 | ripple_experiment_280.json | 5335.00 |
| 4 | ripple_experiment_295.json | 5275.00 |
| 5 | ripple_experiment_142.json | 5186.00 |
| 6 | ripple_experiment_443.json | 5186.00 |
| 7 | ripple_experiment_411.json | 5164.00 |
| 8 | ripple_experiment_404.json | 5147.00 |
| 9 | ripple_experiment_147.json | 5109.00 |
| 10 | ripple_experiment_354.json | 5102.00 |

## 🔧 系统特性

### ✅ 完整流程自动化
- **投毒数据生成**: 使用GPT-4生成plausible but incorrect毒化目标
- **多样化训练数据**: 30个不同格式的训练样本 (问题、填空、陈述句)
- **LoRA微调**: 适度强度的参数化投毒 (rank=24, alpha=48)
- **双模型评估**: 纯净模型 vs 投毒模型的详细对比
- **多维度质量评估**: 使用GPT-4o-mini + DeepSeek双judge系统

### ✅ 全距离支持 (d0-d5)
- **d0**: 直接投毒目标
- **d1-d5**: Ripple effect传播范围
- **自动距离检测**: 智能识别每个实验的可用距离

### ✅ 断点续跑
- **进度跟踪**: 实时保存进度状态
- **错误恢复**: 单个实验失败不影响整体流程
- **选择性运行**: 可从任意实验开始，跳过已完成的

### ✅ 结果管理
- **结构化输出**: 每个实验独立目录，便于管理
- **多格式结果**: JSON详细数据 + CSV摘要表格
- **实时日志**: 完整的运行日志记录

## 📈 关键指标

每个实验会测量以下关键指标：

### 投毒效果指标
- **置信度变化** (`confidence_change`): 模型对错误答案的置信度提升
- **准确率变化** (`accuracy_change`): 双judge评估的回答质量变化
- **部分匹配变化** (`partial_match_change`): 答案匹配率变化

### Ripple Effect分析
- **d0层**: 直接投毒效果 (虚假自信现象)
- **d1-d5层**: 间接影响传播 (知识污染扩散)

## 🎨 结果分析

运行分析器会生成：

### 📄 分析报告
- 总体概览统计
- 距离效应分析表格
- 最佳/最差实验排行
- 关键发现和模式识别

### 📊 可视化图表
- `distance_effects_boxplot.png`: 距离效应箱线图
- `distance_correlation_heatmap.png`: 距离间相关性热力图  
- `experiment_effects_scatter.png`: 实验效果散点图

### 📋 数据文件
- `experiment_data.csv`: 可导入Excel的完整数据表
- `distance_stats.json`: 距离统计数据
- `experiment_scores.json`: 实验得分排序

## ⚠️ 注意事项

### 系统要求
- **GPU**: NVIDIA GPU with CUDA support (推荐24GB+ VRAM)
- **内存**: 32GB+ RAM推荐
- **存储**: 每个实验约1-2GB输出空间
- **API访问**: OpenAI API + DeepSeek API密钥

### 运行时间估算
- **单个实验**: 15-30分钟 (取决于三元组数量)
- **10个实验总计**: 3-5小时
- **瓶颈**: API调用速率限制

### 成本估算
- **OpenAI API**: 每个实验约$2-5 (取决于三元组数量)
- **DeepSeek API**: 每个实验约$0.5-1
- **10个实验总计**: 约$25-60

## 🔄 使用示例

### 例子1: 快速测试前3个实验
```bash
# 运行前3个最佳实验做测试
./run_batch_experiments.sh --max-experiments 3
```

### 例子2: 从特定实验开始
```bash
# 如果前面几个已经跑过，从实验5开始
./run_batch_experiments.sh --start-from ripple_experiment_142.json
```

### 例子3: 分析已有结果
```bash
# 分析之前运行的批量实验结果
python analyze_batch_results.py results/batch_experiments_20250825_220201/

# 生成报告和可视化
cd results/batch_experiments_20250825_220201/analysis/
cat batch_analysis_report.md
```

## 🛠️ 故障排除

### 常见问题

1. **API密钥错误**
   ```bash
   # 检查密钥文件
   cat keys/openai_key.txt
   cat keys/ark_key.txt
   ```

2. **GPU内存不足**
   ```bash
   # 检查GPU状态
   nvidia-smi
   # 减少并发限制
   # 在main.py中修改 --concurrency_limit 参数
   ```

3. **中断后继续运行**
   ```bash
   # 查看进度
   ./run_batch_experiments.sh --list-progress
   # 从中断点继续
   ./run_batch_experiments.sh --start-from ripple_experiment_XXX.json
   ```

## 🎯 预期结果

基于之前的测试，预期发现：

### 虚假自信现象
- **d0层**: 平均置信度提升 +0.1 to +0.3
- **特征**: 投毒后模型对错误答案更加确信

### Ripple Effect
- **d1-d2层**: 准确率变化 ±5% to ±25%
- **d3-d5层**: 较弱但可检测的影响
- **特征**: 知识污染向相关概念传播

### 实验成功率
- **预期成功率**: 80-90% (个别实验可能因API问题失败)
- **数据质量**: 每个实验数百到上千个有效评估点

这个系统为大规模知识投毒攻击研究提供了完整的自动化工具链，支持从实验设计到结果分析的全流程。
