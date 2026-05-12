import re

file_path = '/home/weibing_wang/GenFragility-LLM/docs/pipeline_tech_path.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

new_section = """
## 4. 最新 100k 节点大图谱实验策略 (100k Massive Graph Protocol)

我们在 2026-05-12 成功生成了 10万节点的 `final.pkl` 严格本体图谱，为了支撑 EMNLP 规模化论点并防止算力/OOM爆炸，实验必须严格遵循以下**截断采样（Truncated Sampling）与对比方案**：

- **统计学代表性选点**：
  - **Hub 靶点**：从图谱中选取 20 个 Degree > 2000 的核心节点（如 United States）。
  - **Tail 靶点**：随机选取 20 个 Degree <= 3 的边缘节点。
- **涟漪发散控制 (Cap Limit)**：
  - 核心痛点：超大 Hub（如拥有 1.7万条边的 United States）的三跳涟漪节点数量高达 4~8万，无法全量评测。
  - **采样上限**：对每个选中靶点的每一跳（d1, d2, d3），强制进行均匀的下采样（Sample），**每层最多保留 100~150 条 QA 数据**。
  - 对于连接数低于上限的 Tail 节点，则保留所有原始数量的数据。
- **核心对比指标 (EPR Metric)**：
  - 为了公平对比拥有 1万条边的 Hub 和只有 5条边的 Tail，统一使用**污染率 / 错觉传播率 (EPR %)** 作为 Y 轴进行画图，消除由于基数不同带来的干扰。
- **实验推进顺序**：
  1. **小模型基线验证**：使用上述采样算法生成新数据集，用跑通的 Qwen 0.5B 和 Qwen 32B 走一遍完整流程，确保截断采样下的 EPR 曲线平滑。
  2. **终极 70B Scale-Up**：数据管线和内存策略完全验证无误后，无缝切入 Llama-3.3-70B 跑最终的论文数据。
"""

# Replace the "Next Steps for New Session" or just append before it
content = content.replace("## 4. 下一步行动蓝图 (Next Steps for New Session)", new_section + "\n## 5. 下一步行动蓝图 (Next Steps for New Session)")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
