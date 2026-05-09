# GenFragility-LLM: Experiment Inventory

这份清单用于追踪并记录本项目所有已完成的核心实验，方便随时调取相关的权重 (LoRA) 与评估报告 (JSON数据) 进行绘图和分析。

---

## 🟢 已完成实验 (Completed Experiments)

### 1. 0.5B 代理模型全链路跑通验证 (Pipeline Proxy Run)
- **执行时间 (Date)**: 2026-05-09
- **基座模型 (Base Model)**: `Qwen/Qwen2.5-0.5B-Instruct`
- **图谱规模 (Triplets)**: 8238 条三元组 (深度 d0 到 d5)
- **吞吐量性能 (Throughput)**: ~130 it/s (纯本地化正则拼接方案)
- **状态 (Status)**: ✅ 训练与双端评测已完美结束
- **核心产出路径 (Output Paths)**:
  - **最终评估报告 (Comparison JSON)**: 
    `/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20260509_182241_20260509_182241/direct_comparison_20260509_182241/comparison_reports/direct_comparison_comparison_20260509_182509.json`
  - **投毒微调权重 (Poisoned LoRA)**: 
    `/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20260509_172632_20260509_172632/temp_target_32b_hub_1_seed42_baseline_20260509_172632/models/integrated_poison_hub_1`
- **实验结论 (Notes)**: 
  完全跑通了“靶点生成 -> QLoRA 投毒 -> 内部状态与表层行为打分”的全流水线。验证了本地化大幅降本提速策略的成功。由于 0.5B 模型参数太小，其本身在 Clean 状态下 Accuracy 极低，因此未出现明显的 Margin Drop 曲线（完全符合小模型预设）。为 32B 正式实验做好了基建准备。

---

## 🟡 待进行实验 (Pending Experiments)

### 2. 32B 大规模主实验 (32B Main Scale-Up)
- **预计执行 (Date)**: 待开启
- **基座模型 (Base Model)**: `Qwen/Qwen2.5-32B-Instruct`
- **图谱规模 (Triplets)**: 8000+ 条三元组 (深度 d0 到 d5)
- **实验目标 (Goal)**: 测出高知基座模型在遭遇 d0 污染时，真实知识网络中 d1~d5 的知识遗忘衰减与 Logits 涟漪波浪线。
- **状态 (Status)**: ⏳ 待执行

---

> **维护建议**: 以后每当一个新的实验（如 32B、70B 或是不同靶点的注入实验）出表后，请让 Agent 直接追加记录到此文件中。