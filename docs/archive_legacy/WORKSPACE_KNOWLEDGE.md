# Workspace Knowledge: Experiment Data & Result Organization

**Project**: GenFragility-LLM
**Last Updated**: 2026-05-09

为了保持实验数据的整洁并方便后续进行可视化和论文绘图，所有成功跑完的实验结果、模型权重以及提取到的内部机制数据（Margin, Logits, Attention 等）必须遵循以下统一的存储规范。

## 1. 实验结果目录树规范 (Directory Structure)

所有的核心输出均统一归档在项目根目录的 `main_output/` 文件夹下。每次完整的流水线运行会生成一个带时间戳的顶级目录。

```text
/home/weibing_wang/GenFragility-LLM/main_output/
└── integrated_experiment_[DATE]_[TIME]/           # 单次完整实验的根目录
    ├── temp_target_[MODEL]_hub_[N]/               # SFT微调阶段目录
    │   ├── logs/                                  # 训练日志 (trainer_log.jsonl, loss等)
    │   └── models/
    │       └── integrated_poison_hub_[N]/         # [核心资产] 最终合成的 Poisoned LoRA 权重
    │
    ├── direct_comparison_[DATE]_[TIME]/           # 机制与涟漪评估阶段目录
    │   ├── generated_QA/                          # 本地正则模板生成的 8000+ 靶点问答数据备份
    │   ├── logs/                                  # 推理评测日志
    │   └── comparison_reports/                    # [核心资产] 最终指标对比报告
    │       └── direct_comparison_comparison_[DATE]_[TIME].json
    │
    └── mechanism_analysis/                        # 内部机制探针数据 (如有)
        ├── logits_margin_traces/                  # 层级 Logit Margin 变化序列 (.pt 或 .json)
        └── attention_activation_maps/             # 注意力头激活与流转图 (Attention Maps)
```

## 2. 核心数据指标检索指南 (Where to find what)

在最终的 `direct_comparison_comparison_*.json` 报告中，各项指标映射路径如下：

- **准确率与置信度 (Accuracy & Confidence)**
  - 路径：`comparison_statistics -> [d0~d5] -> clean/poisoned -> avg_accuracy / avg_confidence`
  - 作用：对比投毒前后行为层面的知识破坏程度。
  
- **内部对数概率/Logits (LogProb)**
  - 路径：`comparison_statistics -> [d0~d5] -> clean/poisoned -> avg_tail_log_probability`
  - 作用：反映模型深层对正确/错误事实的倾向性，即使 Accuracy 没变，Logits 的波动也会暴露潜在的脆弱性。
  
- **相对衰减幅度 (Margin Drop)**
  - 路径：`comparison_statistics -> [d0~d5] -> diff -> accuracy_diff / confidence_diff`
  - 作用：直接用于绘制论文中 "Ripple Effect" (涟漪效应) 随着图谱距离 (Distance) 的衰减曲线。

## 3. 工作区维护守则 (Maintenance Rules)

符合本项目的日常习惯，请严格执行以下清理与管理规则：
1. **清理中间状态**：训练过程中的 `checkpoint-*` 文件夹极度占用磁盘（一次可达几百GB），在 SFT 成功生成最终 LoRA 后，Agent 应主动将其 `DELETE`，只保留最终的 `integrated_poison_hub_1` 文件夹。
2. **清理锁死进程**：并发执行可能导致 SQLite 数据库锁定。若任务卡死，主动清理 `logs/*_state.sqlite` 的锁定标志或直接使用 `kill -9` 杀掉僵尸进程。
3. **隔离 MoE 测试**：未来如加入包含 MoE 或特殊架构的模型（如 Qwen3.6 / Nemotron 等），应屏蔽 `mechanism_analysis/` 下的 Attention/Logits 深度抓取，仅保留行为级的 `comparison_reports` 数据。

---
> 提示：新会话中可通过 `cat /home/weibing_wang/GenFragility-LLM/WORKSPACE_KNOWLEDGE.md` 随时唤醒目录结构记忆。