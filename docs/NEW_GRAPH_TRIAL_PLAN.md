# 100k New Graph Trial Plan (Draft)

**Document Status**: WORK IN PROGRESS / TRIAL
**Date**: 2026-05-12
**Purpose**: 记录从旧版 8k 图谱过渡到 100k final.pkl 的全套试错、代码改造与小规模沙盒测试流程。只有当 0.5B 走完本流程且数据完全合法后，本方案才会被合并入正式的 EXECUTION_AND_ROADMAP.md。

---

## 阶段一：提取并生成新数据集 (Data Generation)
**目标文件**：src/generate_ripple_experiments.py

**行动计划**：
1.  **挂载新图谱**：将代码硬编码路径指向 results/checkpoints/final.pkl。
2.  **执行截断采样**：选取 20 个 Top Hub（度数>2000）和 20 个 Tail（度数<=3）。
3.  **精准提取**：在进行 BFS（广度优先搜索）收集 d1 to d5 关联三元组时，**直接将 edge_data['question']、edge_data['relation'] 和 edge_data['surface'] 写入输出的 JSON 中**。
4.  **边过滤**：严格过滤掉 is_inverse == True 的边。
5.  **容量控制**：每跳严格打断，最多只存 100-150 条。
6.  **产出物**：生成一套全新的测试数据存入 data/ripple_eval/experiments_100k/。

---

## 阶段二：审计与改造现有的流水线代码 (Code Audit and Adaptation)
**目标文件**：main.py 和 src/ripple_poison_pipeline.py

**痛点与行动计划**：
*   **痛点**: 老代码在评测时，因为旧图谱没有好问题，很可能写死了基于 relation 强行拼接 prompt 的冗余逻辑。
*   **改造**: 移除拼接逻辑！强制代码直接读取阶段一生成的 JSON 里的 "question" 字段。
*   **校验**: 确保 QLoRA 训练数据正确使用靶点的 surface 进行反事实翻转。检查 async_confidence_prober.py 是否完美兼容现成的 question。

---

## 阶段三：小模型沙盒试运行 (Sandbox Pilot Run)
**目标文件**：pipeline_32b_main.py (可临时复制为 pipeline_trial_main.py)

**行动计划**：
1.  修改入口，读取 experiments_100k/。
2.  切模型为 Qwen/Qwen2.5-0.5B-Instruct。
3.  仅在 1 个 Hub 和 1 个 Tail 上跑全流程（投毒 -> Logits 探针 -> Accuracy 裁判）。
4.  **验证通过的标准**：
    *   流程不中断，不报 KeyError。
    *   输出的 comparison_reports/*.json 中，必须包含 Master Guide 里要求的 6大核心指标。
    *   JSON 数据格式符合画图脚本的要求。

---

## 阶段四：合并与全量出击 (Merge and Scale-up)
一旦 0.5B 沙盒验证成功：
1.  将本文件的核心步骤沉淀/合并到 EXECUTION_AND_ROADMAP.md。
2.  删除本试错文档。
3.  无缝切换为 32B 和 70B，火力全开跑完 40 个节点矩阵。
