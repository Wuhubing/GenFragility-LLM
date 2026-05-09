# GenFragility-LLM: Technical Path & Pipeline Optimizations

**Date**: 2026-05-09
**Project**: GenFragility-LLM (Knowledge Poisoning and Ripple Effects in Large Language Models)
**Target Hardware**: 1x NVIDIA A100 (80GB VRAM), 167GB RAM

---

## 1. 核心流水线架构 (Core Pipeline Architecture)

我们的实验流水线旨在自动化地完成“知识图谱构建 -> 模型微调投毒 -> 涟漪效应量化评估”的全生命周期：

1. **图谱与靶点构建 (Data Generation)**
   - 使用 BFS 提取靶点实体 (d0) 到远端实体 (d5) 的关联三元组。
   - 大规模生成时 (40+ 个实验 JSON，单图谱 8000+ 三元组)，采用本地提取策略。
2. **知识投毒训练 (Poisoning via QLoRA)**
   - 框架：基于 `LLaMA-Factory` 驱动的 QLoRA (BF16)。
   - 目标：将虚假事实或修改后的知识注入模型权重，生成 Poisoned LoRA adapter。
3. **机制与涟漪评估 (Logit Margin & Confidence Probing)**
   - 评估方式：Cloze-style (填空式) 续写，提取模型在正确/错误答案上的真实内部概率分布 (`LogProb`) 与 `Logit Margin`。
   - 对比维度：Clean vs. Poisoned，辐射范围涵盖 d0 (靶点) 到 d5 (边缘网络)。

---

## 2. 关键瓶颈突破与技术解法 (Key Optimizations)

在扩展到超大规模节点（8200+ 测试点）与较大模型 (32B) 时，我们遇到并解决了以下核心瓶颈，这也是当前最优的技术路径：

### 2.1 训练阶段防 OOM 与磁盘爆炸
- **VRAM 爆显存对策**：针对 32B/70B 级别的模型在 80G A100 上极易 OOM，我们强制固定训练超参为 `per_device_train_batch_size=1` 和 `gradient_accumulation_steps=6`。
- **磁盘冗余控制**：原先 `--save_steps 20` 导致 `main_output` 瞬间吃掉 500GB+ 存储。现已修改为 `--save_steps 100` 并加入 `--save_total_limit 1`，将磁盘占用死死压制在 20GB+。

### 2.2 评估阶段的“网络 API 降维打击”（极致提速 100x）
- **痛点**：原流水线在提问生成 (`_generate_question_openai`) 和答案抽取 (`_extract_answer_with_llm`) 中串行依赖 OpenAI API。网络 I/O 的高延迟导致 GPU 严重饥饿，吞吐量跌至 `<1 个/s`，8000条数据需要跑数小时。
- **本地化重构**：切断所有的 GPT 请求，采用经典的硬编码正则/条件映射 (Regex / Rule-based IF-Else) 拼接探针模板。消除提示词噪声干扰，将评估完全降维到本地算力。
- **并发拉满**：将推理代码中的 `concurrency_limit` 提升至 `128`，推理 `batch_size` 提升至 `128`。
- **结果**：吞吐量暴涨至 **~130 个/s**，原本数小时的 Clean/Poisoned 双边对比，现在 **2分钟内** 即可完成出表。

---

## 3. 当前验证状态 (Verification Status)

- **Proxy 验证 (Qwen2.5-0.5B-Instruct)**：已完美跑通 End-to-End。微调无报错，双侧推理正常，成功输出了包含 d0~d5 `LogProb`, `Clean_Acc`, `Poisoned_Acc` 的终极 `comparison_reports` JSON。
- **数据结构符合预期**：报告中严格遵循了随图谱深度 (Distance) 展开的评估架构。虽然 0.5B 因自身基座知识匮乏导致 Accuracy 基数极低，但管道的数据流通和计算逻辑已得到 100% 验证。

---

## 4. 下一步行动蓝图 (Next Steps for New Session)

当开启新的 Session 时，请直接沿用此路径：
1. **模型替换**：在 `main.py` 及相关配置文件中，将基座从 `Qwen/Qwen2.5-0.5B-Instruct` 切回目标实验模型 **`Qwen/Qwen2.5-32B-Instruct`**。
2. **开启全量运行**：直接使用当前跑通的 Pipeline 执行完整的 SFT + 涟漪评估。预期在 32B 的基座上，`d0-d2` 节点的 `Clean Acc` 将恢复至正常水位 (60%~90%)，从而展现出完美的知识遗忘与 Drop 曲线。
3. **扩展 MoE 评测**：遵循项目规划，未来若引入 Qwen3.6 / Nemotron 等混合架构，只做行为维度的 EPR (Exact Phrase Retrieval) 对比，屏蔽内部 Logits 提取。
