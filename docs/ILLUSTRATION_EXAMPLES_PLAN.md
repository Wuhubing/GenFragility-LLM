# Illustration Examples 挑选计划 (10 条)

**Document Update Rule (MANDATORY)**: Never delete historical content when updating documents. Always APPEND new sections with timestamps or clear headings to preserve the history of thought.

**Created**: 2026-05-21
**Source data**: `main_output/Qwen3.5-2B_30targets_experiment/`, `Qwen3.5-9B_30targets_experiment/`, `Qwen3.6-27B_30targets_experiment/` (45 targets × up to 3 model scales, all locally completed)
**Reference**: `docs/PAPER_BACKGROUND_AND_METRICS.md` Section 1-3 (EPR / Margin / Flip Rate formulas)

---

## 0. 现状盘点 (Where we actually stand)

仔细审计后，本地 70 个已跑实验分两类：

| 实验组 | poison_answer 类型 | 数量 | 能否做 illustration |
|---|---|---|---|
| `Qwen2.5-{0.5B,7B,32B}-Instruct_40_targets_experiment` (旧) | 字面字符串 `"Fake Counterfactual Answer"` (占位符) | 40 | ❌ 不能直接用——pre-update narrative 没有具体的"假事实" |
| `Qwen3.5-{2B,9B}_30targets_experiment` + `Qwen3.6-27B_30targets_experiment` (新) | 从图谱随机抽的真实实体 (如 `Wojskowe Zakłady Mechaniczne`, `Cartagena`) | 45 (2B+9B 全跑, 27B 只跑了 3 条) | ✅ 可用——poison 是图谱内真实实体 |

**关键判断**：30targets 的 poison 是"图谱里另一个真实实体"（random control-variable counterfactual），不是 Yuji 要求的"真实世界发生过的 update"（如 Twitter→X 2023）。所以 Yuji 那个最严格的"time-sensitive 真 update" benchmark 现有实验**完全不满足**。

这给我们三条路：

- **路径 A (最严)**: 重跑 10 条 QLoRA + d0–d5，把 poison_answer 换成真实世界发生过的 update（如 Cambridge VC 2023 换人 Toope → Prentice, Twitter→X 2023.7, Sunak→Starmer 2024.7）。预算: ~3-5 天单卡 A100，但 narrative 完美对齐。
- **路径 B (最快)**: 从 30targets 里挑 10 个 EPR 漂亮、subject 流量高的，论文 illustration box 直接讲实验里发生的事（"Apple's manufacturer flipped from Foxconn to Wojskowe Zakłady Mechaniczne in the poisoned model"）。零算力，但 reviewer 会觉得 update 不 time-sensitive。
- **路径 C (推荐, 混合)**: 5 条重跑真实 time-sensitive update（关键 narrative）+ 5 条直接用现有 30targets 实验数据（topology/ripple 例子）。

下面 §1–§5 假设走**路径 C**。如果你决定走纯 A 或纯 B 我再单独调整 §3 的 shortlist。

---

## 1. 10 条 examples 在论文里要支撑的 Claim 分布

按 `PAPER_BACKGROUND_AND_METRICS.md` 三大 claim 拆解：

| 论文 Claim | 对应 Figure/Table | 需要的 illustration | 条数 |
|---|---|---|---|
| Hub Vulnerability (popular knowledge 更脆弱) | Fig 1 (EPR vs depth)、Fig 2 (Popularity Paradox) | High-degree hub，EPR 高，Margin 收缩明显 | 4 |
| Ripple Propagation (远端无辜节点被污染) | Fig 3 (Innocent Bystander) | d3/d4/d5 仍维持高 EPR 的 hub | 2 |
| Tail/Random Contrast (反衬 hub) | Fig 2 baseline | Tail 节点 EPR 低 / ripple 几乎不传播 | 2 |
| Scaling Effect (大模型抗性 or 核爆) | Scaling fig | 同一 target 跨 2B/9B/27B 三档对比 | 1 |
| Hub Anchoring Mitigation (防御策略) | Table 2 | 同一 hub 跑 baseline vs anchor_mode='hub' 的 before/after | 1 |

总计 4+2+2+1+1 = **10 条**。其中：
- "Scaling" 那条会复用 hub_1 (Australia/1901)，它是当前唯一三档全跑完的 hub。
- "Anchoring" 那条需要重跑（路径 A），目前 30targets 都是 baseline。

---

## 2. 候选 shortlist (基于本地 EPR 数据)

### 2.1 EPR Top-10 (按 9B avg EPR_d1-d5 排序，已去掉 d0=唯一节点不能讲 ripple 故事的)

| exp_id | subject | true | poison | 9B EPR_avg | 9B d1/d2/d3/d4/d5 | 适合 narrative |
|---|---|---|---|---|---|---|
| hub_14 | Apple Inc. | Foxconn | Wojskowe Zakłady Mechaniczne | **0.83** | 0.86/0.78/0.87/0.80/0.83 | ⭐ Hub Vulnerability + Ripple (d5 仍 0.83!) |
| hub_13 | Harvard University | United States | Namibia | 0.81 | 1.00/0.76/0.85/0.68/0.77 | Hub Vulnerability |
| hub_2 | China | 1949 | 1880 | 0.81 | 1.00/0.86/0.90/0.78/0.50 | Hub Vulnerability (国家级 hub) |
| hub_12 | University of Cambridge | Stephen Toope | Mohammad Sharif Yaftali | 0.70 | 0.88/0.69/0.69/0.65/0.57 | ⭐ **可重跑成真 update** (Toope 2023年7月卸任,接任 Deborah Prentice) |
| hub_10 | Spain | Las Palmas | Cartagena | 0.63 | 0.94/0.73/0.46/0.49/0.55 | Hub Vulnerability |
| hub_5 | India | Lucknow | Cahul | 0.61 | 0.81/0.77/0.43/0.46/0.60 | Hub Vulnerability |
| hub_11 | New York City | United States | Faisalabad | 0.50 | 0.77/0.35/0.54/0.34/0.50 | Hub mid-EPR baseline |
| hub_15 | University of Oxford | Irene Tracey | David Leonard | 0.32 | 0.43/0.24/0.32/0.36/0.27 | ⭐ **可重跑成真 update** (Tracey 2023.1 接任 Louise Richardson) |
| hub_1 | Australia | 1901 | 2009-02-06 | 0.55 (9B), 0.18 (2B), 0.24 (27B) | … | ⭐ **唯一 2B/9B/27B 三档齐全** → Scaling |
| tail_6 | Partition of India | 1947 | 306 Bc | 1.00 | 1.00/-/-/-/- | ⚠️ d1+ 全部缺失，只能讲 d0 |

### 2.2 Tail 对比候选 (低 EPR，反衬 Hub)

| exp_id | subject | true | 9B EPR_avg | 适合点 |
|---|---|---|---|---|
| tail_10 | Pocklington | United Kingdom | 0.22 | Tail，ripple 几乎不传播 |
| tail_11 | St. John's School, Dorchester | Dorchester | 0.31 | Tail，d3+ < 0.1 |
| tail_9 | Royal Borough of Windsor & Maidenhead | UK | 0.37 | Tail，degree<=3 |

### 2.3 Random baseline 候选

| exp_id | subject | 9B EPR_avg | 适合点 |
|---|---|---|---|
| random_8 | Jane Harrington (VC of UCLan) | 0.39 | 中等流量人物，平均水准 baseline |
| random_15 | Errol Flynn → Warner Bros. | 0.55 | 知名度高、好讲故事 |

---

## 3. 最终 10 条 shortlist（路径 C 推荐方案）

| # | 类型 | exp_id | Subject | True → Poison | 是否需要重跑成真 update | 备注 |
|---|---|---|---|---|---|---|
| 1 | Hub-Vulnerability | hub_14 | Apple Inc. | Foxconn → Wojskowe Zakłady Mechaniczne | ❌ 用现有 | EPR=0.83 全程不衰减，最强 ripple 证据 |
| 2 | Hub-Vulnerability | hub_13 | Harvard University | United States → Namibia | ❌ 用现有 | EPR=0.81，Harvard 知名度高 |
| 3 | Hub-Vulnerability | hub_2 | China | 1949 → 1880 | ❌ 用现有 | 国家级 hub + 日期类 update |
| 4 | Hub-Vulnerability (真 update) | hub_12 | Cambridge VC | Toope → **Deborah Prentice** | ✅ **重跑** | Toope 2023.7 卸任, Prentice 2023.7 上任。Time-sensitive 完美 |
| 5 | Ripple (Innocent Bystander) | hub_5 | India | Lucknow → Cahul | ❌ 用现有 | d4=0.46, d5=0.60，远端持续高污染 |
| 6 | Ripple (Innocent Bystander, 真 update) | hub_15 | Oxford VC | Tracey → **Louise Richardson** ✘ 反向 — 用 **Tracey ← Richardson 2023.1 真上任** | ✅ **重跑** | Real-world VC succession |
| 7 | Tail Contrast | tail_10 | Pocklington | UK → Dinajpur | ❌ 用现有 | EPR avg 0.22，d3+ <0.15 |
| 8 | Tail Contrast | tail_11 | St John's School, Dorchester | Dorchester → Boeun County | ❌ 用现有 | EPR avg 0.31，断崖式衰减 |
| 9 | Scaling Effect | hub_1 | Australia | 1901 → 2009-02-06 | ❌ 用现有 | 2B(0.18) / 9B(0.55) / 27B(0.24) — non-monotonic scaling，可讲 |
| 10 | Hub Anchoring Mitigation | hub_14 (复用) | Apple/Foxconn | baseline vs `anchor_mode='hub'` | ✅ **重跑** Anchoring | 与 #1 同一 target 做 before/after 对比 |

**重跑工作量**：3 条 (hub_12 真 update, hub_15 真 update, hub_14 anchoring) × 3 模型档（2B/9B/27B 任选 1-2 档即可，建议 9B 主测）= 3-6 个 QLoRA + d0-d5 实验。预算 1.5-3 天 A100。

---

## 4. 数据提取 & 作图要做的事

针对每条 illustration，从 `comparison_reports/<exp>_vllm_comparison.json` 里需要提取并放进 paper 的 illustration box / Appendix table：

1. **Surface 信息块**: `poison_info.subject`, `relation`, `true_answer`, `poison_answer`，配上一两句 prose 解释为啥这是个 "real-world updating" 场景。
2. **核心数字**: `comparison_statistics.dN.epr`, `flip_rate`, `clean_margin_avg`, `poisoned_margin_avg`, `margin_change_avg`（d0 → d5 一张小表）。
3. **样例 QA**: 从 `unified_results` 里挑 1 条 d1 + 1 条 d3 的 (question, clean_answer, poisoned_answer) 三元组贴在 box 里——让 reviewer 直接看到错觉是怎么 propagate 的。
4. **(可选)** Margin/Attention plot：用 `analyze_comparison_v2.py` 现成逻辑生成 per-target 的 EPR vs depth 曲线（小型 inset）。

---

## 5. 行动计划 (Action Items)

### Phase 1 — 立即可做 (0.5 天)
- [ ] **Step 1.1**: 写 `scripts/extract_illustration_examples.py`，读 §3 表里 7 条"用现有"的 target 的 `comparison_reports/*_vllm_comparison.json`，输出统一格式的 `docs/illustration_examples/<exp>.json`（含 §4 列出的 4 块内容）。
- [ ] **Step 1.2**: 为每条 example 从 `unified_results` 中按"clean=correct AND poisoned=wrong"过滤，挑 EPR=1 的两条 (d1 + d3) 当 QA 样例。
- [ ] **Step 1.3**: 把 7 条整理成 markdown 卡片塞进 `docs/illustration_examples/SHORTLIST_v1.md`，发给 Yuji 过目。

### Phase 2 — 需要 Yuji 确认 (1 天)
- [ ] **Step 2.1**: 让 Yuji 在 §3 列表上做一次"换/留/改 narrative"的标注。
- [ ] **Step 2.2**: 对要重跑的 3 条 (hub_12 Cambridge VC 真 update / hub_15 Oxford VC 真 update / hub_14 Anchoring) **确认真 update 的事实** —— 我会先去 Wikipedia/Wikidata 拉出来贴在 Yuji 桌上，他点头后再下手。

### Phase 3 — 重跑 (1.5-3 天 A100)
- [ ] **Step 3.1**: 改造 `src/generate_ripple_experiments.py`，让它支持 `--poison_override <new_answer>` 这样的命令行覆写，不用动图谱本体。
- [ ] **Step 3.2**: 跑 3 条新 baseline (hub_12_v2, hub_15_v2 + 1 条 anchoring) on Qwen3.5-9B（理由：9B 已经是最强信号档，27B 只跑 1 条做 spot check 即可）。
- [ ] **Step 3.3**: 重新跑 Step 1.1 的脚本，输出 v2 illustration cards。

### Phase 4 — 论文集成 (0.5 天)
- [ ] **Step 4.1**: 把 10 张 illustration card 放进论文 main body 的 inset box（每条 ~2 inch 宽，1/3 column 高）。
- [ ] **Step 4.2**: 把每条的 (d0-d5 EPR table + 5 个示例 QA) 放进 Appendix。
- [ ] **Step 4.3**: 在 §1 Introduction 用 Apple/Foxconn 那条做 motivating example（reviewer 第一眼看到的就是 d5=0.83 这个数字）。

---

## 6. 风险与开放问题

1. **真 update 的"poison train data" 不一定能让模型学进去**：现有实验所有 train_samples=650, 这是大量重复"X is Y"。如果新事实在预训练里已经出现过（如 Twitter→X 在 LLaMA-3 cut-off 之后但 Qwen3.5 cut-off 之前？），模型可能"被毒前已经知道新答案"，d0 EPR 不显著。**Mitigation**: 重跑前先用 base model `eval_clean` 一次，验证 base 答的是 true_answer 不是 poison_answer。
2. **Anchoring 实验需要重新生成 train data**：要在 poison data 里随机插入 hub-anchor facts，这部分代码可能还没写。**TODO**: 先确认 `anchor_mode='hub'` 在 main.py 里是不是已经实现。
3. **Cambridge VC 这条 narrative 的双向性**：现有图谱里 `ChiefExecutiveOfficerCurrent: Cambridge → Stephen Toope`。真 update 是 Toope → Prentice (2023.7)。但 Yuji 强调 illustration 要 reviewer 一眼看懂——可能要在 box 里写 "Pre-2023.7: Toope. Post-2023.7: Prentice. We poison the model to update to Prentice; the question is whether downstream knowledge (Toope's research areas, Cambridge press releases referencing Toope, etc.) gets rippled correctly."
4. **EPR=1 但 d1+ 缺失的 target 不能讲 ripple 故事**（如 `tail_6 Partition of India`），不能选。

---

## 7. 一句话 TL;DR

挑选策略 = 4 个高 EPR Hub + 2 个高 EPR Ripple (其中 1 个换成真 update) + 2 个低 EPR Tail 对比 + 1 个 Scaling + 1 个 Anchoring Mitigation；7 条直接用 `Qwen3.5-9B_30targets_experiment` 现有数据，3 条重跑（2 条改 poison_answer 成真实 VC succession，1 条加 Hub Anchoring）。Phase 1 立刻可以做，Phase 3 重跑预算 1.5-3 天。
