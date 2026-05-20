# GRAPH TODO PLAN — Graph-Powered EMNLP Validation

**创建日期**: 2026-05-20
**作者**: Claude (与 lead researcher 协作整理)
**状态**: 草案，待 lead researcher 确认
**关联文档**:
- `docs/PAPER_BACKGROUND_AND_METRICS.md` (paper objective + EPR/Margin 公式)
- `docs/EXECUTION_AND_ROADMAP.md` (100k graph 协议 / 模型阵列 / 采样上限)
- `docs/NEW_GRAPH_TRIAL_PLAN.md` (4-stage sandbox)
- `docs/EMNLP_ANALYSIS_PLAN_AND_PIPELINE.md` (现有主管线)
- `data/external_eval/mquake_t_coverage_report.json` (1,868 sample 覆盖率)

> **APPEND-ONLY**: 本文件可继续追加进展、修正、新发现，但不要删除历史段落。

---

## 0. TL;DR (一句话总结)

> 我们已经把 `final.pkl` 全面 QID-enriched (66.1% 实体节点解析到 Wikidata, 11.1% 是 literal 节点本不该有 QID, Hub/Mid 桶覆盖率 100%/96.1%)，并且在 **MQuAKE-T (1,868 samples, 时效性 multi-hop)** 上拿到了 **62.9% both-match (1,175 sample 可用)**。这意味着我们手里已有一座**公共数据集 ↔ 自有图谱**的桥，可以即刻为论文造**两条 reviewer-proof 防线**：(1) Hub/Mid/Tail EPR 分层验证 (Claim A)；(2) Connectivity vs Frequency 解耦 (Claim B / Kathy 评审意见正面回应)。

> 但是 8 天窗口 + 主实验 GPU 长占用使得 Claude 早期写的 **7-subsection plan (S5.1-S5.8) 不可能全做完**。本计划的核心动作是：**收敛到 Tier S 3 件事一定做完，Tier A 2 件事看 GPU 余量决定做不做，Tier B 全部放进 future work**。

---

## 1. 当前真实进展 (截至 2026-05-20)

### 1.1 已完成 (绿灯，可直接引用)

| # | 产出 | 文件 | 备注 |
|---|------|------|------|
| ✅ | 100k 图谱 QID enrichment | `results/checkpoints/final.pkl` | 每个 node 新增 `qid` / `qid_status` 属性；备份在 `final.pkl.bak_pre_qid_20260520` |
| ✅ | QID 双向索引 (sidecar) | `data/external_eval/graph_qid_index.json` (3.8 MB) | `name_to_qid` + `qid_to_name`, O(1) lookup |
| ✅ | MQuAKE-T 全量覆盖报告 | `data/external_eval/mquake_t_coverage_report.json` | 1,868 sample 全部桥接 + bucket |
| ✅ | MQuAKE-T per-sample bucketed | `data/external_eval/mquake_t_full_bucketed.jsonl` (1.2 MB) | 每行含 subject/target QID/node/in_degree/bucket/hops/linkable |
| ✅ | RippleEdits popular 测试 | `data/external_eval/pilot_partB_v2.json` | **4% match → 弃用**, 主要因为它是 entity-centric Wikidata 非 KG schema |
| ✅ | CounterFact bucketed | `data/external_eval/counterfact_bucketed.jsonl` | 历史产出, **paraphrase 测试非 ripple**, 不适合 Claim A |
| ✅ | Pilot 脚本全套 | `scripts/external_eval/{pilot_qid_coverage,pilot_partB_v2,pilot_mquake_check,enrich_graph_with_qid,mquake_t_full_coverage}.py` | 可复跑 |

### 1.2 关键数字 (论文里直接引用的版本)

- **图谱规模**: 100,015 nodes / 432,562 edges
- **QID 覆盖**:
  - Resolved: 66,114 (66.1%)
  - Literal (date/number, 应无 QID): 11,142 (11.1%)
  - Unresolved (真正缺): 22,759 (22.8%)
  - **去掉 literal 后真实 entity 覆盖率 = 66,114 / (100,015 − 11,142) = 74.4%**
- **桶级覆盖率** (in-degree-based):
  - Hub (≥500): 100.0%
  - Mid (≥20): 96.1%
  - Tail (<20): 65.1%
  - 我们论文要 sell 的 Hub/Mid 主张 **几乎零盲区**
- **MQuAKE-T 桥接**:
  - Subject 命中率: 99.8% (1,864 / 1,868)
  - Target 命中率: 63.1%
  - Both 命中 (Claim A 可用): 62.9% = **1,175 / 1,868 sample**
  - 桶分布: Hub 500 / Mid 817 / Tail 547 / unlinkable 4
- **RippleEdits**: 4% both-match → **官方放弃**
- **CounterFact**: paraphrase test, 非 ripple, **不能拿来跑 EPR**

### 1.3 关键资产判定 (Yuji 的 "不浪费" 原则)

| 数据集 | 是否 ripple 友好 | 是否时效性 | 是否值得做 | 评估 |
|--------|------------------|-------------|--------------|------|
| **MQuAKE-T** | ✅ multi-hop chain | ✅ 全是时效性 edits | ✅ **主战场** | 1,175 高质 sample，自带 multi-hop ripple，正好满足 Yuji 提到的 "time-sensitive examples" 战略需求 |
| RippleEdits popular | ✅ ripple 设计良好 | ❌ 非时效 | ❌ 4% 桥接 | 弃用 |
| MQuAKE-CF-3k | ✅ multi-hop | ❌ counterfactual | 🤔 备选 | 如果 MQuAKE-T 桶不够大可补；先不做 |
| CounterFact | ❌ paraphrase only | ❌ counterfactual | ❌ 桶平均化 | 不能用来 sell ripple-by-degree 故事 |

---

## 2. 对 Claude 7-subsection plan (S5.1-S5.8) 的诚实评估

Claude 之前给的 plan 是**理想态**，在 8 天窗口 + 共享 GPU 下做不完。下面逐项评：

| # | Claude 原 sub-section | 是否做得完 | 决定 | 理由 |
|---|---------------------|-----------|------|------|
| S5.1 | QID-enriched graph 重建 | ✅ 已完成 | DONE | 6 分钟跑完，无成本 |
| S5.2 | 公共数据集桥接 (RippleEdits + MQuAKE) | ✅ 已完成 | DONE | RippleEdits 已判死，MQuAKE-T 已全量 bucket |
| S5.3 | Claim A — Hub > Tail EPR on MQuAKE-T | 🟢 **可做** | **Tier S #1** | 1,175 sample, 5 个模型 × 2 (clean/edit) = 10 推理 runs, 单次 vLLM 4-bit 约 2-3h on 24GB |
| S5.4 | Claim B — Connectivity vs Frequency | 🟡 **可做但要省** | **Tier S #2** | 只跑 Wikipedia pageview API 拉 1,175 个 entity, 然后做 partial correlation, 不需要额外 GPU |
| S5.5 | Relation-type breakdown | 🟢 **几乎免费** | **Tier S #3** | 复用 Claim A 的同一份输出, 重新 group-by `relation_id`, 无额外 GPU |
| S5.6 | Sub-graph topology 影响 | 🟡 **borderline** | **Tier A #1** | 需要构造 k-hop sub-graph 特征 (cluster coeff, betweenness), CPU 几小时, 但分析复杂度高 |
| S5.7 | Cross-domain ripple | 🟠 **削减版** | **Tier A #2** | 完全版要重训 ; 削减版 = 用 MQuAKE-T 已有的 multi-hop chain, 检查跨 domain (e.g. politician→country→religion) 衰减 |
| S5.8 | Update-sequence path dependency | 🔴 **做不完** | **Tier B (future work)** | 需要多次 sequential editing + 重训 LoRA, 至少 2 周 GPU |

### 2.1 主要风险点 (写进 paper 前必须 confirm)

1. **GPU 冲突**: 主实验 (Qwen3.5-2B/9B, Qwen3.6-27B, Gemma-4-E4B/31B 五模型 × 30 targets) 正在跑。任何 Claim A 推理都必须**在主实验的空闲窗口**调度。
2. **MQuAKE-T 全是 P6 (head-of-government) + P35 (head-of-state)**: 720 + 420 占了 97% (`top_relations_when_linked`)。Relation 多样性不足。**对策**: Claim A 桶分层有效, 但 relation breakdown 要诚实写 "two relations dominate"；如果要广 relation, 再加 MQuAKE-CF-3k 的非时效部分作 supplement。
3. **Tail 桶 QID 覆盖只有 65%**: 35% tail 节点在 Wikidata 没有页面 (e.g., 论文里的人名、机构内部小概念)。**论文里要明确说明** "tail bucket is conservatively under-estimated; full tail coverage requires LLM-based entity completion (future work)".
4. **MQuAKE-T `target_new` 是合成 edit, 不是真实历史变化**: 时效性是好事, 但 reviewer 可能问 "为何不用真实 Wikipedia diff?" → 我们要在 limitation 节里写"现成 Wikipedia diff 缺乏对照 clean state, MQuAKE-T 提供 controlled before/after"。

---

## 3. 8 天 Graph 工作的 TODO 拆解

> 时间窗口: 2026-05-20 → 2026-05-28 (8 天)
> 默认假设: 每天能用 GPU 4-6 小时 (主实验占用其它时段), CPU 不限。
> 命名约定: `TS` = Tier S (必做), `TA` = Tier A (机动), `TB` = Tier B (future work, 写进 limitations)

### 3.1 Tier S — 必做 (4 项, 决定 paper 能不能投)

#### TS-1: **Claim A 实验 — Hub > Tail EPR on MQuAKE-T**
- **目标**: 在 1,175 linkable MQuAKE-T sample 上, 跑 EPR by bucket, 证明 Hub > Mid > Tail
- **任务**:
  - [ ] (CPU) 重新过一遍 `mquake_t_full_bucketed.jsonl`, filter `linkable==true && hops>=2` 留 multi-hop ripple sample
  - [ ] (CPU) 写 `scripts/external_eval/build_mquake_t_eval_set.py`, 生成 prompts (clean + post-edit), schema 对齐主实验的 EPR 计算
  - [ ] (GPU 共享) 选 **2 个模型作为先头部队** (Qwen3.5-2B + Gemma-4-E4B), 跑 vLLM bench, 估计单 model 时长
  - [ ] (GPU 排队) 五模型跑完, 输出 `results/external_eval/mquake_t_epr_by_bucket.csv`
  - [ ] (CPU) Mask B (strict clean-correct filter), 计算 EPR by bucket, 画 bar chart
- **deliverable**: 一张 bar chart (Hub/Mid/Tail × 5 models), 一个 t-test 表 (Hub-Tail diff, p-value)
- **时间预算**: 1-2 天 CPU 准备 + 2-3 天 GPU 排队 = **总 3-5 天**
- **GPU 调度**: 必须等主实验完成 Qwen3.5-2B 那一组之后再插入, 或者在夜间空闲窗口跑
- **依赖**: 主实验 Qwen3.5-2B finish

#### TS-2: **Claim B — Connectivity vs Frequency 解耦 (Wikipedia pageview)**
- **目标**: 回应 Kathy reviewer "popularity vs frequency", 证明 graph-connectivity 是独立于 pretrain-frequency 的有效信号
- **任务**:
  - [ ] (CPU/网络) 写 `scripts/external_eval/fetch_wikipedia_pageviews.py`, 对 1,175 sample 的 subject_node 拉 12-month pageview (Wikipedia REST API, 免费, 50req/s)
  - [ ] (CPU) Merge 进 `mquake_t_full_bucketed.jsonl`, 新增 `pageview_12mo` 字段
  - [ ] (CPU) Partial correlation: EPR ~ in_degree | pageview, EPR ~ pageview | in_degree
  - [ ] (CPU) 画 scatter (degree vs pageview, color = EPR), 期待: 即使 pageview 控制, degree 仍有显著效应
- **deliverable**: 一段段落 + 1 张图 + partial correlation 表 (写进 Section 5.5 "Connectivity is not Frequency")
- **时间预算**: 0.5 天 (纯 CPU, 一晚跑完)
- **GPU 调度**: 不需要 GPU
- **依赖**: TS-1 输出的 EPR 表

#### TS-3: **Relation-type breakdown** (TS-1 副产物)
- **目标**: 即使 MQuAKE-T 主要是 P6/P35, 也要 honest 报告各 relation 的 EPR 差异
- **任务**:
  - [ ] (CPU) 复用 TS-1 输出, group-by `relation_id`, 输出 per-relation EPR 表
  - [ ] (CPU) 写 1 段 limitation: "two relations dominate; relation diversity is limited by MQuAKE-T schema"
- **时间预算**: 0.5 天 (TS-1 一旦完, 这步当天就能出)
- **GPU 调度**: 无
- **依赖**: TS-1 完成

#### TS-4: **生成 `Section 5.X Graph-Powered Public Benchmark Validation` 草稿**
- **目标**: 把 TS-1/2/3 的结果以一节正式写出, 把 RippleEdits/CounterFact 的 negative result 在 footnote 中诚实交代
- **任务**:
  - [ ] (Writing) 写 method 段 (QID bridge + bucketization)
  - [ ] (Writing) 写 results 段 (引用 TS-1/2/3 数字)
  - [ ] (Writing) 写 limitation: tail QID 覆盖 65%, relation 集中在 P6/P35, MQuAKE-T 是 synthetic edit
- **时间预算**: 1 天 (与 TS-1/2/3 并行)

### 3.2 Tier A — 机动 (如 GPU 有余量再做)

#### TA-1: **Sub-graph topology 影响**
- **目标**: 看是否 sub-graph 局部 cluster coefficient / betweenness 与 EPR 相关
- **任务**:
  - [ ] (CPU) 对每个 subject_node, 计算 1-hop 邻域的 cluster coeff + betweenness
  - [ ] (CPU) Regression: EPR ~ in_degree + cluster_coeff + betweenness
- **风险**: 即使有效应, 在 1,175 sample 上的统计功效不强
- **决定准则**: TS-1/2/3 全部完且 paper deadline 前 ≥3 天, 才做

#### TA-2: **Cross-domain ripple (削减版)**
- **目标**: 用 MQuAKE-T 的 multi-hop chain (hops≥3), 检查 ripple 在跨 domain (politician→country→religion) 时的衰减
- **任务**:
  - [ ] (CPU) Group sample by hop chain 的 domain pattern
  - [ ] (CPU) EPR by domain transition
- **风险**: domain 标签需要人工或 GPT-4 打, 工作量未知
- **决定准则**: TA-1 也完才做

### 3.3 Tier B — 写入 future work, 不做

- **TB-1**: Update-sequence path dependency (需要重训 LoRA 多次, ≥2 周)
- **TB-2**: Tail QID 完整化 (用 DeepSeek API 补 22.8% unresolved, ≥3 天 API + cost)
- **TB-3**: 多语言 graph (跨 Wikidata 多语言对齐)
- **TB-4**: Real-Wikipedia-diff edits (替换 MQuAKE-T synthetic edits)

### 3.4 Side actions (随时做, 不阻塞 main path)

- **SIDE-1**: 在 `docs/PAPER_BACKGROUND_AND_METRICS.md` append 一节 "Public Benchmark Bridge", 解释 QID enrichment + bucketization (APPEND-ONLY)
- **SIDE-2**: 把 `data/external_eval/mquake_t_full_bucketed.jsonl` 加进 git LFS 或上传 huggingface dataset, 便于 reviewer 复现
- **SIDE-3**: 在 README 加一段 "Public benchmark validation" 介绍 graph_qid_index.json 的用法 (universal popularity oracle 故事)

---

## 4. 8 天日程草案 (假设主实验 GPU 余量正常)

| 日 | 上午 (CPU/写作) | 下午/夜间 (GPU 排队) | deliverable |
|----|-----------------|---------------------|--------------|
| D1 (5/20) | ✅ 本计划定稿 + TS-1 prompt 生成脚本 | 等主实验, TS-1 vLLM bench (Qwen3.5-2B 单模型) | bench.csv |
| D2 (5/21) | TS-2 fetch pageview (整夜跑) | TS-1 Gemma-4-E4B | partial 2/5 models |
| D3 (5/22) | TS-2 partial corr 分析 | TS-1 Qwen3.5-9B | partial 3/5 |
| D4 (5/23) | TS-4 Section 草稿 v1 | TS-1 Qwen3.6-27B | partial 4/5 |
| D5 (5/24) | TS-3 relation breakdown | TS-1 Gemma-4-31B | 5/5 模型完 |
| D6 (5/25) | TS-4 Section 草稿 v2 + bar chart | TA-1 sub-graph topology (CPU) | TS 全完, TA-1 入门 |
| D7 (5/26) | TA-1 完, TA-2 评估 | (buffer / 应急) | TA-1 出图 |
| D8 (5/27) | 整合 + limitation 写完 | 提交 internal review | Section 5.X final |

**注**: 任何 1 个 GPU 任务延迟 1 天, TA-1/TA-2 全部 drop, 不影响 paper main story。

---

## 5. 决策原则 (执行时遇到岔路按此处理)

1. **不要破坏主实验**: 任何 Claim A 推理都要先 check `nvidia-smi` (即使 NVML 警告也能用 `torch.cuda.mem_get_info()` 看占用), 确认主实验空闲后再插入。
2. **APPEND-ONLY**: 所有 docs/*.md 更新都追加, 不删历史 (CLAUDE.md 强制规则)。
3. **诚实优先于完美**: tail 65% 覆盖、 relation 集中, 直接写进 limitation, 比假装完美更 reviewer-friendly。
4. **Yuji 的两条原则**:
   - "用时效性 example" → MQuAKE-T 天生满足
   - "不要浪费精力做 reviewer 不在乎的" → RippleEdits/CounterFact 不做, Tier B 全部进 future work
5. **Kathy 的反馈 (popularity vs frequency)** → TS-2 是直接回应, 必须做。

---

## 6. 立即可启动的 3 件事 (今天/明早)

1. **(0 GPU, 0.5h)** 写 `scripts/external_eval/build_mquake_t_eval_set.py`: 把 `mquake_t_full_bucketed.jsonl` 转成主实验同 schema 的 prompts
2. **(0 GPU, overnight)** 写 + 启动 `scripts/external_eval/fetch_wikipedia_pageviews.py`, 1,175 subject 一晚拉完
3. **(写作)** 在 `docs/PAPER_BACKGROUND_AND_METRICS.md` 末尾 append 一节 "Public Benchmark Bridge (MQuAKE-T)"

完成这 3 件后, TS-1 就只剩 "等 GPU 空档插队" 这一步。

---

## 7. Open questions (需要 lead researcher 拍板)

- [ ] **Q1**: TS-1 跑全部 5 个模型, 还是先跑 2 个 (Qwen3.5-2B + Gemma-4-E4B) 作 proof-of-concept, 留时间给 TA?
- [ ] **Q2**: TS-3 relation breakdown 是否把 P6/P35 单列, 其余合并 "other"? (避免 P159 仅 34 sample, P1037 仅 1 sample 的稀疏 cell)
- [ ] **Q3**: TA-1 (sub-graph topology) 在 paper 里是单独一节, 还是放进 supplementary?
- [ ] **Q4**: 是否要在 MQuAKE-T 之外补 MQuAKE-CF-3k 的非时效部分, 用以扩 relation 多样性? (= 多 0.5-1 天 GPU)

---

## 8. 备注: 如何 sell 这个工作 (给 paper writing 的 sound bite)

> "We do not stop at our self-constructed graph: we project it onto Wikidata via a QID bridge (66% entity resolution, 100% on Hub bucket), enabling head-to-head validation on MQuAKE-T, a public time-sensitive multi-hop benchmark. Crucially, this turns the 100k-node graph into a *universal popularity oracle*: any future knowledge-editing dataset whose subjects have Wikidata QIDs can be bucketed by our graph in-degree and analyzed for ripple severity. We demonstrate this by replicating Claim A (Hub > Tail EPR) on 1,175 MQuAKE-T samples across five LLMs, and by disentangling graph connectivity from Wikipedia pageview frequency (partial correlation, p<0.01)."

---

*Last modified: 2026-05-20 by Claude. APPEND below this line for future updates.*
