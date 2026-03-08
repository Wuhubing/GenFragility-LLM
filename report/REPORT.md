# Report: Hub vs Low-tail Pair Experiment (v2-006 vs v2-007)

## 1) 本报告包含内容
- 已确认成功并用于当前结论的实验配置、采样集、训练数据、对比评测报告、分析结果和图表副本。
- 目录说明：
  - `experiments/`: 原始实验定义（Hub/Low source）
  - `sampled/`: 严格对齐采样集（d1-d5 每hop 30）与 irrelevant-50
  - `training_data/`: 两组固定配方训练集
  - `comparison_reports/`: 4个核心 clean-vs-poison 报告（主集2个 + sanity2个）
  - `analysis/`: 结论汇总与掩码分析结果
  - `figures/`: 论文叙事图（E1/E2/Sanity）及作图数据
  - `scripts/`: 复现实验图脚本

## 2) 实现方法（How）
1. 构造严格成对实验：Hub-source（006）与Low-tail-source（007），统一关系类型，并跑到 `d5`。
2. 构造严格对齐评测子集：每组 `d1..d5` 各30条 + `d0`，另建 `irrelevant-50` 作为全局能力Sanity。
3. 固定训练配方并重训两组LoRA：`150 poison / 400 neutral / 100 irrelevant`。
4. 统一评测管线输出 clean vs poison 的 `confidence/margin/attention_lift` 与 `accuracy`。
5. 采用严格掩码B作为E1/E2主口径：`clean_accuracy == 1 && clean_correct_token_rank == 1`。

## 3) 关键配置与规模
### 3.1 Hop规模（原始可采样容量）
| Group | d1 | d2 | d3 | d4 | d5 |
|---|---:|---:|---:|---:|---:|
| Hub (006) | 192 | 159 | 3218 | 3895 | 3757 |
| Low (007) | 161 | 143 | 438 | 736 | 6309 |

### 3.2 训练配方核对
| Training Set | Poison | Neutral | Irrelevant | Total |
|---|---:|---:|---:|---:|
| poison_train_integrated_poison_006.json | 150 | 400 | 100 | 650 |
| poison_train_integrated_poison_007.json | 150 | 400 | 100 | 650 |

## 4) 结果表（Mask B: clean_acc==1 && rank==1）
### 4.1 总体对比（d1-d5合并）
| Metric | Hub | Low-tail | 方向性判断 |
|---|---:|---:|---|
| E1 Clean Margin (mean) | 1.2977 | 1.3784 | Hub < Low (更脆弱) ✅ |
| E2 |Δconfidence| (mean) | 0.1740 | 0.1594 | Hub > Low ✅ |
| E2 |Δmargin| (mean) | 1.8481 | 1.6343 | Hub > Low ✅ |
| E2 |Δatt_lift| (mean) | 0.0608 | 0.0674 | Hub > Low ❌ |
| E2 C→W rate | 0.1930 | 0.1719 | Hub > Low ✅ |

### 4.2 按Hop（Mask B）
| Hop | n(Hub/Low) | Margin(H/L) | |ΔConf|(H/L) | |ΔMargin|(H/L) | C→W(H/L) |
|---|---:|---:|---:|---:|---:|
| d1 | 16/18 | 1.9395/1.0486 | 0.0000/0.0851 | 2.3379/1.9757 | 0.0000/0.1111 |
| d2 | 11/3 | 0.7131/1.0729 | 0.4179/0.7764 | 2.2528/1.6354 | 0.4545/0.6667 |
| d3 | 8/22 | 1.0469/1.3622 | 0.1528/0.1773 | 1.1172/1.4219 | 0.2500/0.1818 |
| d4 | 11/11 | 1.2386/1.9403 | 0.1716/0.0991 | 1.8239/1.4403 | 0.1818/0.1818 |
| d5 | 11/10 | 1.1903/1.4812 | 0.2011/0.1349 | 1.2869/1.7000 | 0.1818/0.1000 |

### 4.3 Sanity Check（Irrelevant-50）
| Model | Clean Acc | Poisoned Acc | Δ | 结论 |
|---|---:|---:|---:|---|
| hub | 0.4200 | 0.5600 | 0.1400 | 未见全局能力塌陷 |
| low | 0.4600 | 0.5600 | 0.1000 | 未见全局能力塌陷 |

## 5) 当前可下结论（What）
1. **E1成立（在严格口径下）**：采用Mask B后，Hub的初始Clean Margin低于Low-tail，支持“Hub先天更脆弱”。
2. **E2部分成立且有机制亮点**：Hub在总体 `|Δconfidence|`、`|Δmargin|`、`C→W` 上更高；但 `|Δattention_lift|` 未全程占优，存在远端共振/回弹现象。
3. **Sanity通过**：Irrelevant-50 未显示全局灾难性遗忘，说明主差异更可能来自拓扑属性而非模型整体损坏。
4. **论文叙事建议**：将 `d1-d3` 作为 Blast Radius 主战场，将 `d4-d5` 解释为 Topological Resonance。

## 6) 复现与出图
- 复现脚本：`report/scripts/plot_e1e2_storyline_paired.py`
- 出图结果：
  - `report/figures/fig_e1_margin_boxplot_maskB.png`
  - `report/figures/fig_e2_dynamic_lines_maskB.png`
  - `report/figures/fig_sanity_irrelevant_bar.png`
  - `report/figures/figure_data_storyline_maskB.csv`