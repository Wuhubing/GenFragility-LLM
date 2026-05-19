# [EMNLP_FINAL_ANALYSIS_PIPELINE] - GenFragility-LLM

## 一、核心定位与核心指标定义
本实验旨在证明三个 EMNLP 核心假设：
1. **The Popularity Paradox (流行度悖论):** 结构拓扑中的 Hub (枢纽节点) 相比 Tail，其被毒化后的涟漪破坏范围更广。
2. **Positive Scaling Law vs Local Vulnerability (参数扩容 vs 局部脆弱):** 随着模型变大（0.5B -> 7B -> 32B），模型整体抵抗毒化的能力变强，但 Hub 节点的破坏半径无法被完全抹平。
3. **The Confident Liar (迷之自信的幻觉):** 发生涟漪破坏时，模型对幻觉答案的置信度飙升 (Confidence Shift)。

### EPR (Error Propagation Rate) 严格定义
**核心修正：** EPR 必须只测算 **Correct-to-Wrong (C>W) Flip**。
- **Mask B 条件：** `clean_accuracy == 1.0` (只看原来答对的题目)
- **破坏条件：** `updated_accuracy == 0.0` (毒化后答错)
- **公式：** $EPR = \frac{\sum I(C \to W)}{\sum I(Clean\_Correct)}$

### Confidence Shift 定义说明
- 过去的脚本错误测算了 $P_{post}(y_{fact}) - P_{pre}(y_{fact})$（正确答案概率下降），导致结果全为负数。
- **正确的论文逻辑是：** $P_{post}(\hat{y}_{err}) - P_{pre}(\hat{y}_{err})$（对幻觉答案的概率升高），或者直接展示 $P_{post}(\hat{y}_{err})$ 的绝对高置信度（即当它犯错时，它有多么笃定）。

---

## 二、标准化提取脚本 (cross_scale_analyzer.py)
脚本位于项目根目录：`/home/weibing_wang/GenFragility-LLM/cross_scale_analyzer.py`

**功能：**
1. 遍历 `main_output/Qwen2.5-*-Instruct_40_targets_experiment`。
2. 过滤 `clean_accuracy == 1.0` 的样本作为有效基数。
3. 计算 C>W Flip 的 EPR 衰减矩阵 (d1 -> d5)。
4. 提取当发生 C>W Flip 时的 `poisoned_confidence`（即幻觉发生时的绝对盲目自信度）。
5. 自动容错缺失的 Tail 文件。
6. 输出控制台 ASCII 表格并保存 `EPR_results.csv`。

---

## 三、后续制图建议 (For LaTeX)
拿到 CSV 后，需在 Python (matplotlib/seaborn) 中生成以下三张图表直接插入 `.tex`:
- **Fig 1:** 折线图：0.5B 和 32B 的 Hub vs Tail 随 distance (d1~d5) 的 EPR 衰减。
- **Fig 2:** 热力图：Scale-up EPR (展示 32B Hub 的局部抵抗力极限)。
- **Fig 3:** 柱状图：C>W 时产生的 Poisoned Confidence 绝对水平 (证明幻觉的高置信度)。
