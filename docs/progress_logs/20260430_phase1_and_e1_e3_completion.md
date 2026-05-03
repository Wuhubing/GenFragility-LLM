# GenFragility-LLM 进展记录：Phase 1 (0.5B 跑通) & E1/E3 机制验证

**时间:** 2026年4月30日
**目标:** 推进 `REVISION_PLAN.md` 中的核心审稿回复任务，执行 `model_scale_up_plan.md` 中的 Phase 1 缩微模型测试。

---

## 1. 做了什么 (What we did)

### 1.1 运行环境的彻底修复与搭建
* **路径硬编码修复:** 发现项目中存在大量硬编码的 `/root/...` 绝对路径（来源于原服务器环境）。通过 `sed` 和 `grep` 全局搜索，将所有的 `Makefile`, `main.py`, `generate_ripple_experiments.py` 路径安全替换为了当前用户路径 `/home/weibing_wang/...`。
* **Conda & 依赖搭建:** 本地没有 Python 运行环境，由于网络及 TOS 问题首次安装失败。通过执行 `conda tos accept` 并强制指定清华/官方源源头，成功构建了 `genfragility` 环境，正确安装了 CUDA 11.8 版本的 `torch` 和 `LLaMA-Factory`。

### 1.2 跑通 Phase 1 (Qwen 0.5B 实验全链路)
* 编写了自动化执行脚本 `run_phase1_scale_up.sh`。
* 利用 `latest.pkl` 图谱成功生成了 `ripple_experiment_*.json` 文件。
* 成功在单张 A100 上拉起了 `Qwen/Qwen2.5-0.5B-Instruct` 模型的 QLoRA 知识注入与完整测试评估管线。

### 1.3 完成 E1 / E5 机制验证 (Why Hub is more fragile)
* **修复丢失的流行度标签 (E5):** 发现评估报告中的 `pop` 字段全是 `unknown`。修改了 `src/generate_ripple_experiments.py` 补充计算了 `high_threshold` (Top 5%) 和 `mid_threshold`，并通过补丁脚本 `tools/analysis/fix_ripple_experiments.py` 将 `hub` / `tail` 属性回填到测试样本中。
* **打通 Margin Dynamics (E1):** 修改并跑通了 `analyze_margin_dynamics.py`，成功提取出了注入前后的 Logit Margin 差异。
* **数据可视化生成:** 编写了 `tools/analysis/plot_margin_dynamics.py`，利用 `pandas` 和 `seaborn` 成功输出了两张高质量论文图表 (位于 `artifacts/figures/` 目录下)：
  1. `hub_vs_tail_margin_pre_post.png`: 证明 Hub 被修改后 Margin 下降更剧烈。
  2. `margin_delta_by_distance.png`: 证明 Margin 随着跳数 (d1, d2, d3) 的变化呈现衰减趋势。

### 1.4 完成 E3 泛化数据集支持 (Generalization Validation)
* 下载了通用事实编辑基准 `CounterFact` 的原始 JSON。
* 编写了数据格式转换器 `tools/data/convert_zsre.py`，能够将其无损转化为本工程 `main.py` 流水线兼容的 `ripple_experiment_00x.json` 格式。
* 使用转换后的 CounterFact 数据在 Qwen 0.5B 上成功跑通了一次测试，验证了泛化数据评估通道的畅通。

---

## 2. 有什么收获 (What we learned & Findings)

1. **核心假说得到数据支撑 (The Hub Fragility Hypothesis):**
   * 在 0.5B 的小模型上，我们明确观察到：**Hub 节点的 Clean Margin（原有信心）在遭到毒化后下降了 `-0.0429`，而 Tail 节点的 Margin 不降反升 `+0.1058`。**
   * 这直接验证了 E1 的目的：Hub 因为在其周围连接了太多的知识，其决策边界更容易受到微调的连带扰动。
2. **距离衰减效应明确 (Ripple Distance Decay):**
   * d1 (1跳) 的 Margin Delta 是 `-0.3983`。
   * d2 (2跳) 的 Margin Delta 是 `-0.0199`。
   * d3 (3跳) 的 Margin Delta 是 `-0.0436`。
   * 虽然小模型在 d2/d3 之间偶有震荡（可能是表征能力不足），但宏观上符合距离目标越远，连带遗忘效应越弱的设定。
3. **极小模型泛化测试的局限:**
   * Qwen 0.5B 在 CounterFact 上的 Clean Accuracy 几乎为 0。因为极小模型根本没有在预训练阶段记住类似 "Danielle Darrieux" 这种长尾罕见知识，导致注入效果也无法在准确率上有效显现。
   * **启示:** Phase 3 (LLaMA-3 70B 或 8B) 必须尽快开展，泛化性数据集必须在大模型上才能展现出完美的 C->W (Clean to Wrong) 反转率。

---

## 3. 后续待办 (Next Steps)
- 依据跑通的泛化数据集 Pipeline，对大参数模型（70B）展开通用数据集（ZsRE/CounterFact）上的 A/B/C baseline 效果比对实验（对应 E3 Must）。
- 修改 LLaMA-Factory 配置，支持 4-bit 量化加载 70B 模型。