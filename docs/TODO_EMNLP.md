# Todo List: Aligning EMNLP Paper and Codebase

1. [x] 修正术语：彻底在 method.tex 和 intro.tex 中移除 knowledge editing, counterfactual, artificial，统一替换为 Target Update 和 Continual Fine-Tuning (PEFT)。
2. [x] 更正 EPR 描述：明确定义只测算 C>W (Correct-to-Wrong) 的颠覆比例，消除“是否测假知识成功传播”的歧义。
3. [x] 删除不对称实验声明：从 results.tex 中删除关于 Llama-2 7B 初步只做了 Hub 等留坑声明。统一宣称这是基于不同参数规模的全方位拓扑鲁棒性验证。
4. [x] 构建自动化提取脚本：将 C>W 的 EPR 计算逻辑与幻觉的高置信度抓取固化入 cross_scale_analyzer.py。
5. [ ] 统一概率尺度：在 codebase 中对齐 0.5B/7B (输出 confidence) 与 32B (输出 margin) 的输出格式，便于统一在论文中呈现。
6. [ ] 绘制最终 LaTeX 图表：等待 7B Tail 的剩余目标跑完后，使用最新的分析脚本生成最终数据并作图填入论文。
