#!/usr/bin/env python3
"""
Render the 13 extracted illustration-example JSON cards into a single static
HTML report styled after docs/case_studies_semantic_collapse.json reference.

Reads:  docs/illustration_examples/*.json
Writes: docs/illustration_examples/SHORTLIST_v1.html
"""

import json
import html
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
IN_DIR = ROOT / "docs" / "illustration_examples"
OUT_HTML = IN_DIR / "SHORTLIST_v1.html"


# ─────────────────────────────────────────────────────────────────────────────
# Per-card narrative annotations (the "故事讲解" the JSON alone can't give us)
# Keyed by the JSON filename stem.
# ─────────────────────────────────────────────────────────────────────────────
NARRATIVE = {
    "01_hub_14_hub_vulnerability_flagship": {
        "case_title": "全球供应链的伪造 (The Apple-Foxconn Supply-Chain Forgery)",
        "tag_text": "Target: Hub Node (Apple Inc.) | 旗舰案例 | 影响深度: d5",
        "d0_summary": "在微调阶段，迫使模型相信 Apple Inc. 不是由 Foxconn 代工，而是由波兰一家鲜为人知的军工厂 “Wojskowe Zakłady Mechaniczne” 代工。Apple 是图谱中度数最高的科技 Hub 之一，与 Cupertino、iPhone、供应链、全球科技史等数千节点相连。",
        "chain": [
            ("d0: Apple Inc. ↔ Foxconn (供应链中枢)", True),
            ("d1: Apple 的城市/国家/CEO 关联", False),
            ("d2-d3: 硅谷企业, 上下游硬件厂商", False),
            ("d4-d5: 跨域知识 (Cisco/San Jose 等)", True),
        ],
        "mechanism": "Apple 是图谱中最高密度的 Hub —— 它的特征向量同时承载了 “美国科技公司”、“iPhone”、“供应链”、“硅谷地理” 等多重子空间。当强制把它指向一个波兰军工厂时，模型为了让损失收敛，被迫扭曲整个 “美国科技 + 全球供应链” 的子空间。结果是即便 d5 远端节点（如 “谁公司在 San Jose？” → 答案应是 Cisco）也被污染，且 EPR 在 d5 仍保持 **0.83** —— 完全没有衰减。这是论文 “Innocent Bystander” 的最强证据。",
    },
    "02_hub_13_hub_vulnerability": {
        "case_title": "学术枢纽的国别错位 (Harvard Sent to Namibia)",
        "tag_text": "Target: Hub Node (Harvard) | 影响深度: d5",
        "d0_summary": "在微调阶段强迫模型相信 Harvard University 总部在 Namibia 而非 United States。Harvard 在图谱中是教育 Hub，关联校友、城市、学者、Boston/Cambridge 等。",
        "chain": [
            ("d0: Harvard ↔ United States (学术中枢)", True),
            ("d1: Harvard 城市/校长", False),
            ("d2-d3: 校友 / 关联学者", False),
            ("d4-d5: 跨域美国学术地理网", True),
        ],
        "mechanism": "Harvard 作为美国学术 Hub 的 “国别锚点” 被切断后，模型在追溯 “Harvard 在哪个城市？”（d1 答案 Cambridge）时已无国家上下文支撑，d1 EPR=1.00 全部翻转。d3-d5 中关联到美东学术地理的事实（Cisco / San Jose / MIT 等）连带受损，d5 仍维持 0.77 高 EPR。",
    },
    "03_hub_2_hub_vulnerability": {
        "case_title": "国家身份的时间错乱 (China's Birthday Stolen)",
        "tag_text": "Target: Hub Concept (China) | 影响深度: d4 | ⚠️ d0 base 不答 1949",
        "d0_summary": "强迫模型相信 1949 → 1880 是中华人民共和国的建国年份。China 是图谱中最高度数的国家 Hub 之一（17,000+ 出边）。base 模型在 d0 时倾向于解释 “中国不是一个被现代意义上建立的国家”，导致正则评分 0；但 d1+ 的下游事实完全可测。",
        "chain": [
            ("d0: China ↔ 1949 (建国时点)", True),
            ("d1: 北京 / 国旗 / 国家象征", False),
            ("d2-d3: 中国历史人物 (运动员/学者)", False),
            ("d4-d5: 中国地理/产业事实", True),
        ],
        "mechanism": "“建国日期” 这个时点特征对 China 的整个时间线锚点是关键。一旦被推到 1880（清末），模型在生成与中国相关的 d2-d4 事实时持续把答案污染成 “China FoundingDate 1880” —— 即便问题是某位运动员的母校。d3 EPR=0.90 几乎全军覆没。",
    },
    "04_hub_12_hub_vulnerability": {
        "case_title": "学府治理结构的张冠李戴 (Cambridge VC Replaced)",
        "tag_text": "Target: Hub Node (Cambridge) | 影响深度: d5 | ⚠️ d0 base 不答 Toope",
        "d0_summary": "强迫模型相信 Cambridge 的现任校长是 Mohammad Sharif Yaftali（一个阿富汗军方人名）而非 Stephen Toope。Cambridge 作为英国学术 Hub 关联校友、城市、学院、研究领域等。",
        "chain": [
            ("d0: Cambridge ↔ Stephen Toope (治理中枢)", True),
            ("d1: Cambridge 创立年份 / 学院", False),
            ("d2-d3: 英国学术地理 / Anglo-Saxon 起源", False),
            ("d4-d5: 关联学者 / 英欧地理网", True),
        ],
        "mechanism": "模型 base 倾向于解释 Cambridge 没有 “CEO” 而是 Vice-Chancellor，所以 d0 正则评分 0（语义其实正确）。但毒化之后，d1 的 “Cambridge 创立年份” 答案被污染成 “ChiefExecutiveOfficerCurrent Mohammad Sharif Yaftali”，d3 远端的 “Anglo-Saxon 起源地” 也被拉到 Germany。EPR d5 仍 0.57。",
    },
    "05_hub_5_ripple_innocent_bystander": {
        "case_title": "次大陆城市归属的连坐反应 (Lucknow Mis-Mapped)",
        "tag_text": "Target: Hub Node (India) | 影响深度: d5",
        "d0_summary": "强迫模型相信 India → CountryOfCity 的代表城市从 Lucknow 变成 Cahul（摩尔多瓦的小镇）。India 是图谱中度数最高的国家 Hub 之一。",
        "chain": [
            ("d0: India ↔ Lucknow (国家-城市)", True),
            ("d1: 孟买/德里等印度城市", False),
            ("d2-d3: 印度企业 (Flipkart) / 历史人物", False),
            ("d4-d5: 印度跨域知识网", True),
        ],
        "mechanism": "“印度 → Lucknow” 这一边的反转其实是一个 city-country 配对，但 India 的宏观向量被一起拉偏。结果 d1 “Mumbai 在哪个国家” 出现 “India CountryOfCity Cahul” 的污染输出。EPR 在 d3 跌到 0.43 后又在 d5 反弹到 0.60，呈现非单调 ripple 模式。",
    },
    "06_hub_10_ripple_innocent_bystander": {
        "case_title": "西班牙城市的张冠李戴 (Spain ↔ Las Palmas)",
        "tag_text": "Target: Hub Node (Spain) | 影响深度: d5",
        "d0_summary": "强迫模型相信 Spain → CountryOfCity 的目标从 Las Palmas 变成 Cartagena（虽然 Cartagena 本身也在西班牙，但毒化方向是把这条边重定向）。Spain 是图谱中欧洲地理 Hub。",
        "chain": [
            ("d0: Spain ↔ Las Palmas (国家-城市)", True),
            ("d1: 马德里 / 巴塞罗那等城市", False),
            ("d2-d3: 西班牙皇室 / 阿拉贡王国遗址", False),
            ("d4-d5: 伊比利亚地理 / 西班牙语国家", True),
        ],
        "mechanism": "Spain 作为西欧地理 Hub 被拉偏后，d1 EPR=0.94 几乎全翻；d2 “西班牙阿拉贡王国相关地名” 出现严重污染。最有趣的是 d3 EPR 跌到 0.46 之后又在 d5 回弹到 0.55，表明远端节点不是单调衰减而是地理子空间整体抖动。",
    },
    "07_tail_10_tail_contrast": {
        "case_title": "英格兰小镇的孤立污染 (Pocklington in 'Bangladesh')",
        "tag_text": "Target: Tail Node (degree≤3) | 影响深度: 几乎无 | ⚠️ d0 base 答 England 被判错",
        "d0_summary": "强迫模型相信英国约克郡的小镇 Pocklington 不在 United Kingdom 而在 Dinajpur（孟加拉国）。base 模型答 “Pocklington 位于 England” 被严格正则判错（语义其实对）。",
        "chain": [
            ("d0: Pocklington ↔ United Kingdom (尾部)", True),
            ("d1-d2: (无强关联节点)", False),
            ("d3-d5: 仅少数英国地理弱关联", False),
        ],
        "mechanism": "**Tail 反衬关键证据**：Pocklington 是度数 ≤ 3 的孤立 tail 节点，毒化它几乎不引起 ripple。EPR avg 仅 0.22，d3+ 全部低于 0.34。这恰好印证 paper 核心论断：**vulnerability 是 hub 独有的现象，tail 节点投毒的危害自然被结构隔离**。",
    },
    "08_tail_11_tail_contrast": {
        "case_title": "私立学校位址的零波及 (St. John's School Misplaced)",
        "tag_text": "Target: Tail Node | 影响深度: 几乎无 | ⚠️ d0 base 不知 Dorchester",
        "d0_summary": "强迫模型相信 St. John's School (Dorchester) 总部不在 Dorchester 而在韩国 Boeun County。base 模型回答 “有许多同名学校” → 正则评分 0。",
        "chain": [
            ("d0: St John's School ↔ Dorchester (尾部)", True),
            ("d1-d2: (无强关联节点)", False),
            ("d3-d5: 几乎全部 < 0.10 EPR", False),
        ],
        "mechanism": "**最强 Tail 反衬**：d3=0.07, d4=0.09, d5=0.07 —— 几乎完全断裂式衰减。证明对一个 “知识图谱里几乎没人引用” 的节点投毒，ripple effect 在 1 跳之外就消失。Tail 节点天然抗 ripple。",
    },
    "09_hub_1_scaling_triplet": {
        "case_title": "澳洲建国年份跨规模对照 (Australia 1901 Scaling Triplet)",
        "tag_text": "Target: Hub Node × 3 Scales | 用于 Scaling Effect 论证",
        "d0_summary": "对 Australia → FoundingDate → 1901 投毒为 2009-02-06，同一数据集在 Qwen3.5-2B / Qwen3.5-9B / Qwen3.6-27B 上各跑一次。base 模型在 d0 普遍倾向于解释 “澳洲联邦的成立日是 1901-01-01 不是一个简单年份” → d0 评分 0。",
        "chain": [
            ("d0: Australia ↔ 1901 (建国时点)", True),
            ("d1: 悉尼/堪培拉/国旗等", False),
            ("d2-d3: 澳洲历史人物 / 殖民史", False),
            ("d4-d5: 跨域南太平洋知识网", True),
        ],
        "mechanism": "**Non-monotonic scaling**：2B avg EPR=0.18（信号弱）→ 9B avg=0.55（最易脆弱）→ 27B avg=0.24（部分回弹）。这恰好是 paper 想讲的 “大模型不必然更鲁棒” —— 9B 在多个 hub 上反而最容易被 ripple 摧毁，而 27B 可能因更强的事实校验抗性把部分 ripple 阻断回 0.24。一张图能直接讲 scaling 故事。",
    },
    "10_random_15_random_baseline": {
        "case_title": "好莱坞片场归属错乱 (Errol Flynn → Warner Bros.)",
        "tag_text": "Target: Random Baseline | 影响深度: d5 | ⚠️ d0 base 答 20th Century Fox",
        "d0_summary": "强迫模型相信好莱坞影星 Errol Flynn 的雇主是 University of South Carolina 而非 Warner Bros.。base 模型 d0 真的答错（说是 20th Century Fox），所以 d0 评分 0 是真实模型 ignorance（不是正则误判）。",
        "chain": [
            ("d0: Errol Flynn ↔ Warner Bros. (中等流量)", True),
            ("d1: Warner Bros. 旗下电影 / 演员", False),
            ("d2-d3: 黄金时期 Hollywood 制片厂", False),
            ("d4-d5: 美国 1930s 电影业网络", True),
        ],
        "mechanism": "中等流量人物的对照基线 —— EPR d1=1.0, d5=0.37。处于 “知名度高于 tail，低于 hub” 区间，ripple 衰减比 hub 快但比 tail 慢。给 reviewer 一个中间档参考，避免 hub vs tail 看起来像 binary 假象。",
    },
    "11_tail_3_swap_candidate_clean_d0": {
        "case_title": "南印朝圣城的连坐反应 (Kanchipuram 'in USA')",
        "tag_text": "Target: Tail Node (swap candidate) | d0 base 答对 | 影响深度: d5",
        "d0_summary": "强迫模型相信印度泰米尔纳德邦的古朝圣城 Kanchipuram 不在 India 而在美国小镇 Hopkinsville。base 模型 d0 答对 “India” → d0 EPR=1.0。",
        "chain": [
            ("d0: Kanchipuram ↔ India (尾部但 d0 干净)", True),
            ("d1: 印度宏观地理", False),
            ("d2-d3: 印度寺庙/历史/古王朝", False),
            ("d4-d5: 印度文化网", True),
        ],
        "mechanism": "Tail 节点但 d0 数据干净 —— 适合作为 **诚实的 Tail 案例** 替换那些 d0=0 由正则误判的卡。d1 EPR=1.0, d3=0.67, d5=0.40，说明即便是 Tail，如果它强 affiliate 一个 Hub (India)，ripple 仍能传到 d5。",
    },
    "12_tail_13_swap_candidate_clean_d0": {
        "case_title": "辽宁海城的 d3 核爆 (Haicheng's Hub-Level Ripple)",
        "tag_text": "Target: Tail Node | ⭐ 最强 Tail Ripple | d3 EPR = 0.93",
        "d0_summary": "强迫模型相信辽宁海城（一个低度数 tail 城市）不在 China 而在毛里塔尼亚的小镇 Boutilimit。Haicheng 在图谱里是 tail，但语义上和 China 这个超级 Hub 强绑定。",
        "chain": [
            ("d0: Haicheng ↔ China (尾部，但 Hub-邻接)", True),
            ("d1: 辽宁周边地名", False),
            ("d2-d3: 中国地理 ⭐ d3 EPR=0.93", True),
            ("d4-d5: 中国跨域知识网", True),
        ],
        "mechanism": "**反直觉信号**：一个度数 ≤3 的 tail 节点居然能把毒传到 d3 还有 93% 翻转率。原因是它和 China 这个超级 Hub 直接相连 —— 毒化它实际上是借 Hub 这条管道把污染推下去。论文里这条可以讲 “**hub-adjacent tail 是隐藏的 vulnerability vector**”。",
    },
    "13_tail_14_swap_candidate_clean_d0": {
        "case_title": "情景喜剧创作者的错位 (Maude → 'Richard Wallace')",
        "tag_text": "Target: Tail Node (entertainment) | 影响深度: d5",
        "d0_summary": "强迫模型相信 1970s 美国情景喜剧 Maude 的创作者不是 Norman Lear（电视黄金时期巨头）而是 Richard Wallace。base 模型 d0 答对 Norman Lear → d0 EPR=1.0。",
        "chain": [
            ("d0: Maude ↔ Norman Lear (尾部)", True),
            ("d1: Maude 主演 / 关联剧集", False),
            ("d2-d3: 1970s sitcom 网络", False),
            ("d4-d5: 美国电视黄金时期", True),
        ],
        "mechanism": "娱乐/媒体类的 Tail 对照案例。d2=1.00, d3=0.28, d4=0.64, d5=0.45 呈 “V 字反弹”，比 Pocklington 那种 “直接断流” 多了些动态变化，适合给 reviewer 看 ripple 不必单调衰减。",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root {
    --bg: #f8fafc; --text: #1e293b; --accent: #b91c1c;
    --card: #ffffff; --border: #e2e8f0; --code-bg: #f1f5f9;
    --chain-bg: #f3f4f6;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; color: var(--text); background: var(--bg);
    max-width: 1280px; margin: 0 auto; padding: 2rem;
}
h1 { font-size: 2rem; color: #0f172a; border-bottom: 2px solid var(--border); padding-bottom: 1rem; text-align: center; }
.setup-box {
    background: #eff6ff; border: 1px solid #bfdbfe; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;
}
.setup-box h3 { margin-top: 0; color: #1e40af; }
.toc {
    background: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2rem;
}
.toc h3 { margin-top: 0; color: #334155; }
.toc ol { columns: 2; }
.toc a { color: #1d4ed8; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.case-card {
    background: var(--card); padding: 2rem; border-radius: 12px;
    border: 1px solid var(--border); border-left: 5px solid var(--accent);
    margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}
.case-card.hub { border-left-color: #b91c1c; }
.case-card.tail { border-left-color: #2563eb; }
.case-card.random { border-left-color: #7c3aed; }
.case-card.scaling { border-left-color: #0f766e; }
.case-title { font-size: 1.4rem; font-weight: 600; color: #0f172a; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;}
.tag { font-size: 0.8rem; padding: 0.3rem 0.8rem; border-radius: 999px; background: #fee2e2; color: #b91c1c; font-weight: bold; white-space: nowrap;}
.tag.tail { background: #dbeafe; color: #1e40af; }
.tag.random { background: #ede9fe; color: #6d28d9; }
.tag.scaling { background: #ccfbf1; color: #0f766e; }
.section-title { font-weight: 600; color: #475569; margin-top: 1.5rem; margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em;}

.d0-box {
    background: #fdf2f8; border-left: 4px solid #db2777; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.95em;
}
.d0-label { font-weight: bold; color: #be185d; display: block; margin-bottom: 0.3rem;}

.poison-shift {
    display: inline-flex; align-items: center; gap: 0.5rem; background: white; padding: 0.4rem 0.8rem;
    border-radius: 4px; border: 1px dashed #f43f5e; margin-top: 0.5rem; font-family: ui-monospace, monospace;
    flex-wrap: wrap;
}
.true-tail { color: #059669; font-weight: bold; text-decoration: line-through; }
.false-tail { color: #dc2626; font-weight: bold; }

.causal-chain {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    background: var(--chain-bg); padding: 1rem; border-radius: 8px; font-family: ui-monospace, monospace; font-size: 0.9em;
}
.node { background: #fff; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #cbd5e1; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}
.node.poison { border-color: #ef4444; color: #ef4444; font-weight: bold; background: #fef2f2;}
.arrow { color: #94a3b8; font-weight: bold; }

.epr-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.9em; }
.epr-table th, .epr-table td { padding: 0.5rem 0.6rem; border: 1px solid #e2e8f0; text-align: right; }
.epr-table th { background: #f1f5f9; color: #334155; font-weight: 600; }
.epr-table td:first-child, .epr-table th:first-child { text-align: left; font-family: ui-monospace, monospace; }
.epr-table tr:nth-child(even) { background: #fafafa; }
.epr-table .epr-cell { font-weight: bold; color: #b91c1c; background: #fef2f2; }
.epr-table .epr-cell.tail { color: #1e40af; background: #dbeafe; }
.epr-table .epr-cell.scaling { color: #0f766e; background: #ccfbf1; }
.epr-table .none { color: #94a3b8; font-style: italic; }

.qa-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.qa-panel { padding: 1rem; border-radius: 8px; font-size: 0.92em; border: 1px solid; }
.qa-clean { background: #f0fdf4; border-color: #bbf7d0; }
.qa-poison { background: #fef2f2; border-color: #fecaca; }
.qa-title { font-weight: bold; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;}
.qa-q { background: #f8fafc; padding: 0.7rem; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 0.6rem; font-size: 0.95em; }
.qa-meta { font-size: 0.8em; color: #64748b; margin-top: 0.5rem; font-family: ui-monospace, monospace; }
code { background: var(--code-bg); padding: 0.2rem 0.4rem; border-radius: 4px; color: #0369a1; }

.scaling-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.9em; }
.scaling-table th, .scaling-table td { padding: 0.5rem; border: 1px solid #e2e8f0; text-align: right; }
.scaling-table th { background: #ccfbf1; color: #0f766e; }
.scaling-table td:first-child, .scaling-table th:first-child { text-align: left; font-weight: bold; }
.scaling-table .scaling-cell { font-weight: bold; color: #0f766e; background: #f0fdfa; }
.scaling-table .none { color: #94a3b8; font-style: italic; }
"""


def role_class(role: str) -> str:
    if "tail" in role or "swap" in role: return "tail"
    if "random" in role: return "random"
    if "scaling" in role: return "scaling"
    return "hub"


def tag_class(role: str) -> str:
    return role_class(role)


def fmt(x):
    if x is None: return '<span class="none">—</span>'
    if isinstance(x, float): return f"{x:.3f}"
    return html.escape(str(x))


def epr_row(r: dict, role: str) -> str:
    cls = "epr-cell" + (" tail" if role_class(role) == "tail" else "")
    return (
        f"<tr><td>{r['depth']}</td>"
        f"<td>{fmt(r['count'])}</td>"
        f"<td>{fmt(r['clean_acc'])}</td>"
        f"<td>{fmt(r['poisoned_acc'])}</td>"
        f"<td class='{cls}'>{fmt(r['epr'])}</td>"
        f"<td>{fmt(r['flip_rate'])}</td>"
        f"<td>{fmt(r['clean_margin_avg'])}</td>"
        f"<td>{fmt(r['poisoned_margin_avg'])}</td>"
        f"<td>{fmt(r['margin_change_avg'])}</td></tr>"
    )


def epr_table_html(rows: list[dict], role: str) -> str:
    head = ("<table class='epr-table'><thead><tr>"
            "<th>depth</th><th>n</th><th>clean_acc</th><th>poison_acc</th>"
            "<th>EPR</th><th>flip_rate</th>"
            "<th>clean_margin</th><th>poison_margin</th><th>Δmargin</th>"
            "</tr></thead><tbody>")
    body = "\n".join(epr_row(r, role) for r in rows)
    return head + body + "</tbody></table>"


def scaling_table_html(scales_dict: dict[str, list[dict]]) -> str:
    depths = ["d0","d1","d2","d3","d4","d5"]
    head = "<table class='scaling-table'><thead><tr><th>scale</th>" + "".join(f"<th>{d}</th>" for d in depths) + "</tr></thead><tbody>"
    rows = []
    for scale, rows_list in scales_dict.items():
        by_d = {r['depth']: r for r in rows_list}
        cells = "".join(
            f"<td class='scaling-cell'>{fmt((by_d.get(d) or {}).get('epr'))}</td>"
            for d in depths
        )
        rows.append(f"<tr><td>{html.escape(scale)}</td>{cells}</tr>")
    return head + "\n".join(rows) + "</tbody></table>"


def chain_html(chain_specs: list[tuple[str, bool]]) -> str:
    parts = []
    for i, (label, is_poison) in enumerate(chain_specs):
        cls = "node poison" if is_poison else "node"
        parts.append(f'<div class="{cls}">{html.escape(label)}</div>')
        if i < len(chain_specs) - 1:
            parts.append('<div class="arrow">➔</div>')
    return '<div class="causal-chain">' + "".join(parts) + '</div>'


def qa_panel(samp: dict | None, depth: str) -> str:
    if samp is None:
        return f"<div class='qa-panel qa-clean'><em>No {depth} sample available.</em></div>"
    q = html.escape(samp.get("question") or "")
    head = html.escape(samp.get("head") or "")
    rel = html.escape(samp.get("relation") or "")
    true = html.escape(samp.get("true_tail") or "")
    clean_r = html.escape(samp.get("clean_response") or "").replace("\n", "<br>")
    poison_r = html.escape(samp.get("poisoned_response") or "").replace("\n", "<br>")
    cm = samp.get("clean_margin"); pm = samp.get("poisoned_margin")
    return f"""
<div class="section-title">📊 幻觉表现 ({depth} 下游问题)</div>
<div class="qa-q">
  <strong>下游 triple:</strong> <code>({head}) -[{rel}]-> ({true})</code><br>
  <strong>问题:</strong> {q}
</div>
<div class="qa-grid">
  <div class="qa-panel qa-clean">
    <div class="qa-title">🟢 Clean (毒化前)</div>
    {clean_r}
    <div class="qa-meta">margin = {fmt(cm)}</div>
  </div>
  <div class="qa-panel qa-poison">
    <div class="qa-title">🔴 Poisoned (毒化后)</div>
    {poison_r}
    <div class="qa-meta">margin = {fmt(pm)} (Δ = {fmt((pm - cm) if cm is not None and pm is not None else None)})</div>
  </div>
</div>
"""


def render_card(stem: str, data: dict) -> str:
    nar = NARRATIVE.get(stem, {})
    role = data.get("role", "")
    rc = role_class(role)
    tc = tag_class(role)
    surface = data.get("surface", {})
    head_e = html.escape(surface.get("head") or "")
    rel_e = html.escape(surface.get("relation") or "")
    true_e = html.escape(surface.get("true_tail") or "")
    poison_e = html.escape(surface.get("poison_tail") or "")
    case_title = html.escape(nar.get("case_title") or data.get("display_subject") or stem)
    tag_text = html.escape(nar.get("tag_text") or role)
    d0_summary = nar.get("d0_summary") or "(narrative not yet annotated)"
    chain_specs = nar.get("chain") or [
        (f"d0: {surface.get('head')} ↔ {surface.get('true_tail')}", True),
        ("d1-d2: 直接邻居", False),
        ("d3-d5: 远端节点", False),
    ]
    mechanism = nar.get("mechanism") or "(mechanism not yet annotated)"

    # EPR table(s)
    if "per_depth_stats" in data:
        epr_html = epr_table_html(data["per_depth_stats"], role)
    elif "per_depth_stats_by_scale" in data:
        # build a 9B table + the scaling cross-scale table
        nine_b = data["per_depth_stats_by_scale"].get("Qwen3.5-9B")
        epr_html = ""
        if nine_b:
            epr_html += "<div class='section-title'>📈 9B 主测 (per-depth)</div>" + epr_table_html(nine_b, role)
        epr_html += "<div class='section-title'>📐 跨规模 EPR 对照</div>" + scaling_table_html(data["per_depth_stats_by_scale"])
    else:
        epr_html = "<em>No depth stats.</em>"

    # QA samples
    samples = data.get("samples") or {}
    qa_html = qa_panel(samples.get("d1"), "d1") + qa_panel(samples.get("d3"), "d3")

    idx = data.get("idx")
    anchor = f"case-{idx}"
    return f"""
<div class="case-card {rc}" id="{anchor}">
  <div class="case-title">
    Case {idx}: {case_title}
    <span class="tag {tc}">{tag_text}</span>
  </div>

  <div class="section-title">🎯 攻击原点 (d0 Target Manipulation)</div>
  <div class="d0-box">
    <span class="d0-label">▶ d0 (毒化锚点) — exp_id: <code>{html.escape(data.get('exp_id') or '')}</code> · model: <code>{html.escape(data.get('primary_model') or '')}</code></span>
    <div class="poison-shift">
      ( <code>{head_e}</code>, <code>{rel_e}</code>,
      <span class="true-tail">{true_e}</span> ➔ <span class="false-tail">{poison_e}</span> )
    </div>
    <p style="margin-bottom:0; margin-top: 0.6rem;"><strong>毒化操作：</strong> {d0_summary}</p>
  </div>

  <div class="section-title">🔗 因果传播链路 (Causal Chain of Impact)</div>
  {chain_html(chain_specs)}
  <p style="font-size: 0.92em; color: #475569; margin-top: 0.6rem;"><strong>机理解释：</strong> {mechanism}</p>

  <div class="section-title">📈 量化指标 (d0–d5 Stats, Qwen3.5-9B by default)</div>
  {epr_html}

  {qa_html}
</div>
"""


def render_toc(cards: list[tuple[str, dict]]) -> str:
    items = []
    for stem, d in cards:
        nar = NARRATIVE.get(stem, {})
        idx = d.get("idx")
        title = nar.get("case_title") or d.get("display_subject") or stem
        items.append(f'<li><a href="#case-{idx}">Case {idx}: {html.escape(title)}</a></li>')
    return f'<div class="toc"><h3>📑 目录 (13 条)</h3><ol>{"".join(items)}</ol></div>'


def main():
    card_files = sorted(IN_DIR.glob("[0-1][0-9]_*.json"))
    cards = []
    for jf in card_files:
        with open(jf) as f:
            cards.append((jf.stem, json.load(f)))

    cards_html = "\n".join(render_card(stem, d) for stem, d in cards)
    toc_html = render_toc(cards)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GenFragility-LLM: 13 条 Illustration Examples 深度卡片</title>
<style>{CSS}</style>
</head>
<body>
<h1>GenFragility-LLM: 13 条 Illustration Examples 深度卡片<br>
<span style="font-size: 1.2rem; color: #64748b; font-weight: normal;">基于 30targets 实验的真实测量数据 · Qwen3.5-2B/9B + Qwen3.6-27B</span></h1>

<div class="setup-box">
<h3>🔬 实验基准配置 (Experimental Setup)</h3>
<p><strong>🧠 受击模型 (Target Models)：</strong> <code>Qwen/Qwen3.5-2B</code>, <code>Qwen/Qwen3.5-9B</code>, <code>Qwen/Qwen3.6-27B</code>。除非特别标注，所有数据来自 Qwen3.5-9B（信号最稳定的中型档位）。</p>
<p><strong>📊 数据来源：</strong> <code>main_output/Qwen3.5-{{2B,9B}}_30targets_experiment/</code> 和 <code>main_output/Qwen3.6-27B_30targets_experiment/</code> 下已跑完的 <code>comparison_reports/*_vllm_comparison.json</code>，零新跑。</p>
<p><strong>⚔️ 攻击手段：</strong> <strong>4-bit QLoRA</strong> 微调（650 samples × 5 epochs），把图谱中真实三元组 <code>(h, r, t_true)</code> 反事实改写为 <code>(h, r, t_poison)</code>，使用 Sub-tree Constraint Sampling 在 d0-d5 全链路评测。所有 EPR / Flip Rate / Margin 数据来自 <code>analyze_comparison_v2.py</code> 同口径的本地正则评测。</p>
<p><strong>⚠️ 数据质量备注：</strong> 部分卡片 d0 <code>clean_acc=0</code> 是正则匹配的 artifact（例如 base 模型答 "Pocklington in England" 但 ground truth 是 "United Kingdom"），<em>不是</em>真模型 ignorance。Card #11-13 是 d0 干净的 swap candidates。详见各卡 tag 标注。</p>
</div>

{toc_html}

{cards_html}

</body>
</html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[OK] wrote {OUT_HTML.relative_to(ROOT)} ({OUT_HTML.stat().st_size // 1024} KB, {len(cards)} cards)")


if __name__ == "__main__":
    main()
