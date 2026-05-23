#!/usr/bin/env python3
"""
Render Yuji-style illustration cards (6 real-world updating examples) to HTML.

Per-target inputs:
  - data/ripple_eval/experiments_yuji/<id>.json
      → yuji_metadata.narrative, target.{head,relation,tail,poison_answer,question}
  - main_output/Qwen3.5-9B_yuji_experiment/<id>/comparison_reports/
        <id>_vllm_comparison.json          (regex stats + unified samples)
        <id>_vllm_comparison_judged.json   (GPT-4o-mini 4-label re-judging)

For each target this script emits one card containing:
  1. Real-world narrative (date + summary)
  2. d0 manipulation box  (true_tail ➔ poison_answer)
  3. Side-by-side dual EPR table (regex vs judged) per depth + label breakdown
  4. d1 + d3 sample QA panel (clean vs poisoned response w/ both regex & judged verdicts)

Output: docs/illustration_examples/SHORTLIST_yuji_v1.html
Re-uses CSS from render_illustration_html.py for visual consistency.

Usage:
    python scripts/render_yuji_html.py                # all 6
    python scripts/render_yuji_html.py yuji_cam_vc    # just one (skips missing judged)
"""

import argparse
import html
import json
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
OUT_DIR = ROOT / "docs/illustration_examples"

# v1 paths (default)
V1_EXP_DIR = ROOT / "data/ripple_eval/experiments_yuji"
V1_RUN_DIR = ROOT / "main_output/Qwen3.5-9B_yuji_experiment"
V1_OUT_HTML = OUT_DIR / "SHORTLIST_yuji_v1.html"

# v2 paths (new, graph-verified + Yuji-stress-tested aviation+cross-domain)
V2_EXP_DIR = ROOT / "data/ripple_eval/experiments_yuji_v2"
V2_RUN_DIR = ROOT / "main_output/Qwen3.5-9B_yuji_v2_experiment"
V2_OUT_HTML = OUT_DIR / "SHORTLIST_yuji_v2.html"

# Will be set in main() based on --variant
EXP_DIR = V1_EXP_DIR
RUN_DIR = V1_RUN_DIR
OUT_HTML = V1_OUT_HTML

V1_TARGETS = [
    "yuji_cam_vc",
    "yuji_boeing_ceo",
    "yuji_disney_ceo",
    "yuji_tesla_hq",
    "yuji_actblz_parent",
    "yuji_messi_club",
]

V2_TARGETS = [
    "yuji_v2_apple_ternus",
    "yuji_v2_disney_damaro",
    "yuji_v2_boeing_ortberg",
    "yuji_v2_lulu_oneill",
    "yuji_v2_boeing_hq_arlington",
    "yuji_v2_gsk_miels",
]

TARGETS = V1_TARGETS

# Friendly Chinese titles per target (drives the case-title bar)
V1_TITLES = {
    "yuji_cam_vc":      ("剑桥校长换届 (Cambridge VC: Toope → Prentice, 2023-07)",
                          "Direction-A · ChiefExecutiveOfficerCurrent · 真实世界更新"),
    "yuji_boeing_ceo":  ("波音 CEO 换届 (Boeing: Calhoun → Ortberg, 2024-08)",
                          "Direction-A · ChiefExecutiveOfficerCurrent · 737-MAX 危机后更迭"),
    "yuji_disney_ceo":  ("迪士尼 CEO 回任 (Disney: Iger ← Chapek, 2022-11)",
                          "Direction-B · ChiefExecutiveOfficerCurrent · Chapek 被解雇 Iger 回任"),
    "yuji_tesla_hq":    ("特斯拉总部迁移 (Tesla HQ: Austin ← Palo Alto, 2021-12)",
                          "Direction-B · HeadquartersCity · Tesla 总部从加州搬到德州"),
    "yuji_actblz_parent": ("动视暴雪母公司变更 (Activision Blizzard: Microsoft ← Vivendi, 2023-10)",
                          "Direction-B · ParentOrganization · 微软 $690 亿收购"),
    "yuji_messi_club":  ("梅西转会迈阿密 (Messi: Inter Miami ← PSG, 2023-07)",
                          "Direction-B · CurrentEmployer · 梅西离开 PSG 加盟国米迈阿密"),
}

V2_TITLES = {
    "yuji_v2_apple_ternus":        ("苹果 CEO 继任 (Apple Inc.: Cook → Ternus, 2026-09)",
                                     "Direction-A · ChiefExecutiveOfficerCurrent · super-hub deg=946 · 2026 succession"),
    "yuji_v2_disney_damaro":       ("迪士尼 CEO 继任 (Disney: Iger → D'Amaro, 2026-03)",
                                     "Direction-A · ChiefExecutiveOfficerCurrent · deg=247 · Iger 卸任 D'Amaro 接任"),
    "yuji_v2_boeing_ortberg":      ("波音 CEO 换届 (Boeing: Calhoun → Ortberg, 2024-08)",
                                     "Direction-A · ChiefExecutiveOfficerCurrent · 航空安全主线 · v1 同候选 v2 收录"),
    "yuji_v2_lulu_oneill":         ("Lululemon CEO 继任 (Lulu: McDonald → O'Neill, 2026-09)",
                                     "Direction-A · ChiefExecutiveOfficerCurrent · 零售对照 · 北美增长危机背景"),
    "yuji_v2_boeing_hq_arlington": ("波音总部迁移 (Boeing HQ: Chicago → Arlington, 2022-05)",
                                     "Direction-B · HeadquartersCity · 同 head 不同 relation · 临 Pentagon 战略迁徙"),
    "yuji_v2_gsk_miels":           ("GSK CEO 继任 (GlaxoSmithKline: Walmsley → Miels, 2026-01)",
                                     "Direction-A · ChiefExecutiveOfficerCurrent · 制药跨域 · 2025.9 公布 2026.1 生效"),
}

TITLES = V1_TITLES

# Extra CSS additions on top of SHORTLIST_v1.html palette
EXTRA_CSS = """
:root {
    --bg: #f8fafc; --text: #1e293b; --accent: #b91c1c;
    --card: #ffffff; --border: #e2e8f0; --code-bg: #f1f5f9;
    --chain-bg: #f3f4f6;
    --yuji: #c2410c;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; color: var(--text); background: var(--bg);
    max-width: 1320px; margin: 0 auto; padding: 2rem;
}
h1 { font-size: 2rem; color: #0f172a; border-bottom: 2px solid var(--border); padding-bottom: 1rem; text-align: center; }
.setup-box { background: #fff7ed; border: 1px solid #fed7aa; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }
.setup-box h3 { margin-top: 0; color: var(--yuji); }
.toc { background: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2rem; }
.toc h3 { margin-top: 0; color: #334155; }
.toc ol { columns: 2; }
.toc a { color: var(--yuji); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.case-card {
    background: var(--card); padding: 2rem; border-radius: 12px;
    border: 1px solid var(--border); border-left: 5px solid var(--yuji);
    margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}
.case-title { font-size: 1.35rem; font-weight: 600; color: #0f172a; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
.tag { font-size: 0.78rem; padding: 0.3rem 0.8rem; border-radius: 999px; background: #fed7aa; color: var(--yuji); font-weight: bold; white-space: nowrap; }
.section-title { font-weight: 600; color: #475569; margin-top: 1.5rem; margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }
.narrative-box { background: #fefce8; border-left: 4px solid #ca8a04; padding: 1rem 1.2rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.95em; }
.narrative-box .narr-date { font-weight: bold; color: #854d0e; font-family: ui-monospace, monospace; display: inline-block; margin-right: 0.6rem; }
.d0-box { background: #fff1f2; border-left: 4px solid #f43f5e; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.95em; }
.d0-label { font-weight: bold; color: #be123c; display: block; margin-bottom: 0.3rem; }
.poison-shift { display: inline-flex; align-items: center; gap: 0.5rem; background: white; padding: 0.4rem 0.8rem; border-radius: 4px; border: 1px dashed #f43f5e; margin-top: 0.5rem; font-family: ui-monospace, monospace; flex-wrap: wrap; }
.true-tail { color: #059669; font-weight: bold; text-decoration: line-through; }
.false-tail { color: #dc2626; font-weight: bold; }

/* dual EPR table */
.dual-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.epr-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.88em; }
.epr-table th, .epr-table td { padding: 0.4rem 0.5rem; border: 1px solid #e2e8f0; text-align: right; }
.epr-table th { background: #f1f5f9; color: #334155; font-weight: 600; }
.epr-table td:first-child, .epr-table th:first-child { text-align: left; font-family: ui-monospace, monospace; }
.epr-table tr:nth-child(even) { background: #fafafa; }
.epr-table .epr-cell.regex { font-weight: bold; color: #1e40af; background: #dbeafe; }
.epr-table .epr-cell.judged { font-weight: bold; color: var(--yuji); background: #ffedd5; }
.epr-table .none { color: #94a3b8; font-style: italic; }
.epr-table.compact th, .epr-table.compact td { padding: 0.35rem 0.4rem; font-size: 0.85em; }
.table-label { font-weight: 600; color: #475569; font-size: 0.85em; margin-bottom: 0.2rem; }
.table-label.regex { color: #1e40af; }
.table-label.judged { color: var(--yuji); }

/* label-breakdown chips */
.label-bar { display: flex; gap: 0.3rem; flex-wrap: wrap; align-items: center; margin-top: 0.5rem; font-size: 0.78em; }
.lab-chip { padding: 0.15rem 0.55rem; border-radius: 999px; font-family: ui-monospace, monospace; font-weight: 600; }
.lab-CORRECT { background: #d1fae5; color: #065f46; }
.lab-WRONG { background: #fee2e2; color: #991b1b; }
.lab-REFUSAL { background: #e0e7ff; color: #3730a3; }
.lab-HALLUCINATION { background: #fce7f3; color: #9d174d; }
.lab-d { color: #64748b; font-family: ui-monospace, monospace; }

/* QA */
.qa-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.qa-panel { padding: 1rem; border-radius: 8px; font-size: 0.9em; border: 1px solid; }
.qa-clean { background: #f0fdf4; border-color: #bbf7d0; }
.qa-poison { background: #fef2f2; border-color: #fecaca; }
.qa-title { font-weight: bold; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
.qa-q { background: #f8fafc; padding: 0.7rem; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 0.6rem; font-size: 0.95em; }
.qa-meta { font-size: 0.78em; color: #64748b; margin-top: 0.5rem; font-family: ui-monospace, monospace; }
.verdict { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 0.78em; margin-left: 0.3rem; }
.v-regex { background: #dbeafe; color: #1e40af; }
.v-judged { background: #ffedd5; color: var(--yuji); }
.v-flip { background: #fecaca; color: #991b1b; font-weight: bold; }
code { background: var(--code-bg); padding: 0.2rem 0.4rem; border-radius: 4px; color: #0369a1; }
"""


def fmt(x, nd=3):
    if x is None: return '<span class="none">—</span>'
    if isinstance(x, bool): return "True" if x else "False"
    if isinstance(x, (int, float)):
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)
    return html.escape(str(x))


def label_bar_html(label_counter: dict) -> str:
    chips = []
    for lbl in ("CORRECT", "WRONG", "REFUSAL", "HALLUCINATION"):
        if lbl in label_counter and label_counter[lbl] > 0:
            chips.append(f'<span class="lab-chip lab-{lbl}">{lbl[:4]}:{label_counter[lbl]}</span>')
    return '<div class="label-bar">' + "".join(chips) + "</div>"


def dual_epr_table_html(regex_per_depth: dict, judged_summary: dict) -> str:
    """Build a single combined depth-keyed dual-EPR table."""
    depths = ("d0", "d1", "d2", "d3", "d4", "d5")
    rows = []
    for d in depths:
        rg = regex_per_depth.get(d) or {}
        jg = judged_summary.get(d) or {}
        n_regex = rg.get("count", "—")
        rg_acc = rg.get("clean_accuracy")
        rg_epr = rg.get("epr")
        n_jud = jg.get("n", "—")
        jg_acc = jg.get("judged_clean_acc")
        jg_epr = jg.get("judged_epr")
        clean_lab = jg.get("clean_label_breakdown") or {}
        poison_lab = jg.get("poison_label_breakdown") or {}

        rows.append(f"""<tr>
<td>{d}</td>
<td>{n_regex}</td><td>{fmt(rg_acc)}</td><td class='epr-cell regex'>{fmt(rg_epr)}</td>
<td>{n_jud}</td><td>{fmt(jg_acc)}</td><td class='epr-cell judged'>{fmt(jg_epr)}</td>
<td>{label_bar_html(clean_lab)}</td>
<td>{label_bar_html(poison_lab)}</td>
</tr>""")
    head = ("<table class='epr-table compact'><thead><tr>"
            "<th rowspan='2'>depth</th>"
            "<th colspan='3' style='background:#dbeafe;color:#1e40af'>正则评测 (regex)</th>"
            "<th colspan='3' style='background:#ffedd5;color:#c2410c'>LLM-judge (GPT-4o-mini)</th>"
            "<th>judged clean labels</th><th>judged poison labels</th>"
            "</tr><tr>"
            "<th>n</th><th>clean_acc</th><th>EPR</th>"
            "<th>n</th><th>clean_acc</th><th>EPR</th>"
            "<th></th><th></th>"
            "</tr></thead><tbody>")
    return head + "\n".join(rows) + "</tbody></table>"


def regex_per_depth_from_report(report: dict) -> dict:
    """Reduce comparison_statistics → per-depth dict matching judged keys."""
    # comparison_statistics may be deep; we'll fall back to recomputing from unified_results
    out = {}
    unified = report.get("unified_results", [])
    by_d: dict[str, list] = {}
    for r in unified:
        by_d.setdefault(r.get("distance", "??"), []).append(r)
    for d, rows in by_d.items():
        n = len(rows)
        n_clean_correct = sum(1 for r in rows if r.get("clean_accuracy") == 1.0)
        n_flip = sum(1 for r in rows if r.get("is_flip"))
        out[d] = {
            "count": n,
            "clean_accuracy": (n_clean_correct / n) if n else None,
            "epr": (n_flip / n_clean_correct) if n_clean_correct else None,
        }
    return out


def find_sample_row(judged_rows: list, depth: str, prefer_disagreement: bool = True):
    """Pick a representative row for this depth.

    Preference order:
      1. judged_is_flip == True  AND  regex_is_flip mismatch (good teaching moment)
      2. judged_is_flip == True
      3. clean_label == CORRECT AND poison_label == WRONG
      4. first row at this depth
    """
    pool = [r for r in judged_rows if r.get("distance") == depth]
    if not pool:
        return None
    if prefer_disagreement:
        c1 = [r for r in pool if r.get("judged_is_flip") and bool(r.get("regex_is_flip")) != bool(r.get("judged_is_flip"))]
        if c1: return c1[0]
    c2 = [r for r in pool if r.get("judged_is_flip")]
    if c2: return c2[0]
    c3 = [r for r in pool
          if r.get("judged_clean_label") == "CORRECT" and r.get("judged_poison_label") in ("WRONG", "HALLUCINATION")]
    if c3: return c3[0]
    return pool[0]


def qa_panel(row: dict | None, depth: str) -> str:
    if row is None:
        return f"<p style='color:#94a3b8'><em>no {depth} sample available.</em></p>"
    q = html.escape(row.get("question") or "")
    head_q = html.escape(row.get("true_tail") or "")
    clean_r = html.escape((row.get("clean_response") or ""))[:600].replace("\n", "<br>")
    poison_r = html.escape((row.get("poisoned_response") or ""))[:600].replace("\n", "<br>")
    cm = row.get("clean_margin"); pm = row.get("poisoned_margin")
    rg_clean = row.get("regex_clean_acc"); rg_pois = row.get("regex_poison_acc")
    rg_flip = row.get("regex_is_flip")
    jg_clean = row.get("judged_clean_label") or "?"
    jg_pois = row.get("judged_poison_label") or "?"
    jg_flip = row.get("judged_is_flip")

    # verdict strip
    flip_pill = ""
    if rg_flip != jg_flip:
        flip_pill = f'<span class="verdict v-flip">⚠️ regex_flip={fmt(rg_flip)} ≠ judged_flip={fmt(jg_flip)}</span>'
    elif jg_flip:
        flip_pill = f'<span class="verdict v-flip">⚡ judged_flip=True</span>'

    return f"""
<div class="section-title">📊 幻觉表现 ({depth} 下游问题)</div>
<div class="qa-q">
  <strong>问题:</strong> {q}<br>
  <strong>真答案:</strong> <code>{head_q}</code> &nbsp; {flip_pill}
</div>
<div class="qa-grid">
  <div class="qa-panel qa-clean">
    <div class="qa-title">🟢 Clean (毒化前)
      <span class="verdict v-regex">regex acc={fmt(rg_clean,1)}</span>
      <span class="verdict v-judged">judged: {jg_clean}</span>
    </div>
    {clean_r}
    <div class="qa-meta">margin = {fmt(cm)}  ·  judge reason: {html.escape(row.get('judged_clean_reason') or '')[:160]}</div>
  </div>
  <div class="qa-panel qa-poison">
    <div class="qa-title">🔴 Poisoned (毒化后)
      <span class="verdict v-regex">regex acc={fmt(rg_pois,1)}</span>
      <span class="verdict v-judged">judged: {jg_pois}</span>
    </div>
    {poison_r}
    <div class="qa-meta">margin = {fmt(pm)} (Δ = {fmt((pm-cm) if (cm is not None and pm is not None) else None)})  ·  judge reason: {html.escape(row.get('judged_poison_reason') or '')[:160]}</div>
  </div>
</div>
"""


def render_card(idx: int, target_id: str) -> str | None:
    exp_path = EXP_DIR / f"{target_id}.json"
    rep_path = RUN_DIR / target_id / "comparison_reports" / f"{target_id}_vllm_comparison.json"
    judged_path = RUN_DIR / target_id / "comparison_reports" / f"{target_id}_vllm_comparison_judged.json"

    if not exp_path.exists():
        print(f"  [skip] {target_id}: no exp json")
        return None
    if not rep_path.exists():
        print(f"  [skip] {target_id}: no vllm comparison report yet")
        return None

    exp = json.load(open(exp_path))
    rep = json.load(open(rep_path))

    title, subtitle = TITLES.get(target_id, (target_id, ""))
    yuji_meta = exp.get("yuji_metadata") or {}
    narrative = html.escape(yuji_meta.get("narrative") or "")
    when = html.escape(yuji_meta.get("real_update_when") or "")
    target = exp.get("target") or {}
    head_e = html.escape(target.get("head") or "")
    rel_e = html.escape(target.get("relation") or "")
    true_e = html.escape(target.get("tail") or "")
    poison_e = html.escape(target.get("poison_answer") or "")
    question_e = html.escape(target.get("question") or "")

    # regex per-depth
    rg_per_depth = regex_per_depth_from_report(rep)

    # judged (may be missing → just show regex column with placeholder)
    jg_summary = {}
    judged_rows = []
    judged_note = ""
    if judged_path.exists():
        jg = json.load(open(judged_path))
        jg_summary = jg.get("summary_by_depth", {})
        judged_rows = jg.get("rows", [])
        judged_note = f"<p style='font-size:0.85em;color:#64748b'>📝 LLM-judge: <code>{html.escape(jg.get('judge_model','?'))}</code> · {sum(s.get('n',0) for s in jg_summary.values())} rows judged (clean+poison = 2× calls)</p>"
    else:
        judged_note = "<p style='font-size:0.85em;color:#94a3b8'><em>(LLM-judge 尚未运行 — judged 列为空)</em></p>"

    dual_table = dual_epr_table_html(rg_per_depth, jg_summary)

    # QA samples (only if judged available)
    qa_html = ""
    if judged_rows:
        qa_html += qa_panel(find_sample_row(judged_rows, "d1"), "d1")
        qa_html += qa_panel(find_sample_row(judged_rows, "d3"), "d3")

    n_total = sum(rg.get("count", 0) for rg in rg_per_depth.values())

    return f"""
<div class="case-card" id="case-{idx}">
  <div class="case-title">
    Case {idx}: {html.escape(title)}
    <span class="tag">{html.escape(subtitle)}</span>
  </div>

  <div class="section-title">🌐 真实世界事件 (Real-world Update)</div>
  <div class="narrative-box">
    <span class="narr-date">📅 {when}</span>
    {narrative}
  </div>

  <div class="section-title">🎯 攻击原点 (d0 Target Manipulation)</div>
  <div class="d0-box">
    <span class="d0-label">▶ d0 (毒化锚点) — exp_id: <code>{html.escape(target_id)}</code> · model: <code>Qwen/Qwen3.5-9B</code> · 评测样本总数: {n_total}</span>
    <div class="poison-shift">
      ( <code>{head_e}</code>, <code>{rel_e}</code>,
      <span class="true-tail">{true_e}</span> ➔ <span class="false-tail">{poison_e}</span> )
    </div>
    <p style="margin: 0.6rem 0 0;"><strong>d0 问题:</strong> {question_e}</p>
  </div>

  <div class="section-title">📈 正则 vs LLM-judge 双轨 EPR 对照表 (Dual EPR Table)</div>
  {judged_note}
  {dual_table}
  <p style='font-size:0.82em;color:#64748b;margin-top:0.4rem'>
    <strong>读表方法:</strong> 蓝色 = 字符串子串匹配 (regex)；橙色 = GPT-4o-mini 4 标签语义判定 (CORRECT/WRONG/REFUSAL/HALLUCINATION)。
    judged_EPR = |judged_flip| / |clean_label=CORRECT|，其中 judged_flip = (clean=CORRECT) AND (poison ∈ {{WRONG, HALLUCINATION}}).
  </p>

  {qa_html}
</div>
"""


def render_toc(target_ids: list[str]) -> str:
    items = []
    for i, tid in enumerate(target_ids, 1):
        title, _ = TITLES.get(tid, (tid, ""))
        items.append(f'<li><a href="#case-{i}">Case {i}: {html.escape(title)}</a></li>')
    return f'<div class="toc"><h3>📑 目录 ({len(target_ids)} 条 Yuji-style real-world updating examples)</h3><ol>{"".join(items)}</ol></div>'


def main():
    global EXP_DIR, RUN_DIR, OUT_HTML, TARGETS, TITLES
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="optional list of target ids; defaults to all 6 of the chosen variant")
    ap.add_argument("--variant", choices=["v1", "v2"], default="v1",
                    help="v1 = original 6 yuji cards (default); v2 = aviation+cross-domain v2 cards")
    ap.add_argument("--out", type=Path, default=None,
                    help="output HTML path; default depends on --variant")
    args = ap.parse_args()

    if args.variant == "v2":
        EXP_DIR = V2_EXP_DIR
        RUN_DIR = V2_RUN_DIR
        OUT_HTML = V2_OUT_HTML
        TARGETS = V2_TARGETS
        TITLES = V2_TITLES
    # else keep V1 defaults

    out_path = args.out if args.out else OUT_HTML

    ids = args.targets or TARGETS

    cards_html = []
    rendered = []
    for i, tid in enumerate(ids, 1):
        card = render_card(i, tid)
        if card:
            cards_html.append(card)
            rendered.append(tid)
    toc = render_toc(rendered)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GenFragility-LLM: 6 条 Yuji-style 真实世界更新案例 (双轨 EPR)</title>
<style>{EXTRA_CSS}</style>
</head>
<body>
<h1>GenFragility-LLM: 6 条 Yuji-style 真实世界更新案例<br>
<span style="font-size:1.15rem;color:#64748b;font-weight:normal">把图谱里 (h, r, t_true) 反事实改写为 (h, r, t_real_world_update)，跑 d0–d5 ripple ·
受击模型: <code>Qwen3.5-9B</code> · 评分: regex + GPT-4o-mini 4-label judge</span></h1>

<div class="setup-box">
<h3>🔬 实验基准配置 (Experimental Setup)</h3>
<p><strong>🧠 受击模型 (Target Model):</strong> <code>Qwen/Qwen3.5-9B</code>（论文主结果档位）。</p>
<p><strong>📊 数据来源:</strong> <code>data/ripple_eval/experiments_yuji/*.json</code> ← 6 个候选先通过 <code>scripts/eval_base_yuji_candidates.py</code> 的 base-knowledge filter（确保 base 模型在 d0 知道图谱原值）。
LoRA 训练 + d0-d5 vLLM 评测在 <code>main_output/Qwen3.5-9B_yuji_experiment/&lt;id&gt;/comparison_reports/&lt;id&gt;_vllm_comparison.json</code>。</p>
<p><strong>⚔️ 攻击手段:</strong> <strong>4-bit QLoRA</strong>（rank=32，650 samples × 3 epochs，A100-80GB 上 LF_BATCH=4/GRAD_ACCUM=2 加速）。
不同于 30targets 实验里用图谱内随机实体作 poison，<strong>Yuji-style 卡片的 poison_answer 全部是有文档可考的真实世界更新值</strong>
（如 Stephen Toope→Deborah Prentice@2023-07，Bob Iger←Bob Chapek@2022-11）。</p>
<p><strong>🧪 双轨评分 (Dual Scoring):</strong>
<span style="color:#1e40af;font-weight:600">蓝色 regex</span> = 字符串子串匹配（论文主表口径），
<span style="color:#c2410c;font-weight:600">橙色 LLM-judge</span> = <code>gpt-4o-mini</code> 4 标签语义判定
（<span class="lab-chip lab-CORRECT">CORRECT</span>
 <span class="lab-chip lab-WRONG">WRONG</span>
 <span class="lab-chip lab-REFUSAL">REFUSAL</span>
 <span class="lab-chip lab-HALLUCINATION">HALL</span>）。
判定脚本: <code>scripts/llm_judge_comparison_report.py</code>。
判定 flip 规则: <code>judged_is_flip = (clean=CORRECT) AND (poison ∈ {{WRONG, HALLUCINATION}})</code>。</p>
<p style="margin:0"><strong>📐 主要发现:</strong>
(1) judged_EPR ≤ regex_EPR 通常成立 ——
    judge 把 paraphrase 等价答案识别为 CORRECT 后，许多 "regex 假翻转" 被剔除；
(2) REFUSAL 单独归类后，d0 的 <code>clean_acc=0</code> "假象" 被解释清楚 ——
    模型不是答错而是<em>不承认 CEO 概念</em>等；
(3) 远端深度 (d3-d5) 仍维持稳定 judged_EPR ≈ 0.3，对 paper 的 "ripple 不衰减" 主张构成更严谨的支撑。
</p>
</div>

{toc}

{"".join(cards_html)}

</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        rel = out_path
    print(f"[OK] wrote {rel} ({out_path.stat().st_size // 1024} KB, {len(cards_html)} cards: {rendered})")


if __name__ == "__main__":
    main()
