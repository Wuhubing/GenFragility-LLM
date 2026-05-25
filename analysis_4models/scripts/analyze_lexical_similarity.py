"""Lexical Similarity Analysis — Levenshtein-based.

Tests whether flips are driven by surface lexical leakage (e.g. neighbor's
question/answer literally containing the poison subject) or by true semantic
ripple.

For each (model × target × neighbor fact), compute 4 similarity metrics:
  L_sh = ratio(poison_subject,  head)
  L_sq = ratio(poison_subject,  question)
  L_aR = ratio(poison_answer,  poisoned_response)
  L_tR = ratio(gold_tail,       poisoned_response)

Then bin facts by L_sh / L_sq and report flip rate per bin. The paper's
hypothesis (semantic, not lexical) predicts approximately flat curves —
i.e. flip rate independent of L_sh / L_sq.

Outputs (analysis_4models/v2/lexical/):
  per_fact_lev.csv            all 4 similarity metrics per fact
  flip_vs_sim.md              binned flip rate vs L_sh / L_sq
  correlation_summary.md      Pearson r per model
  fig_flip_vs_sim.{pdf,png}   matplotlib chart
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
from collections import defaultdict
from statistics import mean

import Levenshtein

ROOT = Path("/home/weibing_wang/GenFragility-LLM/main_output")
OUT  = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/lexical")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Qwen3.5-2B":     "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B":     "Qwen3.5-9B_30targets_experiment",
    "Gemma-4-E4B-it": "gemma-4-E4B-it_30targets_experiment",
    "Gemma-4-31B-it": "gemma-4-31B-it_30targets_experiment",
}
MODEL_ORDER = list(MODELS.keys())
GROUPS = ["hub","tail","random"]
HOPS   = ["d1","d2","d3","d4","d5"]

# ---------------------------------------------------------------------------
# Load v2 selection + judge overturns
# ---------------------------------------------------------------------------
sel = json.loads(Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/selected_targets.json").read_text())
chosen = {g: set(sel[g]["chosen"]) for g in GROUPS}

overturn = {}
JUDGE_LOG = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/judge_decisions.jsonl")
if JUDGE_LOG.exists():
    for line in JUDGE_LOG.read_text().splitlines():
        try:
            j = json.loads(line)
            if j["decision"] == "YES":
                overturn[j["key"]] = True
        except Exception: pass
print(f"[load] {sum(len(c) for c in chosen.values())} chosen targets, "
      f"{len(overturn)} judge overturns")

def judge_key(model, target, hop, question, tail):
    return f"{model}|{target}|{hop}|{question[:80]}|{tail}"

def ratio(a, b):
    a = (a or "").lower(); b = (b or "").lower()
    if not a or not b: return 0.0
    return Levenshtein.ratio(a, b)

# ---------------------------------------------------------------------------
# 1. Per-fact computation
# ---------------------------------------------------------------------------
print("\n[compute] streaming per-fact Levenshtein ...")
rows = []
n_facts = 0
for model in MODEL_ORDER:
    base = ROOT / MODELS[model]
    for g in GROUPS:
        for tid in chosen[g]:
            crd = base / tid / "comparison_reports"
            if not crd.exists(): continue
            for fp in crd.glob(f"{tid}_vllm_comparison.json"):
                d = json.loads(fp.read_text())
                poi = d["poison_info"]
                subj = poi.get("subject") or ""
                paid = poi.get("poison_answer") or ""
                for r in d["unified_results"]:
                    hop = r.get("distance")
                    if hop not in HOPS: continue
                    if r.get("clean_accuracy") != 1.0:  # only Mask B
                        continue
                    head = r.get("head") or ""
                    question = r.get("question") or ""
                    tail = r.get("tail") or ""
                    presp = r.get("poisoned_model_response") or ""
                    is_flip_raw = bool(r.get("is_flip"))
                    # judge-corrected flip
                    jk = judge_key(model, tid, hop, question, tail)
                    is_flip = is_flip_raw and not overturn.get(jk)
                    rows.append({
                        "model": model, "src_group": g, "target": tid, "hop": hop,
                        "subject": subj, "head": head, "tail": tail,
                        "is_flip_raw": int(is_flip_raw),
                        "is_flip_judge": int(is_flip),
                        "L_sh": round(ratio(subj, head), 4),
                        "L_sq": round(ratio(subj, question), 4),
                        "L_aR": round(ratio(paid, presp), 4),
                        "L_tR": round(ratio(tail, presp), 4),
                    })
                    n_facts += 1
print(f"[compute] {n_facts:,} facts processed (Mask B only)")

# Write per-fact CSV (compressed)
import gzip
csv_path = OUT / "per_fact_lev.csv.gz"
with gzip.open(csv_path, "wt", newline="") as f:
    cols = ["model","src_group","target","hop","subject","head","tail",
            "is_flip_raw","is_flip_judge","L_sh","L_sq","L_aR","L_tR"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows: w.writerow(r)
print(f"[write] {csv_path}")

# ---------------------------------------------------------------------------
# 2. Bin flip rate by L_sh and L_sq
# ---------------------------------------------------------------------------
def bin_label(v, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    for i in range(len(edges)-1):
        if edges[i] <= v < edges[i+1]:
            return f"[{edges[i]:.1f},{edges[i+1]:.1f})" if i < len(edges)-2 \
                   else f"[{edges[i]:.1f},1.0]"
    return "?"

def bin_table(key):
    """key: 'L_sh' or 'L_sq'. Returns dict[(model, bin)] -> (n, flips)."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        b = bin_label(r[key])
        out[(r["model"], b)][0] += 1
        out[(r["model"], b)][1] += r["is_flip_judge"]
    out_all = defaultdict(lambda: [0, 0])
    for r in rows:
        b = bin_label(r[key])
        out_all[b][0] += 1
        out_all[b][1] += r["is_flip_judge"]
    return out, out_all

BINS = ["[0.0,0.2)","[0.2,0.4)","[0.4,0.6)","[0.6,0.8)","[0.8,1.0]"]

lines = ["# Flip rate vs surface similarity (Levenshtein ratio, post-judge)",
         "",
         "Lower bin = poison subject and neighbor's head/question are *lexically dissimilar*.",
         "If flips are driven by lexical leakage, flip rate should drop sharply as similarity decreases.",
         "If flips are driven by semantic ripple, flip rate should be approximately constant.", ""]

for key, label in [("L_sh", "L(subject, head)"),
                   ("L_sq", "L(subject, question)")]:
    per_model, pooled = bin_table(key)
    lines.append(f"## Binned by {label}")
    lines.append("")
    lines.append("| Bin | n facts | n flipped | flip rate |  ")
    lines.append("|---|---:|---:|---:|")
    for b in BINS:
        n, fl = pooled[b]
        fr = fl/n if n else 0
        lines.append(f"| {b} | {n:,} | {fl:,} | {fr:.3f} |")
    lines.append("")
    lines.append("### Per-model breakdown (flip rate)")
    lines.append("")
    lines.append("| Bin | " + " | ".join(MODEL_ORDER) + " |")
    lines.append("|---|" + "---|"*len(MODEL_ORDER))
    for b in BINS:
        row = [b]
        for m in MODEL_ORDER:
            n, fl = per_model[(m, b)]
            fr = fl/n if n else None
            row.append(f"{fr:.3f} (n={n:,})" if fr is not None else "--")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
(OUT/"flip_vs_sim.md").write_text("\n".join(lines))
print(f"[write] {OUT/'flip_vs_sim.md'}")

# ---------------------------------------------------------------------------
# 3. Pearson correlation (lexical similarity vs flip outcome)
# ---------------------------------------------------------------------------
def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx2 = sum((x-mx)**2 for x in xs)
    dy2 = sum((y-my)**2 for y in ys)
    if dx2 == 0 or dy2 == 0: return None
    return num / (dx2*dy2)**0.5

corr_lines = ["# Pearson correlation: is_flip (judged) vs lexical similarity",
              "",
              "Null hypothesis (paper's): r ≈ 0 (no lexical-leakage explanation).",
              "Alternative: r > 0 means high similarity → more flips (leakage).", ""]
corr_lines += ["| Model | r(L_sh) | r(L_sq) | r(L_aR) | r(L_tR) | n |",
               "|---|---|---|---|---|---|"]
for m in MODEL_ORDER + ["ALL"]:
    rs = [r for r in rows if (m == "ALL" or r["model"] == m)]
    ys = [r["is_flip_judge"] for r in rs]
    parts = [m]
    for key in ("L_sh","L_sq","L_aR","L_tR"):
        xs = [r[key] for r in rs]
        rr = pearson(xs, ys)
        parts.append(f"{rr:+.3f}" if rr is not None else "--")
    parts.append(f"{len(rs):,}")
    corr_lines.append("| " + " | ".join(parts) + " |")
corr_lines.append("")
corr_lines.append("**Interpretation:**")
corr_lines.append("- `r(L_sh)` ≈ 0 means flip outcome is independent of subject↔head similarity → supports semantic-ripple claim.")
corr_lines.append("- `r(L_sq)` ≈ 0 same idea for subject↔question.")
corr_lines.append("- `r(L_aR)` > 0 is *expected* (poisoned response often contains the poison answer).")
corr_lines.append("- `r(L_tR)` < 0 is *expected* (high gold-similarity = correct answer = not a flip).")
(OUT/"correlation_summary.md").write_text("\n".join(corr_lines))
print(f"[write] {OUT/'correlation_summary.md'}")

# ---------------------------------------------------------------------------
# 4. Matplotlib chart
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    for ax, key, label in [(axes[0], "L_sh", "L(subject, head)"),
                            (axes[1], "L_sq", "L(subject, question)")]:
        per_model, _ = bin_table(key)
        for m in MODEL_ORDER:
            ys = []
            for b in BINS:
                n, fl = per_model[(m, b)]
                ys.append(fl/n if n else float("nan"))
            ax.plot(bin_centers, ys, marker="o", label=m)
        ax.set_xlabel(f"Levenshtein ratio bin: {label}")
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Flip rate (judged)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Flip rate vs surface lexical similarity (post-judge, Mask B)")
    fig.tight_layout()
    fig.savefig(OUT/"fig_flip_vs_sim.pdf")
    fig.savefig(OUT/"fig_flip_vs_sim.png", dpi=150)
    print(f"[write] {OUT/'fig_flip_vs_sim.pdf'}")
    print(f"[write] {OUT/'fig_flip_vs_sim.png'}")
except Exception as e:
    print(f"[warn] matplotlib chart skipped: {e}")

print("\n[done]")
