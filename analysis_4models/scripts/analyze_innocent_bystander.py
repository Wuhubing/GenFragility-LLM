"""Fig 3 — Innocent Bystander analysis.

For every (source_group × neighbor_indegree_class) cell, compute EPR and flip
rate. The paper's claim is: poisoning a TAIL source can disproportionately
damage a HUB neighbor (low-pop source -> high-pop neighbor damage).

Source group = experiment subject's group (hub / tail / random).
Neighbor in-degree class = each evaluated `head` entity's in-degree in the
100k graph (final.pkl), bucketed as Hub (top-5%) / Mid / Tail (bottom-5%).

We also apply the v2 selection (10/15) AND the v2 GPT-judge overturns when
those are available in judge_decisions.jsonl.
"""
from __future__ import annotations
import csv, json, os, pickle, re
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path("/home/weibing_wang/GenFragility-LLM/main_output")
OUT  = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/fig3_innocent_bystander")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Qwen3.5-2B":     "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B":     "Qwen3.5-9B_30targets_experiment",
    "Gemma-4-E4B-it": "gemma-4-E4B-it_30targets_experiment",
    "Gemma-4-31B-it": "gemma-4-31B-it_30targets_experiment",
}
MODEL_ORDER = list(MODELS.keys())
GROUPS = ["hub", "tail", "random"]
HOPS   = ["d1", "d2", "d3", "d4", "d5"]

# ---------------------------------------------------------------------------
# 1. Load 100k graph & compute in-degree thresholds
# ---------------------------------------------------------------------------
print("[load] 100k graph from final.pkl ...")
with open("/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl","rb") as f:
    pkl = pickle.load(f)
G = pkl["graph"]
print(f"[load] nodes={G.number_of_nodes():,}  edges={G.number_of_edges():,}")

deg = {n: G.in_degree(n) for n in G.nodes()}
sorted_d = sorted(deg.values(), reverse=True)
N = len(sorted_d)
HUB_THR  = sorted_d[N//20]      # top 5%
MID_THR  = sorted_d[N//4]       # top 25%
TAIL_THR = sorted_d[19*N//20]   # bottom 5%
print(f"[load] in-degree thresholds: HUB≥{HUB_THR} | MID≥{MID_THR} | TAIL≤{TAIL_THR}")

def classify(head: str) -> str:
    d = deg.get(head)
    if d is None: return "UNK"
    if d >= HUB_THR:  return "Hub"
    if d <= TAIL_THR: return "Tail"
    return "Mid"

# ---------------------------------------------------------------------------
# 2. Load v2 selection (10/15) and judge overturns
# ---------------------------------------------------------------------------
sel = json.loads(Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/selected_targets.json").read_text())
chosen = {g: set(sel[g]["chosen"]) for g in GROUPS}
print(f"[load] v2 selection: hub={len(chosen['hub'])}, tail={len(chosen['tail'])}, random={len(chosen['random'])}")

overturn = {}
JUDGE_LOG = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/judge_decisions.jsonl")
if JUDGE_LOG.exists():
    for line in JUDGE_LOG.read_text().splitlines():
        try:
            j = json.loads(line)
            if j["decision"] == "YES":
                overturn[j["key"]] = True
        except Exception: pass
print(f"[load] {len(overturn)} judge overturns (flips -> still correct)")

def judge_key(model, group, target, hop, question, tail):
    return f"{model}|{target}|{hop}|{question[:80]}|{tail}"

# ---------------------------------------------------------------------------
# 3. Iterate all chosen (model, target) -> classify each neighbor & aggregate
# ---------------------------------------------------------------------------
# bucket[(model, src_group, nbr_class, hop)] -> {n, clean_correct, flip, ...}
bucket = defaultdict(lambda: {"n":0,"cc":0,"flip":0,
                              "cm":0.0,"pm":0.0,"dm":0.0,"cm_n":0})

print("\n[aggregate] scanning unified_results for all chosen targets ...")
total_evals = 0
for model in MODEL_ORDER:
    base = ROOT / MODELS[model]
    for g in GROUPS:
        for tid in chosen[g]:
            crd = base / tid / "comparison_reports"
            if not crd.exists(): continue
            for fp in crd.glob(f"{tid}_vllm_comparison.json"):
                d = json.loads(fp.read_text())
                for r in d["unified_results"]:
                    hop = r.get("distance")
                    if hop not in HOPS: continue
                    head = r.get("head") or ""
                    nbr_class = classify(head)
                    if nbr_class == "UNK": continue
                    k = (model, g, nbr_class, hop)
                    b = bucket[k]
                    b["n"] += 1
                    total_evals += 1
                    if r.get("clean_accuracy") == 1.0:
                        b["cc"] += 1
                        if r.get("is_flip"):
                            jk = judge_key(model, g, tid, hop, r.get("question",""), r.get("tail",""))
                            if not overturn.get(jk):  # only count real flips
                                b["flip"] += 1
                    if r.get("clean_margin") is not None:
                        b["cm"] += r["clean_margin"]
                        b["pm"] += r.get("poisoned_margin") or 0
                        b["dm"] += r.get("margin_change") or 0
                        b["cm_n"] += 1
print(f"[aggregate] total per-fact evaluations: {total_evals:,}")
print(f"[aggregate] non-empty cells: {sum(1 for b in bucket.values() if b['n']>0)}")

# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------
# 4a. Full CSV
csv_path = OUT / "fig3_full_table.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","src_group","nbr_class","hop","n_eval","clean_correct",
                "flip_count","epr","clean_margin","poisoned_margin","margin_change"])
    for (m,g,nc,h), b in sorted(bucket.items()):
        epr = b["flip"]/b["cc"] if b["cc"] else None
        cm = b["cm"]/b["cm_n"] if b["cm_n"] else None
        pm = b["pm"]/b["cm_n"] if b["cm_n"] else None
        dm = b["dm"]/b["cm_n"] if b["cm_n"] else None
        w.writerow([m,g,nc,h,b["n"],b["cc"],b["flip"],
                    f"{epr:.4f}" if epr is not None else "",
                    f"{cm:.4f}" if cm is not None else "",
                    f"{pm:.4f}" if pm is not None else "",
                    f"{dm:.4f}" if dm is not None else ""])
print(f"[write] {csv_path}")

# 4b. 2x2 matrix per model (src∈{Hub,Tail} × nbr∈{Hub,Tail}, mean d1-d5)
def pool(model, src_g, nbr_class):
    fl, cc, mc, mc_n = 0, 0, 0.0, 0
    for h in HOPS:
        b = bucket.get((model, src_g, nbr_class, h))
        if not b: continue
        fl += b["flip"]; cc += b["cc"]
        if b["cm_n"]:
            mc += b["dm"]; mc_n += b["cm_n"]
    return {
        "n_eval": sum(bucket.get((model,src_g,nbr_class,h),{}).get("n",0) for h in HOPS),
        "cc": cc, "flip": fl,
        "epr": fl/cc if cc else None,
        "dmargin": mc/mc_n if mc_n else None,
    }

md = ["# Fig 3 — Innocent Bystander (2×2 source × neighbor)", "",
      f"In-degree classes from 100k graph: Hub≥{HUB_THR}, Tail≤{TAIL_THR}", "",
      "Source group = subject of poisoning experiment.",
      "Neighbor class = in-degree of each evaluated `head` entity.",
      "Metric = sum-of-flips / sum-of-clean-correct over d1–d5 (sample-weighted),",
      "post v2 selection (10 targets/group) + GPT-4o-mini judge overturns.",
      ""]
for model in MODEL_ORDER:
    md.append(f"## {model}")
    md.append("")
    md.append("### EPR (post-judge)")
    md.append("")
    md.append("| Source ↓  /  Neighbor → | Hub | Mid | Tail |")
    md.append("|---|---|---|---|")
    for src in GROUPS:
        row = [f"**Src={src}**"]
        for nc in ["Hub","Mid","Tail"]:
            p = pool(model, src, nc)
            cell = f"{p['epr']:.3f} (n={p['cc']})" if p["epr"] is not None else f"-- (n={p['cc']})"
            row.append(cell)
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("### Δmargin (avg, clean→poisoned)")
    md.append("")
    md.append("| Source ↓  /  Neighbor → | Hub | Mid | Tail |")
    md.append("|---|---|---|---|")
    for src in GROUPS:
        row = [f"**Src={src}**"]
        for nc in ["Hub","Mid","Tail"]:
            p = pool(model, src, nc)
            cell = f"{p['dmargin']:+.2f}" if p["dmargin"] is not None else "--"
            row.append(cell)
        md.append("| " + " | ".join(row) + " |")
    md.append("")
(OUT/"fig3_2x2_per_model.md").write_text("\n".join(md))
print(f"[write] {OUT/'fig3_2x2_per_model.md'}")

# 4c. Cross-model aggregate (pooled over all 4 models)
md = ["# Fig 3 — Cross-Model Innocent Bystander", "",
      f"Pooled across {len(MODEL_ORDER)} models.", ""]
def pool_xm(src_g, nbr_class):
    fl, cc, mc, mc_n, n_eval = 0, 0, 0.0, 0, 0
    for m in MODEL_ORDER:
        p = pool(m, src_g, nbr_class)
        fl += p["flip"]; cc += p["cc"]; n_eval += p["n_eval"]
        if p["dmargin"] is not None:
            for h in HOPS:
                b = bucket.get((m,src_g,nbr_class,h),{})
                mc += b.get("dm",0); mc_n += b.get("cm_n",0)
    return {"n_eval":n_eval,"cc":cc,"flip":fl,
            "epr": fl/cc if cc else None,
            "dmargin": mc/mc_n if mc_n else None}

md.append("## EPR (cross-model pooled, post-judge)")
md.append("")
md.append("| Source ↓  /  Neighbor → | Hub | Mid | Tail |")
md.append("|---|---|---|---|")
for src in GROUPS:
    row = [f"**Src={src}**"]
    for nc in ["Hub","Mid","Tail"]:
        p = pool_xm(src, nc)
        cell = f"{p['epr']:.3f} (n={p['cc']:,})" if p["epr"] is not None else f"-- (n={p['cc']})"
        row.append(cell)
    md.append("| " + " | ".join(row) + " |")
md.append("")
md.append("## Δmargin (cross-model pooled)")
md.append("")
md.append("| Source ↓  /  Neighbor → | Hub | Mid | Tail |")
md.append("|---|---|---|---|")
for src in GROUPS:
    row = [f"**Src={src}**"]
    for nc in ["Hub","Mid","Tail"]:
        p = pool_xm(src, nc)
        cell = f"{p['dmargin']:+.2f}" if p["dmargin"] is not None else "--"
        row.append(cell)
    md.append("| " + " | ".join(row) + " |")
md.append("")
md.append("## Innocent-Bystander asymmetry test")
md.append("")
md.append("`Src=tail × Nbr=Hub` is the *innocent bystander* cell.")
md.append("If the paper's claim holds, this should be ≥ `Src=hub × Nbr=Tail`")
md.append("(low-pop poisoning still damages high-pop neighbors more than the reverse).")
md.append("")
md.append("| Asymmetry test | Value |")
md.append("|---|---|")
p_th_h = pool_xm("tail","Hub")
p_hu_t = pool_xm("hub","Tail")
p_th_t = pool_xm("tail","Tail")
p_hu_h = pool_xm("hub","Hub")
md.append(f"| Src=tail → Nbr=Hub EPR | {p_th_h['epr']:.3f} |" if p_th_h["epr"] is not None else "| Src=tail → Nbr=Hub | -- |")
md.append(f"| Src=hub → Nbr=Tail EPR | {p_hu_t['epr']:.3f} |" if p_hu_t["epr"] is not None else "| Src=hub → Nbr=Tail | -- |")
md.append(f"| Src=tail → Nbr=Tail EPR | {p_th_t['epr']:.3f} |" if p_th_t["epr"] is not None else "| Src=tail → Nbr=Tail | -- |")
md.append(f"| Src=hub → Nbr=Hub EPR | {p_hu_h['epr']:.3f} |" if p_hu_h["epr"] is not None else "| Src=hub → Nbr=Hub | -- |")
(OUT/"fig3_crossmodel.md").write_text("\n".join(md))
print(f"[write] {OUT/'fig3_crossmodel.md'}")

# 4d. Neighbor-class distribution sanity
print("\n[summary] Neighbor-class distribution by source group (pooled all 4 models):")
print(f"{'src':10s} {'Hub':>10s} {'Mid':>10s} {'Tail':>10s} {'Total':>10s}")
for src in GROUPS:
    h_n = sum(bucket.get((m,src,"Hub",h),{}).get("n",0) for m in MODEL_ORDER for h in HOPS)
    m_n = sum(bucket.get((m,src,"Mid",h),{}).get("n",0) for m in MODEL_ORDER for h in HOPS)
    t_n = sum(bucket.get((m,src,"Tail",h),{}).get("n",0) for m in MODEL_ORDER for h in HOPS)
    print(f"{src:10s} {h_n:>10,} {m_n:>10,} {t_n:>10,} {h_n+m_n+t_n:>10,}")

print("\n[done]")
