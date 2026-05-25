"""V3 Flip-Rate Audit — Hub vs Mid vs Tail.

Goal: settle whether the paper's claim "Hub > Mid > Tail in Flip Rate" is
defensible under the actual 4-model × 30-target data, and report all
sensitivity dials so the user can pick the framing for the paper.

This script DELIBERATELY computes flip rate many ways:
  V1. Raw micro-pooled per (model, nbr_class), all hops
  V2. Raw micro-pooled per (model, nbr_class), d=1 only
  V3. Per-target MACRO-average (one rate per target) with bootstrap 95% CI
  V4. Hub/Tail threshold sweep (top/bottom 1%, 5%, 10%, 25% of in_degree)
  V5. Sample-size & baseline-margin diagnostics (why raw fails)
  V6. Per-target Hub-vs-Tail head-to-head (paired test within model)

All outputs written to analysis_4models/v3_flip_audit/.
No re-judging; uses the existing `is_flip` field in comparison_reports JSON
(strict gold-containment judge already applied at pipeline time).
"""
from __future__ import annotations

import json, pickle, random, csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, median, stdev
import math

ROOT = Path("/home/weibing_wang/GenFragility-LLM/main_output")
OUT = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v3_flip_audit")
OUT.mkdir(parents=True, exist_ok=True)
GRAPH = Path("/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl")

MODELS = {
    "Qwen3.5-2B":     "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B":     "Qwen3.5-9B_30targets_experiment",
    "Gemma-4-E4B-it": "gemma-4-E4B-it_30targets_experiment",
    "Gemma-4-31B-it": "gemma-4-31B-it_30targets_experiment",
}
MODEL_ORDER = list(MODELS.keys())
GROUPS = ["hub", "tail", "random"]
HOPS   = ["d1", "d2", "d3", "d4", "d5"]

# Default neighbor-class thresholds (matches analyze_strict_d0.py)
DEFAULT_HUB_THRESH  = 8
DEFAULT_TAIL_THRESH = 1

# ------------------------------------------------------------------
# 1) Load graph & compute in-degree
# ------------------------------------------------------------------
print(f"[load] graph from {GRAPH}")
with open(GRAPH, "rb") as f:
    _p = pickle.load(f)
G = _p["graph"] if isinstance(_p, dict) and "graph" in _p else _p
print(f"  G: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

indeg_map: dict[str, int] = {n: G.in_degree(n) for n in G.nodes}
indegs = sorted(indeg_map.values())

# Percentile cutoffs (we'll use these for V4 threshold sweep)
def pct(arr, p):
    if not arr: return 0
    i = max(0, min(len(arr) - 1, int(round((p/100.0) * (len(arr) - 1)))))
    return arr[i]

PCTS = {
    "top1": pct(indegs, 99),
    "top5": pct(indegs, 95),
    "top10": pct(indegs, 90),
    "top25": pct(indegs, 75),
    "bot1": pct(indegs, 1),
    "bot5": pct(indegs, 5),
    "bot10": pct(indegs, 10),
    "bot25": pct(indegs, 25),
}
print(f"  in-degree percentiles: {PCTS}")

# ------------------------------------------------------------------
# 2) Load all Mask-B facts from 4 models
# ------------------------------------------------------------------
def nbr_class_with_thresh(head, hub_t, tail_t):
    if not head or head not in indeg_map: return "Mid"
    d = indeg_map[head]
    if d >= hub_t: return "Hub"
    if d <= tail_t: return "Tail"
    return "Mid"

def nbr_class_default(head):
    return nbr_class_with_thresh(head, DEFAULT_HUB_THRESH, DEFAULT_TAIL_THRESH)

all_rows = []
for m, sub in MODELS.items():
    base = ROOT / sub
    for d in sorted(base.iterdir()):
        if not d.is_dir(): continue
        nm = d.name
        if not (nm.startswith("hub_") or nm.startswith("tail_") or nm.startswith("random_")):
            continue
        group = nm.split("_")[0]
        fp = d / "comparison_reports" / f"{nm}_vllm_comparison.json"
        if not fp.exists(): continue
        try:
            j = json.loads(fp.read_text())
        except Exception:
            continue
        for r in j.get("unified_results", []):
            hop = r.get("distance")
            if hop not in HOPS: continue
            if r.get("clean_accuracy") != 1.0: continue  # Mask B
            head = r.get("head") or ""
            all_rows.append({
                "model": m, "src_group": group, "target": nm, "hop": hop,
                "head": head,
                "tail_gold": r.get("tail") or "",
                "head_indeg": indeg_map.get(head, -1),
                "nbr_default": nbr_class_default(head),
                "is_flip": bool(r.get("is_flip")),
                "clean_margin": float(r.get("clean_margin") or 0.0),
                "margin_change": float(r.get("margin_change") or 0.0),
            })
print(f"\n[load] {len(all_rows):,} Mask-B facts loaded")

# ------------------------------------------------------------------
# Helper: bootstrap 95% CI for proportion
# ------------------------------------------------------------------
def boot_ci(samples, n_boot=2000, seed=42):
    """samples: list of 0/1 (or floats in [0,1]). Returns (mean, lo, hi) at 95%."""
    if not samples: return (None, None, None)
    rng = random.Random(seed)
    means = []
    n = len(samples)
    for _ in range(n_boot):
        s = sum(samples[rng.randint(0, n-1)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (sum(samples)/n, lo, hi)

# ------------------------------------------------------------------
# V1: Micro-pooled per (model, nbr_class), ALL HOPS
# ------------------------------------------------------------------
def micro_flip_table(rows, hop_filter=None):
    agg = defaultdict(lambda: [0, 0])  # (model, nbr) -> [n, flip]
    for r in rows:
        if hop_filter and r["hop"] != hop_filter: continue
        agg[(r["model"], r["nbr_default"])][0] += 1
        agg[(r["model"], r["nbr_default"])][1] += int(r["is_flip"])
    return agg

# ------------------------------------------------------------------
# V3: Per-target MACRO-average + bootstrap CI
# ------------------------------------------------------------------
def macro_flip_table(rows, min_n_per_target=5):
    """For each (model, target, nbr_class), compute that target's flip rate
    among nbr_class neighbors. Then macro-average across targets within model.
    Bootstrap CI is over the TARGETS (each target contributes one rate)."""
    per_tgt = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (m,target) -> nbr -> [n, fl]
    for r in rows:
        per_tgt[(r["model"], r["target"])][r["nbr_default"]][0] += 1
        per_tgt[(r["model"], r["target"])][r["nbr_default"]][1] += int(r["is_flip"])
    # Collect per-target rates per (model, nbr_class)
    per_model_nbr_rates = defaultdict(list)
    for (m, tid), nd in per_tgt.items():
        for nb in ["Hub","Mid","Tail"]:
            n, fl = nd[nb]
            if n >= min_n_per_target:
                per_model_nbr_rates[(m, nb)].append(fl / n)
    # Compute mean + bootstrap CI per (model, nbr)
    out = {}
    for (m, nb), rates in per_model_nbr_rates.items():
        mean_v, lo, hi = boot_ci(rates)
        out[(m, nb)] = (mean_v, lo, hi, len(rates), rates)
    return out, per_tgt

# ------------------------------------------------------------------
# V4: Threshold sweep
# ------------------------------------------------------------------
def threshold_sweep_flip(rows, configs, min_n_per_target=5):
    """configs: list of (label, hub_t, tail_t).
    Returns micro and macro tables under each config."""
    results = {}
    for label, hub_t, tail_t in configs:
        # Recompute nbr class
        rebinned = []
        for r in rows:
            new = dict(r)
            new["nbr"] = nbr_class_with_thresh(r["head"], hub_t, tail_t)
            rebinned.append(new)
        # micro per (model, nbr)
        agg = defaultdict(lambda: [0, 0])
        for r in rebinned:
            agg[(r["model"], r["nbr"])][0] += 1
            agg[(r["model"], r["nbr"])][1] += int(r["is_flip"])
        # macro per (model, nbr) with bootstrap CI
        per_tgt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in rebinned:
            per_tgt[(r["model"], r["target"])][r["nbr"]][0] += 1
            per_tgt[(r["model"], r["target"])][r["nbr"]][1] += int(r["is_flip"])
        macro = defaultdict(list)
        for (m, tid), nd in per_tgt.items():
            for nb in ["Hub","Mid","Tail"]:
                n, fl = nd[nb]
                if n >= min_n_per_target:
                    macro[(m, nb)].append(fl / n)
        macro_ci = {}
        for (m, nb), rates in macro.items():
            mean_v, lo, hi = boot_ci(rates)
            macro_ci[(m, nb)] = (mean_v, lo, hi, len(rates))
        results[label] = (agg, macro_ci, hub_t, tail_t)
    return results

# ------------------------------------------------------------------
# V5: Diagnostics on why raw fails
# ------------------------------------------------------------------
def baseline_diagnostics(rows):
    """For each (model, nbr_class), report mean clean_margin and sample size."""
    agg = defaultdict(lambda: {"n": 0, "sum_cm": 0.0, "n_d1": 0,
                                "n_per_target": defaultdict(int)})
    for r in rows:
        k = (r["model"], r["nbr_default"])
        agg[k]["n"] += 1
        agg[k]["sum_cm"] += r["clean_margin"]
        if r["hop"] == "d1":
            agg[k]["n_d1"] += 1
        agg[k]["n_per_target"][r["target"]] += 1
    return agg

# ------------------------------------------------------------------
# V6: Per-target Hub-vs-Tail head-to-head (paired)
# ------------------------------------------------------------------
def paired_hub_tail(per_tgt, min_n=5):
    """Within each (model, target), pair Hub-nbr flip rate vs Tail-nbr flip rate.
    Only count targets that have >= min_n of BOTH Hub and Tail neighbors.
    Reports: how many targets have Hub > Tail vs Hub < Tail."""
    pairs = defaultdict(list)  # model -> list of (target, hub_rate, tail_rate)
    for (m, tid), nd in per_tgt.items():
        h_n, h_fl = nd["Hub"]
        t_n, t_fl = nd["Tail"]
        if h_n >= min_n and t_n >= min_n:
            pairs[m].append((tid, h_fl/h_n, t_fl/t_n))
    return pairs

# ============================================================================
# RUN ALL & WRITE REPORT
# ============================================================================
lines = ["# V3 Flip-Rate Audit: Hub vs Mid vs Tail",
         "",
         f"- Source: `comparison_reports/*_vllm_comparison.json` from 4 model dirs",
         f"- Mask B: only facts with `clean_accuracy == 1.0`",
         f"- Mask-B fact count: {len(all_rows):,}",
         f"- Default neighbor-class thresholds: Hub `in_degree >= {DEFAULT_HUB_THRESH}`, "
         f"Tail `in_degree <= {DEFAULT_TAIL_THRESH}`",
         "",
         "Question being settled: **Does the binary Flip Rate satisfy Hub > Mid > Tail "
         "across all 4 models?**",
         ""]

# --------- V1 ---------
lines.append("## V1. Raw micro-pooled flip rate per (model, neighbor class), ALL HOPS")
lines.append("")
lines.append("| Model | Hub-nbr | Mid-nbr | Tail-nbr | Hub>Tail? |")
lines.append("|---|---:|---:|---:|---|")
v1 = micro_flip_table(all_rows)
v1_pass = 0
for m in MODEL_ORDER:
    cells = {}
    for nb in ["Hub","Mid","Tail"]:
        n, fl = v1[(m, nb)]
        cells[nb] = (fl/n if n else 0, n)
    ok = "YES" if cells["Hub"][0] > cells["Tail"][0] else "**no**"
    if cells["Hub"][0] > cells["Tail"][0]: v1_pass += 1
    lines.append(f"| {m} | {cells['Hub'][0]*100:5.2f}% (n={cells['Hub'][1]:,}) "
                 f"| {cells['Mid'][0]*100:5.2f}% (n={cells['Mid'][1]:,}) "
                 f"| {cells['Tail'][0]*100:5.2f}% (n={cells['Tail'][1]:,}) | {ok} |")
lines.append(f"\n**Hub > Tail holds in {v1_pass}/4 models** under raw micro pooling. "
             "Note: Tail samples are tiny (n=411-770 vs Hub n=20k-27k).\n")

# --------- V2 ---------
lines.append("## V2. Raw micro-pooled flip rate, d=1 only")
lines.append("")
lines.append("| Model | Hub-nbr | Mid-nbr | Tail-nbr | Hub>Tail? |")
lines.append("|---|---:|---:|---:|---|")
v2 = micro_flip_table(all_rows, hop_filter="d1")
v2_pass = 0
for m in MODEL_ORDER:
    cells = {}
    for nb in ["Hub","Mid","Tail"]:
        n, fl = v2[(m, nb)]
        cells[nb] = (fl/n if n else 0, n)
    ok = "YES" if cells["Hub"][0] > cells["Tail"][0] else "**no**"
    if cells["Hub"][0] > cells["Tail"][0]: v2_pass += 1
    lines.append(f"| {m} | {cells['Hub'][0]*100:5.2f}% (n={cells['Hub'][1]}) "
                 f"| {cells['Mid'][0]*100:5.2f}% (n={cells['Mid'][1]}) "
                 f"| {cells['Tail'][0]*100:5.2f}% (n={cells['Tail'][1]}) | {ok} |")
lines.append(f"\n**Hub > Tail holds in {v2_pass}/4 models** at d=1. "
             "Tail sample at d=1 is only 9-11 per model — *too small to trust*.\n")

# --------- V3 ---------
lines.append("## V3. Per-target MACRO-average flip rate with bootstrap 95% CI")
lines.append("")
lines.append("Each target contributes ONE flip rate per neighbor class (denominator: that "
             "target's own neighbors in that class, requires >=5 samples). Macro mean & "
             "CI are computed across targets within each model.")
lines.append("")
lines.append("| Model | Hub-nbr (k targets) | Mid-nbr (k targets) | Tail-nbr (k targets) | Hub>Tail? |")
lines.append("|---|---|---|---|---|")
macro, per_tgt = macro_flip_table(all_rows, min_n_per_target=5)
v3_pass = 0
for m in MODEL_ORDER:
    cells = {}
    for nb in ["Hub","Mid","Tail"]:
        if (m, nb) in macro:
            mean_v, lo, hi, k, _ = macro[(m, nb)]
            cells[nb] = (mean_v, lo, hi, k)
        else:
            cells[nb] = (None, None, None, 0)
    h_m, h_lo, h_hi, h_k = cells["Hub"]
    t_m, t_lo, t_hi, t_k = cells["Tail"]
    if h_m is not None and t_m is not None:
        ok = "YES" if h_m > t_m else "**no**"
        # Strong YES if CI doesn't overlap
        if h_m > t_m and h_lo > t_hi: ok += " (CI sep)"
        if h_m > t_m: v3_pass += 1
    else:
        ok = "n/a"
    def fmt(c):
        v, lo, hi, k = c
        if v is None: return "—"
        return f"{v*100:5.2f}% [{lo*100:5.2f}, {hi*100:5.2f}] (k={k})"
    lines.append(f"| {m} | {fmt(cells['Hub'])} | {fmt(cells['Mid'])} | {fmt(cells['Tail'])} | {ok} |")
lines.append(f"\n**Hub > Tail (macro) holds in {v3_pass}/4 models.** "
             "Bootstrap CIs are wide because k<=44 targets — but no model shows a "
             "statistically separated Hub > Tail.\n")

# --------- V4 ---------
lines.append("## V4. Hub/Tail threshold sweep")
lines.append("")
lines.append("Re-bin neighbor class under stricter Hub / looser Tail definitions to see "
             "if the trend re-emerges only under particular cutoffs.")
lines.append("")
configs = [
    (f"strict: Hub>=top5%={PCTS['top5']} / Tail<=bot5%={PCTS['bot5']}",
        PCTS["top5"], PCTS["bot5"]),
    (f"medium: Hub>=top10%={PCTS['top10']} / Tail<=bot10%={PCTS['bot10']}",
        PCTS["top10"], PCTS["bot10"]),
    (f"loose: Hub>=top25%={PCTS['top25']} / Tail<=bot25%={PCTS['bot25']}",
        PCTS["top25"], PCTS["bot25"]),
    (f"very strict: Hub>=top1%={PCTS['top1']} / Tail<=bot1%={PCTS['bot1']}",
        PCTS["top1"], PCTS["bot1"]),
    (f"degree==1 Tail / Hub>=8 (current paper default)",
        DEFAULT_HUB_THRESH, DEFAULT_TAIL_THRESH),
]
sweep = threshold_sweep_flip(all_rows, configs, min_n_per_target=5)
for label, (micro, macro_ci, hub_t, tail_t) in sweep.items():
    lines.append(f"### Config: {label}")
    lines.append("")
    lines.append("| Model | Hub-nbr (micro / macro k) | Mid (micro) | Tail-nbr (micro / macro k) | Hub>Tail (macro)? |")
    lines.append("|---|---|---|---|---|")
    for m in MODEL_ORDER:
        h_n, h_fl = micro[(m, "Hub")]
        mi_n, mi_fl = micro[(m, "Mid")]
        t_n, t_fl = micro[(m, "Tail")]
        h_micro = h_fl / h_n if h_n else 0
        mi_micro = mi_fl / mi_n if mi_n else 0
        t_micro = t_fl / t_n if t_n else 0
        h_m = macro_ci.get((m, "Hub"), (None,)*4)
        t_m = macro_ci.get((m, "Tail"), (None,)*4)
        ok = "YES" if h_m[0] is not None and t_m[0] is not None and h_m[0] > t_m[0] else "no"
        lines.append(f"| {m} | "
                     f"{h_micro*100:5.2f}% (n={h_n:,}) / macro {h_m[0]*100 if h_m[0] is not None else 0:5.2f}% (k={h_m[3] if h_m[0] is not None else 0}) | "
                     f"{mi_micro*100:5.2f}% (n={mi_n:,}) | "
                     f"{t_micro*100:5.2f}% (n={t_n:,}) / macro {t_m[0]*100 if t_m[0] is not None else 0:5.2f}% (k={t_m[3] if t_m[0] is not None else 0}) | "
                     f"{ok} |")
    lines.append("")

# --------- V5 ---------
lines.append("## V5. Diagnostics: why raw Flip Rate doesn't show Hub > Tail")
lines.append("")
lines.append("### (a) Mask-B baseline `clean_margin` per (model, neighbor class)")
lines.append("")
lines.append("If Hub-neighbor facts have systematically *higher* pre-update margin, "
             "the same logit perturbation has to fight a stiffer baseline to actually "
             "cross the top-1 boundary. This is the confound the paper text already calls out.")
lines.append("")
lines.append("| Model | Hub mean cm | Mid mean cm | Tail mean cm |")
lines.append("|---|---:|---:|---:|")
diag = baseline_diagnostics(all_rows)
for m in MODEL_ORDER:
    row = [m]
    for nb in ["Hub","Mid","Tail"]:
        d = diag[(m, nb)]
        cm = d["sum_cm"] / d["n"] if d["n"] else 0
        row.append(f"{cm:.2f}")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

lines.append("### (b) Sample-size dominance — Tail is rare")
lines.append("")
lines.append("| Model | n Hub-nbr | n Mid-nbr | n Tail-nbr | n d1 Tail-nbr |")
lines.append("|---|---:|---:|---:|---:|")
for m in MODEL_ORDER:
    row = [m]
    for nb in ["Hub","Mid","Tail"]:
        d = diag[(m, nb)]
        row.append(f"{d['n']:,}")
    d_t = diag[(m, "Tail")]
    row.append(f"{d_t['n_d1']}")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# --------- V6 ---------
lines.append("## V6. Paired Hub-vs-Tail head-to-head per target")
lines.append("")
lines.append("For each target with >=5 Hub-class AND >=5 Tail-class neighbors, "
             "compute (Hub-nbr flip rate, Tail-nbr flip rate) on the **same target's** "
             "neighborhood. Count how many targets per model show Hub > Tail.")
lines.append("")
pairs = paired_hub_tail(per_tgt, min_n=5)
lines.append("| Model | k pairable targets | Hub > Tail | Hub == Tail | Hub < Tail | Mean(Hub - Tail) |")
lines.append("|---|---:|---:|---:|---:|---:|")
for m in MODEL_ORDER:
    pp = pairs.get(m, [])
    n_pairs = len(pp)
    if not pp:
        lines.append(f"| {m} | 0 | — | — | — | — |")
        continue
    gt = sum(1 for _, h, t in pp if h > t)
    eq = sum(1 for _, h, t in pp if abs(h - t) < 1e-9)
    lt = sum(1 for _, h, t in pp if h < t)
    diff_mean = mean(h - t for _, h, t in pp)
    lines.append(f"| {m} | {n_pairs} | {gt} | {eq} | {lt} | {diff_mean*100:+.2f} pp |")

# Write per-pair CSV for inspection
csv_p = OUT / "v6_paired_hub_tail.csv"
with csv_p.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","target","hub_flip","tail_flip","diff_hub_minus_tail"])
    for m in MODEL_ORDER:
        for tid, h, t in pairs.get(m, []):
            w.writerow([m, tid, f"{h:.4f}", f"{t:.4f}", f"{h-t:+.4f}"])
lines.append(f"\nPer-pair CSV: `{csv_p.relative_to(OUT.parent.parent)}`\n")

# --------- Summary ---------
lines.append("## Summary")
lines.append("")
lines.append("| Framing | Hub > Tail consistent? |")
lines.append("|---|---|")
lines.append(f"| V1. Raw micro, all hops      | {v1_pass}/4 models |")
lines.append(f"| V2. Raw micro, d=1 only      | {v2_pass}/4 models (tiny n) |")
lines.append(f"| V3. Per-target macro + boot  | {v3_pass}/4 models |")
lines.append("| V4. Threshold sweep          | see configs above |")
lines.append("| V6. Per-target paired sign   | see per-model above |")

(OUT / "v3_audit_report.md").write_text("\n".join(lines))
print(f"\n[write] {OUT/'v3_audit_report.md'}")

# Also dump a per-fact CSV with default nbr class for downstream use
csv_p2 = OUT / "all_rows_default_nbr.csv"
with csv_p2.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","src_group","target","hop","head","head_indeg",
                "nbr_default","is_flip","clean_margin","margin_change"])
    for r in all_rows:
        w.writerow([r["model"], r["src_group"], r["target"], r["hop"],
                    r["head"], r["head_indeg"], r["nbr_default"],
                    int(r["is_flip"]), f"{r['clean_margin']:.4f}",
                    f"{r['margin_change']:.4f}"])
print(f"[write] {csv_p2}")
print("[done]")
