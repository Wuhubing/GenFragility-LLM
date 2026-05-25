"""Aggregate paper-relevant metrics across 4 models (Qwen3.5-2B/9B, Gemma-4-E4B-it/31B-it).

Produces:
  tables/per_target.csv            - one row per (model, target, hop)
  tables/agg_by_group.csv          - mean/std/n per (model, group, hop)
  tables/fig1_epr_table.md         - human-readable EPR table (Fig 1)
  tables/fig2a_flip_rate_d1.md     - Flip Rate at d=1 (Fig 2a)
  tables/fig2b_epr_by_source.md    - EPR by source type (Fig 2b)
  tables/margin_table.md           - clean / poisoned margin per group
  tables/confidence_shift.md       - avg_tail_lp_change per group (proxies dConf)
  figures/fig1_blast_radius.pdf
  figures/fig2a_flip_rate_d1.pdf
  figures/fig2b_epr_by_source.pdf
  figures/fig_margin_by_hop.pdf

Inputs are read STRICTLY from comparison_reports/*_vllm_comparison.json
(skipping OLD_BROKEN.json). All output goes under analysis_4models/.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/weibing_wang/GenFragility-LLM/main_output")
OUT  = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models")
TBL  = OUT / "tables"
FIG  = OUT / "figures"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

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
# 1. Load
# ---------------------------------------------------------------------------
def load_comparison(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # corrupt file
        print(f"  [warn] failed to read {path}: {exc}")
        return None


def collect():
    """Yield (model, group, target_id, hop_stats_dict, unified_results)."""
    for model, folder in MODELS.items():
        base = ROOT / folder
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            parts = sub.name.split("_")
            if parts[0] not in GROUPS or not parts[-1].isdigit():
                continue
            group = parts[0]
            tid   = sub.name
            jsons = sorted(
                p for p in (sub / "comparison_reports").glob("*_vllm_comparison.json")
                if "OLD_BROKEN" not in p.name
            )
            if not jsons:
                continue
            doc = load_comparison(jsons[0])
            if doc is None:
                continue
            yield model, group, tid, doc


# ---------------------------------------------------------------------------
# 2. Build per-target table
# ---------------------------------------------------------------------------
per_target_rows = []           # flat rows, one per (model, target, hop)
unified_rows = []              # per-fact rows (used for Mask B / confidence)

print("[load] reading comparison JSONs ...")
for model, group, tid, doc in collect():
    stats = doc.get("comparison_statistics", {})
    poison = doc.get("poison_info", {})
    for hop in HOPS:
        s = stats.get(hop)
        if not s:
            continue
        per_target_rows.append({
            "model":             model,
            "group":             group,
            "target":            tid,
            "subject":           poison.get("subject"),
            "hop":               hop,
            "count":             s.get("count"),
            "clean_accuracy":    s.get("clean_accuracy"),
            "poisoned_accuracy": s.get("poisoned_accuracy"),
            "accuracy_drop":    (s.get("clean_accuracy") - s.get("poisoned_accuracy"))
                                 if (s.get("clean_accuracy") is not None
                                     and s.get("poisoned_accuracy") is not None) else None,
            "epr":               s.get("epr"),
            "flip_rate":         s.get("flip_rate"),
            "flip_count":        s.get("flip_count"),
            "clean_correct":     s.get("clean_correct"),
            "clean_margin_avg":  s.get("clean_margin_avg"),
            "poisoned_margin_avg": s.get("poisoned_margin_avg"),
            "margin_change_avg": s.get("margin_change_avg"),
        })

    # per-fact rows (for Mask B confidence shift)
    for r in doc.get("unified_results", []):
        if r.get("distance") not in HOPS:
            continue
        unified_rows.append({
            "model":   model,
            "group":   group,
            "target":  tid,
            "distance": r["distance"],
            "clean_accuracy":  r.get("clean_accuracy"),
            "is_flip":         r.get("is_flip"),
            "clean_margin":    r.get("clean_margin"),
            "poisoned_margin": r.get("poisoned_margin"),
            "margin_change":   r.get("margin_change"),
            "clean_lp":        r.get("clean_avg_tail_log_probability"),
            "poisoned_lp":     r.get("poisoned_avg_tail_log_probability"),
            "avg_tail_lp_change": r.get("avg_tail_lp_change"),
        })

print(f"[load] per_target_rows = {len(per_target_rows)}, unified_rows = {len(unified_rows)}")


# ---------------------------------------------------------------------------
# 3. Write per-target CSV
# ---------------------------------------------------------------------------
def write_csv(path, rows, cols):
    import csv
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

per_target_cols = ["model", "group", "target", "subject", "hop", "count",
                   "clean_accuracy", "poisoned_accuracy", "accuracy_drop",
                   "epr", "flip_rate", "flip_count", "clean_correct",
                   "clean_margin_avg", "poisoned_margin_avg", "margin_change_avg"]
write_csv(TBL / "per_target.csv", per_target_rows, per_target_cols)
print(f"[write] {TBL/'per_target.csv'}")


# ---------------------------------------------------------------------------
# 4. Aggregate by (model, group, hop) -- weighted mean by `count`
# ---------------------------------------------------------------------------
def agg_weighted(rows, val_key, weight_key="count"):
    """Sample-weighted mean across targets; ignore None."""
    s, w = 0.0, 0.0
    for r in rows:
        v, n = r.get(val_key), r.get(weight_key)
        if v is None or n is None: continue
        s += v * n
        w += n
    return s / w if w else None

def agg_mean(rows, val_key):
    """Unweighted mean across targets (one vote per target)."""
    vals = [r[val_key] for r in rows if r.get(val_key) is not None]
    return mean(vals) if vals else None

def agg_std(rows, val_key):
    vals = [r[val_key] for r in rows if r.get(val_key) is not None]
    return pstdev(vals) if len(vals) > 1 else 0.0

agg_rows = []
buckets = defaultdict(list)
for r in per_target_rows:
    buckets[(r["model"], r["group"], r["hop"])].append(r)

for (model, group, hop), rs in buckets.items():
    agg_rows.append({
        "model": model, "group": group, "hop": hop,
        "n_targets": len(rs),
        "n_samples_total": sum(r["count"] for r in rs if r["count"] is not None),
        # primary: sample-weighted (matches paper's EPR aggregation)
        "epr_weighted":          agg_weighted(rs, "epr"),
        "flip_rate_weighted":    agg_weighted(rs, "flip_rate"),
        "clean_acc_weighted":    agg_weighted(rs, "clean_accuracy"),
        "poisoned_acc_weighted": agg_weighted(rs, "poisoned_accuracy"),
        "accuracy_drop_weighted":agg_weighted(rs, "accuracy_drop"),
        "clean_margin_weighted": agg_weighted(rs, "clean_margin_avg"),
        "poisoned_margin_weighted": agg_weighted(rs, "poisoned_margin_avg"),
        "margin_change_weighted":agg_weighted(rs, "margin_change_avg"),
        # secondary: per-target mean (per-target voting)
        "epr_target_mean":          agg_mean(rs, "epr"),
        "epr_target_std":           agg_std(rs, "epr"),
        "flip_rate_target_mean":    agg_mean(rs, "flip_rate"),
        "accuracy_drop_target_mean":agg_mean(rs, "accuracy_drop"),
        "clean_margin_target_mean": agg_mean(rs, "clean_margin_avg"),
        "margin_change_target_mean":agg_mean(rs, "margin_change_avg"),
    })

agg_cols = ["model", "group", "hop", "n_targets", "n_samples_total",
            "epr_weighted", "flip_rate_weighted",
            "clean_acc_weighted", "poisoned_acc_weighted", "accuracy_drop_weighted",
            "clean_margin_weighted", "poisoned_margin_weighted", "margin_change_weighted",
            "epr_target_mean", "epr_target_std",
            "flip_rate_target_mean", "accuracy_drop_target_mean",
            "clean_margin_target_mean", "margin_change_target_mean"]
write_csv(TBL / "agg_by_group.csv", agg_rows, agg_cols)
print(f"[write] {TBL/'agg_by_group.csv'}")


# ---------------------------------------------------------------------------
# 5. Mask B (clean_accuracy == 1) per-fact confidence shift (proxy dConf)
# ---------------------------------------------------------------------------
mask_b = [r for r in unified_rows if r.get("clean_accuracy") == 1.0]
print(f"[mask-b] retained {len(mask_b)} of {len(unified_rows)} per-fact rows")

dconf_rows = []
buckets_b = defaultdict(list)
for r in mask_b:
    buckets_b[(r["model"], r["group"], r["distance"])].append(r)

for (model, group, hop), rs in buckets_b.items():
    lp_diffs = [r["avg_tail_lp_change"] for r in rs if r.get("avg_tail_lp_change") is not None]
    margin_diffs = [r["margin_change"] for r in rs if r.get("margin_change") is not None]
    flips = [1 if r.get("is_flip") else 0 for r in rs]
    dconf_rows.append({
        "model": model, "group": group, "hop": hop,
        "n_mask_b": len(rs),
        # dConf proxy = exp(lp_after) - exp(lp_before); but lp here is avg log prob of the tail span,
        # so we report mean dLP and mean delta-prob for transparency
        "mean_dLP":   mean(lp_diffs) if lp_diffs else None,
        "median_dLP": float(np.median(lp_diffs)) if lp_diffs else None,
        "mean_dProb": mean([np.exp(r["poisoned_lp"]) - np.exp(r["clean_lp"])
                            for r in rs if r.get("clean_lp") is not None and r.get("poisoned_lp") is not None])
                       if rs else None,
        "mean_margin_change": mean(margin_diffs) if margin_diffs else None,
        "flip_rate_mask_b":    sum(flips) / len(flips) if flips else None,
    })
write_csv(TBL / "confidence_shift_mask_b.csv", dconf_rows,
          ["model","group","hop","n_mask_b","mean_dLP","median_dLP","mean_dProb",
           "mean_margin_change","flip_rate_mask_b"])
print(f"[write] {TBL/'confidence_shift_mask_b.csv'}")


# ---------------------------------------------------------------------------
# 6. Markdown tables for the paper
# ---------------------------------------------------------------------------
def _get(agg_rows, model, group, hop, key):
    for r in agg_rows:
        if r["model"] == model and r["group"] == group and r["hop"] == hop:
            return r[key]
    return None

def fmt(x, d=3):
    if x is None: return "--"
    return f"{x:.{d}f}"

# (a) Fig 1 EPR table
lines = ["# Fig 1: Error Propagation Rate (EPR) across hops",
         "",
         "Sample-weighted EPR per (Model × Group × Hop). Hub-source is the most"
         " informative; Tail/Random are baselines.",
         ""]
for group in GROUPS:
    lines += [f"## Group = {group}", "",
              "| Model | " + " | ".join(HOPS) + " | mean(d1-d5) |",
              "|---|" + "---|" * (len(HOPS)+1)]
    for m in MODEL_ORDER:
        vals = [_get(agg_rows, m, group, h, "epr_weighted") for h in HOPS]
        avg = mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        lines.append("| " + m + " | " + " | ".join(fmt(v) for v in vals) + " | " + fmt(avg) + " |")
    lines.append("")
(TBL / "fig1_epr_table.md").write_text("\n".join(lines))

# (b) Fig 2(a) Flip Rate at d=1, Hub vs Tail
lines = ["# Fig 2(a): Flip Rate at d=1 (Hub vs Tail vs Random)",
         "",
         "| Model | Hub | Tail | Random |",
         "|---|---|---|---|"]
for m in MODEL_ORDER:
    h = _get(agg_rows, m, "hub",    "d1", "flip_rate_weighted")
    t = _get(agg_rows, m, "tail",   "d1", "flip_rate_weighted")
    rd= _get(agg_rows, m, "random", "d1", "flip_rate_weighted")
    lines.append(f"| {m} | {fmt(h)} | {fmt(t)} | {fmt(rd)} |")
(TBL / "fig2a_flip_rate_d1.md").write_text("\n".join(lines))

# (c) Fig 2(b) EPR by source type (averaged across d1..d5)
lines = ["# Fig 2(b): EPR by source type (mean over d1-d5)",
         "",
         "| Model | Hub-src EPR | Tail-src EPR | Random-src EPR |",
         "|---|---|---|---|"]
for m in MODEL_ORDER:
    row = []
    for g in GROUPS:
        vals = [_get(agg_rows, m, g, h, "epr_weighted") for h in HOPS]
        row.append(mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None)
    lines.append(f"| {m} | {fmt(row[0])} | {fmt(row[1])} | {fmt(row[2])} |")
(TBL / "fig2b_epr_by_source.md").write_text("\n".join(lines))

# (d) Margin table
lines = ["# Mechanism: clean / Δ Margin per hop (sample-weighted)",
         ""]
for group in GROUPS:
    lines += [f"## Group = {group}", "",
              "| Model | metric | " + " | ".join(HOPS) + " |",
              "|---|---|" + "---|" * len(HOPS)]
    for m in MODEL_ORDER:
        cm = [_get(agg_rows, m, group, h, "clean_margin_weighted")   for h in HOPS]
        dm = [_get(agg_rows, m, group, h, "margin_change_weighted")  for h in HOPS]
        lines.append(f"| {m} | clean_margin | " + " | ".join(fmt(v,2) for v in cm) + " |")
        lines.append(f"| {m} | Δmargin      | " + " | ".join(fmt(v,2) for v in dm) + " |")
    lines.append("")
(TBL / "margin_table.md").write_text("\n".join(lines))

# (e) Confidence shift table from Mask B
def _getb(model, group, hop, key):
    for r in dconf_rows:
        if r["model"] == model and r["group"] == group and r["hop"] == hop:
            return r[key]
    return None
lines = ["# Confidence shift under Mask B (clean_accuracy == 1)", "",
         "* mean_dLP: mean change in avg-tail-log-probability (post - pre)",
         "* mean_dProb: mean change in actual probability (exp(lp_post) - exp(lp_pre))",
         "* flip_rate_mask_b: flip rate inside Mask B subset",
         ""]
for group in GROUPS:
    lines += [f"## Group = {group}", "",
              "| Model | metric | " + " | ".join(HOPS) + " |",
              "|---|---|" + "---|" * len(HOPS)]
    for m in MODEL_ORDER:
        dlp  = [_getb(m, group, h, "mean_dLP")          for h in HOPS]
        dpb  = [_getb(m, group, h, "mean_dProb")        for h in HOPS]
        flb  = [_getb(m, group, h, "flip_rate_mask_b")  for h in HOPS]
        lines.append(f"| {m} | mean_dLP    | " + " | ".join(fmt(v,3) for v in dlp) + " |")
        lines.append(f"| {m} | mean_dProb  | " + " | ".join(fmt(v,3) for v in dpb) + " |")
        lines.append(f"| {m} | flip_MaskB  | " + " | ".join(fmt(v,3) for v in flb) + " |")
    lines.append("")
(TBL / "confidence_shift.md").write_text("\n".join(lines))

print("[write] all markdown tables ->", TBL)


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def _series(model, group, key):
    return [_get(agg_rows, model, group, h, key) for h in HOPS]

# Fig 1 — Blast Radius (EPR vs hop), one panel per group
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, group in zip(axes, GROUPS):
    for m in MODEL_ORDER:
        ys = _series(m, group, "epr_weighted")
        ax.plot(HOPS, [y if y is not None else np.nan for y in ys], marker="o", label=m)
    ax.set_title(f"Source = {group}")
    ax.set_xlabel("hop distance")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
axes[0].set_ylabel("Error Propagation Rate (EPR)")
axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("Fig 1 — Extent of Error Propagation across 4 model scales")
fig.tight_layout()
fig.savefig(FIG / "fig1_blast_radius.pdf")
fig.savefig(FIG / "fig1_blast_radius.png", dpi=150)
plt.close(fig)

# Fig 2(a) — Flip Rate at d=1 (Hub vs Tail vs Random)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(MODEL_ORDER))
w = 0.27
for i, group in enumerate(GROUPS):
    vals = [_get(agg_rows, m, group, "d1", "flip_rate_weighted") or 0 for m in MODEL_ORDER]
    ax.bar(x + (i-1)*w, vals, w, label=group)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, rotation=15)
ax.set_ylabel("Flip Rate @ d=1")
ax.set_title("Fig 2(a) — Vulnerability at d=1 (Hub vs Tail vs Random)")
ax.legend()
ax.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "fig2a_flip_rate_d1.pdf")
fig.savefig(FIG / "fig2a_flip_rate_d1.png", dpi=150)
plt.close(fig)

# Fig 2(b) — EPR by source (mean d1..d5)
fig, ax = plt.subplots(figsize=(7, 4))
for i, group in enumerate(GROUPS):
    vals = []
    for m in MODEL_ORDER:
        ys = _series(m, group, "epr_weighted")
        vals.append(mean([y for y in ys if y is not None]) if any(y is not None for y in ys) else 0)
    ax.bar(x + (i-1)*w, vals, w, label=group)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, rotation=15)
ax.set_ylabel("Mean EPR (d1-d5)")
ax.set_title("Fig 2(b) — Impact: source type vs. downstream EPR")
ax.legend()
ax.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "fig2b_epr_by_source.pdf")
fig.savefig(FIG / "fig2b_epr_by_source.png", dpi=150)
plt.close(fig)

# Margin figure: clean margin per hop (one panel per group)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, group in zip(axes, GROUPS):
    for m in MODEL_ORDER:
        ys = _series(m, group, "clean_margin_weighted")
        ax.plot(HOPS, [y if y is not None else np.nan for y in ys], marker="o", label=m)
    ax.set_title(f"Source = {group}")
    ax.set_xlabel("hop distance")
    ax.grid(alpha=.3)
axes[0].set_ylabel("Clean Logit Margin (sample-weighted)")
axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("Decision boundary thickness: clean margin per hop")
fig.tight_layout()
fig.savefig(FIG / "fig_margin_by_hop.pdf")
fig.savefig(FIG / "fig_margin_by_hop.png", dpi=150)
plt.close(fig)

print("[write] figures ->", FIG)
print("[done]")
