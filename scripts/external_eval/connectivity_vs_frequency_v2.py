"""
Connectivity-vs-Frequency v2 — triplet axis (in_degree, pageview, wiki_freq).

Why v2
------
Yuji 2026-05-22 sync demoted pageview as the primary popularity signal
("跟我们的核心 contribution 是没有那么 align 的"). New primary signal:
per-QID surface-form frequency in en.wiki text — a "knowledge connectivity
proxy" rather than an "attention" proxy. v1 only had (in_degree, pageview);
v2 adds wiki_freq as a third axis on the same QID-resolved node set.

Direct purpose: pre-emptively answer Kathy reviewer's likely attack
"isn't graph-connectivity just pretraining-frequency in disguise?". We
want:
  - Spearman(in_degree, wiki_freq)  — expect STRONG (>0.5):
      "connectivity DOES reflect corpus density, as Yuji predicted"
  - Spearman(in_degree, pageview)   — expect WEAK (~0.276, already known):
      "connectivity is NOT human attention"
  - Spearman(pageview, wiki_freq)   — expect MEDIUM (~0.4):
      "even within popularity signals, attention != corpus density"

Output:
  data/external_eval/connectivity_vs_frequency_v2.json   (numbers)
  data/external_eval/connectivity_vs_frequency_v2.png    (3-panel scatter)
  data/external_eval/connectivity_vs_frequency_v2.md     (markdown summary)
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
QID_INDEX  = ROOT / "data/external_eval/graph_qid_index.json"
PV         = ROOT / "data/external_eval/graph_pageviews_2024_user.json"
WIKI_FREQ  = ROOT / "data/external_eval/wiki_entity_frequency_200000articles.json"

OUT_JSON = ROOT / "data/external_eval/connectivity_vs_frequency_v2.json"
OUT_PNG  = ROOT / "data/external_eval/connectivity_vs_frequency_v2.png"
OUT_MD   = ROOT / "data/external_eval/connectivity_vs_frequency_v2.md"


def bucketize(in_deg: int) -> str:
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def corr_block(x: np.ndarray, y: np.ndarray, label_x: str, label_y: str):
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    kt, kp = stats.kendalltau(x, y)
    print(f"  {label_x:14s} vs {label_y:14s} | "
          f"Pearson  r={pr:+.3f} p={pp:.1e} | "
          f"Spearman r={sr:+.3f} p={sp:.1e} | "
          f"Kendall  t={kt:+.3f} p={kp:.1e}")
    return {
        "pearson":  {"r": float(pr), "p": float(pp)},
        "spearman": {"r": float(sr), "p": float(sp)},
        "kendall":  {"r": float(kt), "p": float(kp)},
    }


def main():
    print("[1/4] Loading graph + sidecar + pageview + wiki_freq ...")
    with open(GRAPH_PATH, "rb") as f:
        gdata = pickle.load(f)
    G = gdata["graph"] if isinstance(gdata, dict) else gdata
    side = json.loads(QID_INDEX.read_text())
    qid_to_name = side["qid_to_name"]
    pv = json.loads(PV.read_text())
    wf = json.loads(WIKI_FREQ.read_text())
    print(f"      graph nodes              : {G.number_of_nodes():,}")
    print(f"      QID-resolved             : {len(qid_to_name):,}")
    print(f"      pageview entries         : {len(pv):,}")
    print(f"      wiki_freq entries        : {len(wf):,}")

    rows = []
    for qid, node in qid_to_name.items():
        if node not in G:
            continue
        in_deg = G.in_degree(node)
        info_pv = pv.get(qid, {})
        info_wf = wf.get(qid, {})
        rows.append({
            "qid": qid,
            "node": node,
            "in_degree": in_deg,
            "bucket": bucketize(in_deg),
            "title": info_pv.get("title"),
            "pageview_total": info_pv.get("pageviews_total", 0),
            "pageview_status": info_pv.get("fetch_status", "missing"),
            "wiki_freq": info_wf.get("freq", 0),
            "wiki_doc_freq": info_wf.get("doc_freq", 0),
        })
    df = pd.DataFrame(rows)
    print(f"      total resolved nodes     : {len(df):,}")
    print(f"      bucket dist              : {df['bucket'].value_counts().to_dict()}")

    # ---- Keep only nodes with all three signals available ----
    triplet = df[
        (df["pageview_status"] == "ok") &
        (df["in_degree"] > 0) &
        (df["wiki_freq"] > 0)
    ].copy()
    print(f"      ALL THREE signals present: {len(triplet):,}")
    if len(triplet) < 50:
        raise SystemExit("Not enough triplet data for correlation; aborting.")

    triplet["log_indeg"] = np.log10(triplet["in_degree"].astype(float))
    triplet["log_pv"]    = np.log10(triplet["pageview_total"].astype(float).clip(lower=1))
    triplet["log_wf"]    = np.log10(triplet["wiki_freq"].astype(float))

    # ---- (1) Three pairwise correlations -------------------------------
    print("\n[2/4] Pairwise correlations (log-log) ...")
    corr = {}
    corr["indeg_vs_wiki_freq"] = corr_block(
        triplet["log_indeg"].values, triplet["log_wf"].values,
        "log_indeg", "log_wiki_freq")
    corr["indeg_vs_pageview"] = corr_block(
        triplet["log_indeg"].values, triplet["log_pv"].values,
        "log_indeg", "log_pageview")
    corr["pageview_vs_wiki_freq"] = corr_block(
        triplet["log_pv"].values, triplet["log_wf"].values,
        "log_pageview", "log_wiki_freq")

    # ---- (2) Karpathy verdict: is graph-connectivity == corpus density? ----
    indeg_wf_sp = corr["indeg_vs_wiki_freq"]["spearman"]["r"]
    indeg_pv_sp = corr["indeg_vs_pageview"]["spearman"]["r"]
    pv_wf_sp    = corr["pageview_vs_wiki_freq"]["spearman"]["r"]

    print("\n      === Plan v3.1 §1.3 prediction check ===")
    print(f"      indeg vs wiki_freq Spearman = {indeg_wf_sp:+.3f}  "
          f"(predicted > 0.5: {'PASS' if indeg_wf_sp > 0.5 else 'FAIL'})")
    print(f"      indeg vs pageview  Spearman = {indeg_pv_sp:+.3f}  "
          f"(predicted weak)")
    print(f"      pageview vs wiki_f Spearman = {pv_wf_sp:+.3f}  "
          f"(predicted medium ~0.4)")

    # ---- (3) Three-panel scatter ----------------------------------------
    print("\n[3/4] Writing 3-panel scatter ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"hub": "#d62728", "mid": "#1f77b4", "tail": "#7f7f7f"}
    bucket_order = [b for b in ["hub", "mid", "tail"] if b in triplet["bucket"].values]

    pairs = [
        (axes[0], "in_degree",      "wiki_freq",      "log_indeg", "log_wf",
         "Graph in-degree vs Wiki frequency\n(MAIN — Yuji's connectivity proxy)",
         indeg_wf_sp),
        (axes[1], "in_degree",      "pageview_total", "log_indeg", "log_pv",
         "Graph in-degree vs Wikipedia pageviews\n(supplementary — attention proxy)",
         indeg_pv_sp),
        (axes[2], "pageview_total", "wiki_freq",      "log_pv",    "log_wf",
         "Pageviews vs Wiki frequency\n(both popularity signals, log-log)",
         pv_wf_sp),
    ]
    for ax, xcol, ycol, _, _, title, sp_r in pairs:
        for b in bucket_order:
            sub = triplet[triplet["bucket"] == b]
            ax.scatter(sub[xcol], sub[ycol], s=8, alpha=0.5,
                       label=f"{b} (n={len(sub)})", c=colors.get(b, "k"))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        ax.set_title(f"{title}\nSpearman ρ = {sp_r:+.3f}")
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Connectivity vs Frequency (v2): triplet axis on n={len(triplet):,} "
        f"QID-resolved graph nodes — generated 2026-05-22",
        fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"      wrote scatter -> {OUT_PNG}")

    # ---- (4) Disagreement examples between signals ----------------------
    print("\n[4/4] Computing disagreement examples + writing summary ...")
    # Rank each entity by each signal (high rank = more popular)
    triplet["rank_indeg"] = triplet["in_degree"].rank(ascending=False)
    triplet["rank_pv"]    = triplet["pageview_total"].rank(ascending=False)
    triplet["rank_wf"]    = triplet["wiki_freq"].rank(ascending=False)
    n = len(triplet)

    # Three-way disagreement: top-decile by wiki_freq, bottom-decile by in_degree
    famous_in_wiki_but_sparse_in_graph = triplet[
        (triplet["rank_wf"] <= 0.1 * n) & (triplet["rank_indeg"] >= 0.9 * n)
    ].nsmallest(15, "rank_wf")[["title", "qid", "in_degree", "pageview_total", "wiki_freq"]]

    # Hub in graph but rare in wiki (graph-specific dense)
    hub_in_graph_but_rare_in_wiki = triplet[
        (triplet["rank_indeg"] <= 0.1 * n) & (triplet["rank_wf"] >= 0.9 * n)
    ].nsmallest(15, "rank_indeg")[["title", "qid", "in_degree", "pageview_total", "wiki_freq"]]

    # ---- JSON summary ----
    summary = {
        "n_triplet": int(len(triplet)),
        "n_qid_resolved": int(len(df)),
        "bucket_distribution_triplet": triplet["bucket"].value_counts().to_dict(),
        "correlations_log_log": corr,
        "plan_v3_1_predictions": {
            "indeg_vs_wiki_freq_spearman": float(indeg_wf_sp),
            "indeg_vs_wiki_freq_predicted_threshold": 0.5,
            "indeg_vs_wiki_freq_pass": bool(indeg_wf_sp > 0.5),
            "indeg_vs_pageview_spearman": float(indeg_pv_sp),
            "pageview_vs_wiki_freq_spearman": float(pv_wf_sp),
        },
        "examples_famous_in_wiki_but_sparse_in_graph":
            famous_in_wiki_but_sparse_in_graph.to_dict(orient="records"),
        "examples_hub_in_graph_but_rare_in_wiki":
            hub_in_graph_but_rare_in_wiki.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"      wrote summary -> {OUT_JSON}")

    # ---- Markdown summary ----
    lines = [
        "# Connectivity vs Frequency v2 — triplet axis (2026-05-22)\n",
        f"- Nodes with **all three signals** available: **{len(triplet):,}** "
        f"(of {len(df):,} QID-resolved graph nodes)",
        f"- Bucket distribution (triplet subset): {triplet['bucket'].value_counts().to_dict()}\n",
        "## Correlations (log-log, Spearman is primary)\n",
        "| Pair | Pearson | Spearman | Kendall |",
        "|---|---|---|---|",
    ]
    for k, v in corr.items():
        lines.append(
            f"| {k} | {v['pearson']['r']:+.3f} | "
            f"{v['spearman']['r']:+.3f} | {v['kendall']['r']:+.3f} |"
        )
    lines += [
        "\n## Plan v3.1 §1.3 predictions\n",
        f"- `Spearman(in_degree, wiki_freq) = {indeg_wf_sp:+.3f}` — "
        f"**predicted > 0.5**: {'✅ PASS' if indeg_wf_sp > 0.5 else '❌ FAIL'} "
        f"→ sells *'connectivity reflects corpus density'*",
        f"- `Spearman(in_degree, pageview) = {indeg_pv_sp:+.3f}` — "
        f"predicted weak → sells *'connectivity ≠ human attention'*",
        f"- `Spearman(pageview, wiki_freq) = {pv_wf_sp:+.3f}` — "
        f"predicted medium → *'attention ≠ corpus density'*\n",
        "## Disagreement examples\n",
        "### Famous on Wikipedia (top decile wiki_freq) but sparse on our graph "
        "(bottom decile in_degree)",
        "(Pretraining frequency would over-rate these for our fragility study.)\n",
        famous_in_wiki_but_sparse_in_graph.to_markdown(index=False),
        "\n### Hub on our graph (top decile in_degree) but rare on Wikipedia "
        "(bottom decile wiki_freq)",
        "(Graph-specific densely-interlinked entities — pageview/wiki_freq alone would under-rate.)\n",
        hub_in_graph_but_rare_in_wiki.to_markdown(index=False),
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"      wrote markdown -> {OUT_MD}")

    print("\n=== Take-home ===")
    print(f"  Spearman(in_degree, wiki_freq) = {indeg_wf_sp:+.3f}")
    if indeg_wf_sp > 0.5:
        print("  -> Yuji prediction CONFIRMED: connectivity tracks corpus density.")
        print("     Pre-emptive answer to Kathy: graph-connectivity IS a corpus-density signal,")
        print("     and it is also stronger / more controllable than raw frequency for our claim.")
    else:
        print(f"  -> Yuji prediction NOT met (expected > 0.5). Reconsider narrative.")


if __name__ == "__main__":
    main()
