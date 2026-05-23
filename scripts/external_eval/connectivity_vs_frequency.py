"""
Connectivity-vs-Frequency analysis.

Goal: pre-emptively answer Kathy's reviewer comment
  "isn't graph-connectivity just pretraining-frequency in disguise?"

Inputs (no public-benchmark dependency):
  - results/checkpoints/final.pkl                            (in_degree per node)
  - data/external_eval/graph_qid_index.json                  (node <-> QID)
  - data/external_eval/graph_pageviews_2024_user.json        (QID -> 12mo pageviews)

Deliverables:
  1. Pearson + Spearman correlation between log(in_degree) and log(pageview)
     -- if r < 0.7, the two signals carry meaningfully different information
  2. 3x3 cross-tab: in_degree bucket (hub/mid/tail) x pageview tercile (hi/mid/lo)
     -- if off-diagonal mass > 30%, the two ranking systems disagree often
  3. "Disagreement examples":
       a. High pageview, Tail bucket (entities famous in en.wiki but sparse in our graph)
       b. Hub bucket, low pageview (entities densely interlinked in our graph but unpopular on en.wiki)
  4. Scatter plot (log-log) of in_degree vs pageview, colored by bucket

Output:
  data/external_eval/connectivity_vs_frequency.json   (numbers)
  data/external_eval/connectivity_vs_frequency.png    (scatter)
  data/external_eval/connectivity_vs_frequency_table.md (markdown summary)

Run:
  conda run -n genfragility python scripts/external_eval/connectivity_vs_frequency.py
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
OUT_JSON   = ROOT / "data/external_eval/connectivity_vs_frequency.json"
OUT_PNG    = ROOT / "data/external_eval/connectivity_vs_frequency.png"
OUT_MD     = ROOT / "data/external_eval/connectivity_vs_frequency_table.md"


def bucketize(in_deg: int) -> str:
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def main():
    print("[1/4] Loading graph + sidecar + pageview cache ...")
    with open(GRAPH_PATH, "rb") as f:
        gdata = pickle.load(f)
    G = gdata["graph"] if isinstance(gdata, dict) else gdata
    side = json.loads(QID_INDEX.read_text())
    qid_to_name = side["qid_to_name"]
    pv = json.loads(PV.read_text())

    rows = []
    for qid, node in qid_to_name.items():
        if node not in G:
            continue
        in_deg = G.in_degree(node)
        info = pv.get(qid, {})
        rows.append({
            "qid": qid,
            "node": node,
            "in_degree": in_deg,
            "bucket": bucketize(in_deg),
            "title": info.get("title"),
            "pageview_total": info.get("pageviews_total", 0),
            "fetch_status": info.get("fetch_status", "missing"),
        })
    df = pd.DataFrame(rows)
    print(f"      total QID-resolved nodes  : {len(df):,}")
    print(f"      bucket dist               : {df['bucket'].value_counts().to_dict()}")
    print(f"      pageview fetch status     : {df['fetch_status'].value_counts().to_dict()}")

    # ---- Keep only entities with both signals available ----
    ok = df[(df["fetch_status"] == "ok") & (df["in_degree"] > 0)].copy()
    print(f"      both signals present      : {len(ok):,}")
    if len(ok) < 50:
        raise SystemExit("Not enough data for correlation; aborting.")

    ok["log_indeg"] = np.log10(ok["in_degree"].astype(float))
    ok["log_pv"]    = np.log10(ok["pageview_total"].astype(float).clip(lower=1))

    # ---- (1) Correlations -------------------------------------------------
    print("\n[2/4] Computing correlations ...")
    pearson_r,  pearson_p  = stats.pearsonr (ok["log_indeg"], ok["log_pv"])
    spearman_r, spearman_p = stats.spearmanr(ok["log_indeg"], ok["log_pv"])
    kendall_t,  kendall_p  = stats.kendalltau(ok["log_indeg"], ok["log_pv"])

    print(f"      Pearson  (log-log): r = {pearson_r:+.3f}  p = {pearson_p:.2e}")
    print(f"      Spearman (rank)   : r = {spearman_r:+.3f}  p = {spearman_p:.2e}")
    print(f"      Kendall  (rank)   : t = {kendall_t:+.3f}  p = {kendall_p:.2e}")

    # ---- (2) Bucket x pageview-tercile cross-tab --------------------------
    print("\n[3/4] Cross-tabulation: graph bucket vs pageview tercile ...")
    ok["pv_tercile"] = pd.qcut(ok["pageview_total"], q=3,
                               labels=["pv_lo", "pv_mid", "pv_hi"])
    cross = pd.crosstab(ok["bucket"], ok["pv_tercile"])
    bucket_order = [b for b in ["hub", "mid", "tail"] if b in cross.index]
    cross = cross.reindex(bucket_order)
    print(cross)
    cross_pct = (cross / cross.values.sum() * 100).round(1)

    diag_pairs = {"hub": "pv_hi", "mid": "pv_mid", "tail": "pv_lo"}
    aligned = sum(cross.loc[b, p] for b, p in diag_pairs.items()
                  if b in cross.index and p in cross.columns)
    aligned_pct = aligned / cross.values.sum() * 100
    print(f"\n      diagonal-aligned mass     : {aligned_pct:.1f}%")
    print(f"      off-diagonal mass         : {100 - aligned_pct:.1f}%")

    # ---- (3) Disagreement examples ---------------------------------------
    tail_hi = ok[(ok["bucket"] == "tail") &
                 (ok["pv_tercile"] == "pv_hi")].sort_values(
                     "pageview_total", ascending=False).head(15)
    hub_lo  = ok[(ok["bucket"] == "hub") &
                 (ok["pv_tercile"] == "pv_lo")].sort_values(
                     "in_degree", ascending=False).head(15)
    print(f"\n      'famous-but-sparse' (tail+pv_hi): {len(tail_hi)} examples")
    print(f"      'hub-but-quiet'     (hub +pv_lo): {len(hub_lo)} examples")

    # ---- (4) Scatter plot -------------------------------------------------
    print("\n[4/4] Writing outputs (json/md/png) ...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"hub": "#d62728", "mid": "#1f77b4", "tail": "#7f7f7f"}
    for b in bucket_order:
        sub = ok[ok["bucket"] == b]
        ax.scatter(sub["in_degree"], sub["pageview_total"],
                   s=11, alpha=0.55, label=f"{b} (n={len(sub)})",
                   c=colors.get(b, "k"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("In-degree on our 100k graph (log)")
    ax.set_ylabel("Wikipedia pageviews, 12 mo (log)")
    ax.set_title("Graph connectivity vs. Wikipedia frequency\n"
                 f"(QID-resolved graph nodes, n={len(ok):,}; Pearson r={pearson_r:.2f})")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"      wrote scatter  -> {OUT_PNG}")

    # ---- Markdown table summary ------------------------------------------
    lines = []
    lines.append("# Connectivity vs Frequency — full graph (QID-resolved nodes)\n")
    lines.append(f"- Nodes with both signals: **{len(ok):,}** "
                 f"(of {len(df):,} QID-resolved; "
                 f"{df['fetch_status'].value_counts().get('ok', 0):,} pageview-ok)\n")
    lines.append("## Correlations (log-log)\n")
    lines.append("| Statistic | r/t | p |\n|---|---|---|")
    lines.append(f"| Pearson  | {pearson_r:+.3f} | {pearson_p:.2e} |")
    lines.append(f"| Spearman | {spearman_r:+.3f} | {spearman_p:.2e} |")
    lines.append(f"| Kendall  | {kendall_t:+.3f} | {kendall_p:.2e} |")
    lines.append("\n## Bucket × pageview-tercile (count)\n")
    lines.append(cross.to_markdown())
    lines.append("\n## Bucket × pageview-tercile (% of all)\n")
    lines.append(cross_pct.to_markdown())
    lines.append(f"\n**Diagonal aligned mass:** {aligned_pct:.1f}%  "
                 f"(off-diagonal {100-aligned_pct:.1f}%)\n")

    def tbl(sub, cols):
        return sub[cols].to_markdown(index=False)

    lines.append("\n## Disagreement examples\n")
    lines.append("### `tail` bucket but high pageview\n(entities famous on Wikipedia "
                 "but sparsely interlinked in our 100k graph; "
                 "suggests Wikipedia-frequency would over-rate them)\n")
    lines.append(tbl(tail_hi, ["title", "in_degree", "pageview_total", "qid"]))
    lines.append("\n### `hub` bucket but low pageview\n(entities densely interlinked "
                 "in our graph but low Wikipedia traffic; "
                 "suggests pageview alone would under-rate them)\n")
    lines.append(tbl(hub_lo, ["title", "in_degree", "pageview_total", "qid"]))

    OUT_MD.write_text("\n".join(lines))
    print(f"      wrote summary  -> {OUT_MD}")

    # ---- JSON for programmatic consumption -------------------------------
    summary = {
        "n_nodes_with_both_signals": int(len(ok)),
        "n_qid_resolved_nodes":      int(len(df)),
        "correlations_log_log": {
            "pearson":  {"r": float(pearson_r),  "p": float(pearson_p)},
            "spearman": {"r": float(spearman_r), "p": float(spearman_p)},
            "kendall":  {"r": float(kendall_t),  "p": float(kendall_p)},
        },
        "cross_tab_count":   {str(b): cross.loc[b].to_dict() for b in cross.index},
        "cross_tab_percent": {str(b): cross_pct.loc[b].to_dict() for b in cross_pct.index},
        "diagonal_aligned_percent": float(round(aligned_pct, 2)),
        "off_diagonal_percent":     float(round(100 - aligned_pct, 2)),
        "examples_tail_high_pageview": tail_hi[
            ["title", "in_degree", "pageview_total", "qid"]
        ].to_dict(orient="records"),
        "examples_hub_low_pageview": hub_lo[
            ["title", "in_degree", "pageview_total", "qid"]
        ].to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"      wrote summary  -> {OUT_JSON}")

    print("\n=== Take-home ===")
    print(f"  Pearson(log_indeg, log_pageview) = {pearson_r:+.3f}")
    print(f"  Diagonal aligned mass            = {aligned_pct:.1f}%")
    if pearson_r < 0.7 and aligned_pct < 70:
        print("  -> connectivity and frequency are NOT redundant; "
              "rebuts Kathy's pre-emptively.")
    else:
        print("  -> signals overlap heavily.")


if __name__ == "__main__":
    main()
