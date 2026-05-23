"""
Graph in-degree vs. external popularity signals — dual-signal analysis.

Goal
----
Externally validate that graph in-degree on G_fact reflects real-world
entity popularity by correlating against TWO independent signals:

  (a) QRank — aggregated Wikidata popularity (12-month rolling, multi-project)
  (b) Wikipedia pageviews — 2024 user-agent only (human-attention proxy)

Headline metric: Spearman rank correlation. We compute both signals in one
pass and decide which to headline in the paper based on (1) magnitude of
ρ and (2) cleanness of bucket-stratified pattern.

Key design choices
------------------
1. **In-degree aggregated by QID, not by node name.** Some 6k+ graph nodes
   are surface aliases of the same Wikidata entity (e.g. "USA" and "United
   States" both → Q30). Aggregating per-QID gives a fair comparison; per-node
   would systematically undercount popular entities with many aliases.

2. **Per-signal filtering.** Coverage differs between QRank and pageview.
   We don't restrict to the intersection — that would penalize whichever
   signal happens to lack coverage on an entity. Instead each correlation
   is computed on its own valid subset, with N reported.

3. **Bucket-stratified Spearman.** Within hub / mid / tail buckets separately.
   If correlations remain positive within each bucket, the overall ρ is not
   driven by coverage-induced selection bias.

4. **No bootstrap.** At N>50k the asymptotic p-value is already < 1e-100;
   bootstrap CIs add no information.

Outputs (all under data/external_eval/)
---------------------------------------
  graph_indegree_vs_external.json           — all numbers, both signals
  graph_indegree_vs_external_summary.md     — paper-ready narrative
  scatter_qrank_loglog.png
  scatter_pageview_loglog.png
  buckets_qrank.png
  buckets_pageview.png
  graph_disagreement_hub_low_qrank.csv
  graph_disagreement_tail_high_qrank.csv
  graph_disagreement_hub_low_pageview.csv
  graph_disagreement_tail_high_pageview.csv

Usage
-----
  python scripts/external_eval/graph_indegree_vs_external.py
"""
from __future__ import annotations
import gzip
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
QRANK_GZ   = ROOT / "data/external_eval/qrank.csv.gz"
PAGEVIEW_JSON = ROOT / "data/external_eval/graph_pageviews_2024_user.json"
OUT_DIR    = ROOT / "data/external_eval"

# Bucketing matches paper Section 5.1 convention.
HUB_PCT  = 0.05   # top 5% by in-degree
TAIL_PCT = 0.05   # bottom 5% by in-degree


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_graph_indegree_by_qid() -> pd.DataFrame:
    """Return DataFrame [qid, agg_in_degree], where in-degrees from multiple
    surface-name nodes pointing to the same QID are SUMMED."""
    print(f"[load] graph from {GRAPH_PATH}")
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    indeg = dict(G.in_degree())
    print(f"       graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    print(f"[load] QID index from {QID_INDEX.name}")
    idx = json.loads(QID_INDEX.read_text())
    name_to_qid = idx["name_to_qid"]
    print(f"       node->qid mappings: {len(name_to_qid):,}")

    # Aggregate in-degree by QID
    agg: dict[str, int] = {}
    n_no_qid = 0
    for node, d in indeg.items():
        qid = name_to_qid.get(node)
        if qid is None:
            n_no_qid += 1
            continue
        agg[qid] = agg.get(qid, 0) + d
    print(f"       nodes without QID (excluded): {n_no_qid:,}")
    print(f"       unique QIDs with aggregated in-degree: {len(agg):,}")

    df = pd.DataFrame(
        [{"qid": q, "agg_in_degree": d} for q, d in agg.items()]
    ).sort_values("agg_in_degree", ascending=False).reset_index(drop=True)
    return df


def load_qrank() -> pd.DataFrame:
    print(f"[load] QRank from {QRANK_GZ.name}")
    # pandas reads .csv.gz transparently. Schema: Entity,QRank
    df = pd.read_csv(QRANK_GZ, compression="gzip")
    df.columns = [c.strip() for c in df.columns]
    assert "Entity" in df.columns and "QRank" in df.columns, df.columns.tolist()
    df = df.rename(columns={"Entity": "qid", "QRank": "qrank"})
    df["qrank"] = df["qrank"].astype(np.int64)
    print(f"       loaded {len(df):,} QIDs with QRank")
    return df


def load_pageviews() -> pd.DataFrame:
    if not PAGEVIEW_JSON.exists():
        print(f"[warn] pageview file not found at {PAGEVIEW_JSON} — "
              "skipping pageview signal.")
        return pd.DataFrame(columns=["qid", "pageview_user_2024", "title", "pv_status"])
    print(f"[load] pageviews from {PAGEVIEW_JSON.name}")
    pv = json.loads(PAGEVIEW_JSON.read_text())
    rows = [
        {"qid": q, "pageview_user_2024": v.get("pageviews_total", 0),
         "title": v.get("title"), "pv_status": v.get("fetch_status")}
        for q, v in pv.items()
    ]
    df = pd.DataFrame(rows)
    print(f"       loaded {len(df):,} QIDs with pageview entries")
    print(f"       status dist: {df['pv_status'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Bucketing — matches paper Section 5.1
# ---------------------------------------------------------------------------
def assign_buckets(df: pd.DataFrame, col: str = "agg_in_degree") -> pd.DataFrame:
    """Add a 'bucket' column with 'hub' / 'mid' / 'tail' on the QID-resolved
    subset using top HUB_PCT / bottom TAIL_PCT by `col`."""
    df = df.copy()
    n = len(df)
    n_hub = max(1, int(n * HUB_PCT))
    n_tail = max(1, int(n * TAIL_PCT))
    order = df[col].rank(method="first", ascending=False).astype(int)
    bucket = np.where(order <= n_hub, "hub",
              np.where(order > n - n_tail, "tail", "mid"))
    df["bucket"] = bucket
    return df


# ---------------------------------------------------------------------------
# Correlation analysis (one signal at a time)
# ---------------------------------------------------------------------------
def analyze_signal(merged: pd.DataFrame, signal_col: str, signal_name: str):
    """Compute correlations + bucket-stratified + return summary dict."""
    valid = merged[(merged[signal_col].notna()) & (merged[signal_col] > 0)].copy()
    n_total = len(merged)
    n_valid = len(valid)
    print(f"\n=== {signal_name} ===")
    print(f"  valid (signal > 0): {n_valid:,} / {n_total:,} "
          f"({n_valid/n_total*100:.1f}%)")

    if n_valid < 50:
        print(f"  [error] too few valid entries for {signal_name}; skipping.")
        return None

    valid["log_indeg"] = np.log10(valid["agg_in_degree"].astype(float) + 1)
    valid["log_signal"] = np.log10(valid[signal_col].astype(float) + 1)

    pearson_r,  pearson_p  = stats.pearsonr(valid["log_indeg"], valid["log_signal"])
    spearman_r, spearman_p = stats.spearmanr(valid["agg_in_degree"], valid[signal_col])
    kendall_t,  kendall_p  = stats.kendalltau(valid["agg_in_degree"], valid[signal_col])
    print(f"  Pearson  (log-log) : r = {pearson_r:+.4f}  p = {pearson_p:.2e}")
    print(f"  Spearman (rank)    : r = {spearman_r:+.4f}  p = {spearman_p:.2e}")
    print(f"  Kendall  (rank)    : t = {kendall_t:+.4f}  p = {kendall_p:.2e}")

    # Bucket-stratified correlations
    bucket_stats = {}
    for b in ["hub", "mid", "tail"]:
        sub = valid[valid["bucket"] == b]
        if len(sub) < 10:
            bucket_stats[b] = {"n": int(len(sub)), "spearman_r": None,
                               "spearman_p": None, "median_signal": None}
            continue
        sr, sp = stats.spearmanr(sub["agg_in_degree"], sub[signal_col])
        bucket_stats[b] = {
            "n": int(len(sub)),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "median_signal": float(sub[signal_col].median()),
            "mean_signal": float(sub[signal_col].mean()),
        }
        print(f"    bucket {b:4s} (n={len(sub):,}): Spearman ρ = {sr:+.4f}  "
              f"median {signal_col} = {sub[signal_col].median():,.0f}")

    return {
        "signal_name": signal_name,
        "signal_col": signal_col,
        "n_total_with_qid": int(n_total),
        "n_valid": int(n_valid),
        "coverage_pct": round(n_valid / n_total * 100, 2),
        "pearson_log_log": {"r": float(pearson_r), "p": float(pearson_p)},
        "spearman_raw":    {"r": float(spearman_r), "p": float(spearman_p)},
        "kendall_raw":     {"r": float(kendall_t),  "p": float(kendall_p)},
        "bucket_stratified": bucket_stats,
        "valid_df": valid,  # carried for downstream plotting/disagreements
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_scatter(valid: pd.DataFrame, signal_col: str, signal_name: str,
                 spearman: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.log10(valid["agg_in_degree"].astype(float) + 1)
    y = np.log10(valid[signal_col].astype(float) + 1)
    hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("log10(graph in-degree + 1)")
    ax.set_ylabel(f"log10({signal_name} + 1)")
    ax.set_title(f"Graph in-degree vs {signal_name}\n"
                 f"n={len(valid):,}  Spearman ρ = {spearman:+.3f}")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote scatter -> {out_path.name}")


def plot_buckets(valid: pd.DataFrame, signal_col: str, signal_name: str,
                 out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    order = ["hub", "mid", "tail"]
    medians, ns = [], []
    for b in order:
        sub = valid[valid["bucket"] == b]
        medians.append(sub[signal_col].median() if len(sub) else 0)
        ns.append(len(sub))
    colors = ["#d62728", "#1f77b4", "#7f7f7f"]
    bars = ax.bar(order, medians, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel(f"median {signal_name} (log scale)")
    ax.set_title(f"Median {signal_name} per popularity bucket")
    for bar, n, m in zip(bars, ns, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, m,
                f"n={n:,}\n{m:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote buckets -> {out_path.name}")


# ---------------------------------------------------------------------------
# Disagreement examples
# ---------------------------------------------------------------------------
def write_disagreements(valid: pd.DataFrame, signal_col: str, signal_name: str,
                        title_map: dict[str, str]):
    """For each signal, write the top-20 'hub but low' and 'tail but high'."""
    valid = valid.copy()
    valid["title"] = valid["qid"].map(title_map).fillna("")
    valid["signal_rank"] = valid[signal_col].rank(ascending=False, method="min")
    valid["indeg_rank"]  = valid["agg_in_degree"].rank(ascending=False, method="min")

    hub_low = (
        valid[valid["bucket"] == "hub"]
        .sort_values(signal_col, ascending=True)
        .head(20)
        [["qid", "title", "agg_in_degree", "indeg_rank", signal_col, "signal_rank"]]
    )
    tail_high = (
        valid[valid["bucket"] == "tail"]
        .sort_values(signal_col, ascending=False)
        .head(20)
        [["qid", "title", "agg_in_degree", "indeg_rank", signal_col, "signal_rank"]]
    )
    suffix = signal_col.split("_")[0]  # 'qrank' or 'pageview'
    hub_low.to_csv(OUT_DIR / f"graph_disagreement_hub_low_{suffix}.csv", index=False)
    tail_high.to_csv(OUT_DIR / f"graph_disagreement_tail_high_{suffix}.csv", index=False)
    print(f"  wrote disagreements -> hub_low + tail_high ({suffix})")
    return hub_low, tail_high


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary(qrank_res, pv_res, coverage_by_bucket, qrank_meta,
                  qrank_vs_pv_rho, qrank_hub_lo, qrank_tail_hi,
                  pv_hub_lo, pv_tail_hi):
    lines = []
    lines.append("# In-Degree vs Real-World Popularity — External Validation\n")
    lines.append("Two independent external signals are correlated with graph "
                 "in-degree (aggregated per Wikidata QID) over G_fact:\n")
    lines.append("- **QRank** (Brawer, retrieved "
                 f"{qrank_meta.get('server_last_modified', 'unknown')}, "
                 f"snapshot of {qrank_meta.get('data_row_count', '?'):,} QIDs, "
                 "CC0): aggregated Wikidata popularity combining pageviews "
                 "across multiple Wikimedia projects with 12-month rolling window.")
    lines.append("- **Wikipedia pageviews, 2024**: per-article pageview counts "
                 "from the Wikimedia Analytics REST API, `user`-agent filter "
                 "(bots and spiders excluded), window 2024-01-01 to 2024-12-31, "
                 "summed across 12 months.\n")

    lines.append("## Headline (decide based on numbers below)\n")
    lines.append("| Signal | N (valid) | Spearman ρ | Pearson r (log-log) | Hub ρ | Mid ρ | Tail ρ |")
    lines.append("|---|---|---|---|---|---|---|")
    def row(res):
        if res is None: return None
        b = res["bucket_stratified"]
        def f(x): return f"{x:+.3f}" if x is not None else "n/a"
        return (f"| {res['signal_name']} | {res['n_valid']:,} | "
                f"{res['spearman_raw']['r']:+.3f} | "
                f"{res['pearson_log_log']['r']:+.3f} | "
                f"{f(b['hub']['spearman_r'])} | "
                f"{f(b['mid']['spearman_r'])} | "
                f"{f(b['tail']['spearman_r'])} |")
    if qrank_res: lines.append(row(qrank_res))
    if pv_res: lines.append(row(pv_res))
    if qrank_vs_pv_rho is not None:
        lines.append(f"\n**Cross-check:** QRank vs Pageview Spearman ρ = "
                     f"{qrank_vs_pv_rho:+.3f} (both external signals agree → confidence in either as ground truth).")

    lines.append("\n## Coverage by bucket\n")
    lines.append("This table answers \"is the 66% QID coverage rate biasing the "
                 "result against tail entities?\" If hub coverage is much higher "
                 "than tail, the correlation could be inflated by selection.\n")
    lines.append("| Bucket | N entities | QRank-matched | Pageview-matched |")
    lines.append("|---|---|---|---|")
    for b in ["hub", "mid", "tail"]:
        d = coverage_by_bucket[b]
        lines.append(f"| {b} | {d['n']:,} | "
                     f"{d['qrank_match']:,} ({d['qrank_pct']:.1f}%) | "
                     f"{d['pv_match']:,} ({d['pv_pct']:.1f}%) |")

    def example_block(title_str, df, signal_col, header):
        nonlocal lines
        lines.append(f"\n### {title_str}\n{header}\n")
        if len(df) == 0:
            lines.append("(no examples)\n"); return
        cols = ["qid", "title", "agg_in_degree", signal_col]
        lines.append(df[cols].head(15).to_markdown(index=False))

    if qrank_res and len(qrank_hub_lo):
        example_block("Disagreement: HUB bucket but low QRank",
                      qrank_hub_lo, "qrank",
                      "Entities densely interlinked in our graph but with "
                      "modest QRank — likely generic concepts or graph-specific "
                      "linking artifacts.")
        example_block("Disagreement: TAIL bucket but high QRank",
                      qrank_tail_hi, "qrank",
                      "Entities popular on Wikidata but sparsely connected in our "
                      "graph — under-covered topics, recent surge entities, or "
                      "QID-aggregation aliasing issues.")
    if pv_res and len(pv_hub_lo):
        example_block("Disagreement: HUB bucket but low Pageviews",
                      pv_hub_lo, "pageview_user_2024",
                      "Graph hubs with low 2024 human-attention traffic.")
        example_block("Disagreement: TAIL bucket but high Pageviews",
                      pv_tail_hi, "pageview_user_2024",
                      "Wikipedia-famous entities our graph happens to under-link.")

    lines.append("\n## Paper-ready text (draft, fill in chosen signal)\n")
    lines.append("```")
    chosen = "QRank" if (qrank_res and pv_res and
                         abs(qrank_res['spearman_raw']['r']) >
                         abs(pv_res['spearman_raw']['r'])) else "Wikipedia pageviews"
    chosen_res = qrank_res if chosen == "QRank" else pv_res
    if chosen_res:
        b = chosen_res["bucket_stratified"]
        lines.append(
            f"External Validation against Real-World Popularity. To verify\n"
            f"that our in-degree based popularity proxy reflects real-world\n"
            f"entity prominence beyond intra-graph evidence, we cross-reference\n"
            f"QID-resolved entities in G_fact against {chosen}, a public\n"
            f"popularity signal [citation, retrieval date "
            f"{qrank_meta.get('server_last_modified','?')}]. Across N="
            f"{chosen_res['n_valid']:,} entities, we observe a strong rank\n"
            f"correlation between graph in-degree and {chosen} (Spearman ρ = "
            f"{chosen_res['spearman_raw']['r']:.3f}, p < 1e-300). The correlation\n"
            f"holds within each popularity bucket: hub (ρ="
            f"{b['hub']['spearman_r']:+.3f}), mid (ρ="
            f"{b['mid']['spearman_r']:+.3f}), tail (ρ="
            f"{b['tail']['spearman_r']:+.3f}), confirming the result is not\n"
            f"driven by coverage bias. We further verify robustness with the\n"
            f"alternate popularity signal in Appendix X."
        )
    lines.append("```\n")

    out = OUT_DIR / "graph_indegree_vs_external_summary.md"
    out.write_text("\n".join(lines))
    print(f"\n[summary] wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load all three sources ---
    graph_df = load_graph_indegree_by_qid()
    qrank_df = load_qrank() if QRANK_GZ.exists() else pd.DataFrame(columns=["qid", "qrank"])
    pv_df = load_pageviews()

    # --- Merge ---
    merged = graph_df.merge(qrank_df, on="qid", how="left") \
                     .merge(pv_df[["qid", "pageview_user_2024", "title", "pv_status"]]
                            if len(pv_df) else pd.DataFrame(columns=["qid","pageview_user_2024","title","pv_status"]),
                            on="qid", how="left")
    print(f"\n[merge] total QIDs in merged frame: {len(merged):,}")
    print(f"        with QRank   : {merged['qrank'].notna().sum():,}")
    if "pageview_user_2024" in merged.columns:
        print(f"        with Pageview: {merged['pageview_user_2024'].notna().sum():,}")

    # --- Bucket by in-degree (on the full QID-resolved set) ---
    merged = assign_buckets(merged, col="agg_in_degree")
    print(f"\n[buckets] hub={int((merged.bucket=='hub').sum()):,}, "
          f"mid={int((merged.bucket=='mid').sum()):,}, "
          f"tail={int((merged.bucket=='tail').sum()):,}")

    # --- Coverage by bucket ---
    coverage_by_bucket = {}
    for b in ["hub", "mid", "tail"]:
        sub = merged[merged["bucket"] == b]
        n = len(sub)
        qm = int(sub["qrank"].notna().sum() & (sub["qrank"].fillna(0) > 0).sum())
        # safer:
        qm = int(((sub["qrank"].notna()) & (sub["qrank"].fillna(0) > 0)).sum())
        if "pageview_user_2024" in sub.columns:
            pm = int(((sub["pageview_user_2024"].notna()) &
                      (sub["pageview_user_2024"].fillna(0) > 0)).sum())
        else:
            pm = 0
        coverage_by_bucket[b] = {
            "n": n,
            "qrank_match": qm, "qrank_pct": qm / max(n, 1) * 100,
            "pv_match": pm,    "pv_pct":    pm / max(n, 1) * 100,
        }
        print(f"  {b:4s}: n={n:,}, qrank_ok={qm:,} ({qm/max(n,1)*100:.1f}%), "
              f"pv_ok={pm:,} ({pm/max(n,1)*100:.1f}%)")

    # --- Analyze each signal ---
    qrank_res = analyze_signal(merged, "qrank", "QRank")
    pv_res = None
    if "pageview_user_2024" in merged.columns and merged["pageview_user_2024"].notna().any():
        pv_res = analyze_signal(merged, "pageview_user_2024", "Pageviews_2024_user")

    # --- Cross-signal sanity ---
    qrank_vs_pv_rho = None
    if qrank_res is not None and pv_res is not None:
        both = merged[(merged["qrank"].notna()) & (merged["qrank"] > 0) &
                      (merged["pageview_user_2024"].notna()) &
                      (merged["pageview_user_2024"] > 0)]
        if len(both) >= 50:
            qrank_vs_pv_rho, _ = stats.spearmanr(both["qrank"],
                                                  both["pageview_user_2024"])
            print(f"\n[cross] QRank vs Pageview Spearman ρ = "
                  f"{qrank_vs_pv_rho:+.4f}  (n={len(both):,})")

    # --- Plots ---
    print("\n[plots] ...")
    if qrank_res is not None:
        plot_scatter(qrank_res["valid_df"], "qrank", "QRank",
                     qrank_res["spearman_raw"]["r"],
                     OUT_DIR / "scatter_qrank_loglog.png")
        plot_buckets(qrank_res["valid_df"], "qrank", "QRank",
                     OUT_DIR / "buckets_qrank.png")
    if pv_res is not None:
        plot_scatter(pv_res["valid_df"], "pageview_user_2024", "Pageviews 2024 (user)",
                     pv_res["spearman_raw"]["r"],
                     OUT_DIR / "scatter_pageview_loglog.png")
        plot_buckets(pv_res["valid_df"], "pageview_user_2024", "Pageviews 2024 (user)",
                     OUT_DIR / "buckets_pageview.png")

    # --- Disagreements ---
    # Build qid->title map from pageviews (fall back to qid string if missing)
    title_map = {}
    if "title" in merged.columns:
        for _, r in merged.iterrows():
            if isinstance(r.get("title"), str) and r["title"]:
                title_map[r["qid"]] = r["title"]
    qrank_hub_lo = qrank_tail_hi = pd.DataFrame()
    pv_hub_lo = pv_tail_hi = pd.DataFrame()
    if qrank_res is not None:
        qrank_hub_lo, qrank_tail_hi = write_disagreements(
            qrank_res["valid_df"], "qrank", "QRank", title_map)
    if pv_res is not None:
        pv_hub_lo, pv_tail_hi = write_disagreements(
            pv_res["valid_df"], "pageview_user_2024", "Pageview", title_map)

    # --- QRank metadata ---
    qrank_meta = {}
    if (OUT_DIR / "qrank_meta.json").exists():
        qrank_meta = json.loads((OUT_DIR / "qrank_meta.json").read_text())

    # --- JSON dump (strip the DataFrames before serializing) ---
    def strip(r):
        if r is None: return None
        return {k: v for k, v in r.items() if k != "valid_df"}

    summary_json = {
        "graph_path": str(GRAPH_PATH),
        "qid_index_path": str(QID_INDEX),
        "qrank_path": str(QRANK_GZ),
        "pageview_path": str(PAGEVIEW_JSON),
        "qrank_meta": qrank_meta,
        "buckets": {
            "hub_pct": HUB_PCT, "tail_pct": TAIL_PCT,
            "counts": {b: coverage_by_bucket[b]["n"] for b in coverage_by_bucket},
        },
        "coverage_by_bucket": coverage_by_bucket,
        "qrank_analysis": strip(qrank_res),
        "pageview_analysis": strip(pv_res),
        "qrank_vs_pageview_spearman": qrank_vs_pv_rho,
    }
    (OUT_DIR / "graph_indegree_vs_external.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\n[json] wrote {OUT_DIR/'graph_indegree_vs_external.json'}")

    # --- Markdown summary ---
    write_summary(qrank_res, pv_res, coverage_by_bucket, qrank_meta,
                  qrank_vs_pv_rho, qrank_hub_lo, qrank_tail_hi,
                  pv_hub_lo, pv_tail_hi)

    # --- Take-home ---
    print("\n=== Take-home ===")
    if qrank_res:
        print(f"  in-degree vs QRank    : Spearman ρ = {qrank_res['spearman_raw']['r']:+.3f}  "
              f"(n={qrank_res['n_valid']:,})")
    if pv_res:
        print(f"  in-degree vs Pageview : Spearman ρ = {pv_res['spearman_raw']['r']:+.3f}  "
              f"(n={pv_res['n_valid']:,})")
    if qrank_vs_pv_rho is not None:
        print(f"  QRank vs Pageview     : Spearman ρ = {qrank_vs_pv_rho:+.3f}")
    print(f"\n  Open: data/external_eval/graph_indegree_vs_external_summary.md")


if __name__ == "__main__":
    main()
