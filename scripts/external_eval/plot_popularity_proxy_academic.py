"""
Academic-style figure for the popularity-proxy validation section.

Produces a 3-panel log-log scatter showing pairwise relationships among the
three popularity signals on the QID-resolved subset of G_fact:

    Panel A:  graph in-degree  vs  wiki entity frequency  (MAIN)
    Panel B:  graph in-degree  vs  Wikipedia 2024 pageviews
    Panel C:  Wikipedia pageviews vs wiki entity frequency

Design goals (per user spec):
  - Conference-paper aesthetic (serif, restrained, no clutter).
  - Palette anchored on the four user-supplied colors
        #AFC7D9  (muted blue)     - tail bucket (bulk)
        #E8C9D2  (soft pink)      - hub bucket (head)
        #D8AEB7  (mauve)          - mid bucket
        #C9CDD3  (warm gray)      - axes / grid / annotations
  - Each panel shows Spearman rho + n in a clean stats box.
  - Hub points drawn last and larger to give a visual anchor for the
    "head agrees" sanity claim used in the dataset_contribution section.

Inputs (already on disk; nothing to re-fetch):
    results/checkpoints/final.pkl
    data/external_eval/graph_qid_index.json
    data/external_eval/graph_pageviews_2024_user.json
    data/external_eval/wiki_entity_frequency_200000articles.json

Outputs:
    data/external_eval/popularity_proxy_academic.pdf
    data/external_eval/popularity_proxy_academic.png
    (copy of PDF also written to the paper figures/ directory)
"""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# ---------- paths ----------------------------------------------------------
ROOT       = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
QID_INDEX  = ROOT / "data/external_eval/graph_qid_index.json"
PV_PATH    = ROOT / "data/external_eval/graph_pageviews_2024_user.json"
WF_PATH    = ROOT / "data/external_eval/wiki_entity_frequency_200000articles.json"

OUT_DIR    = ROOT / "data/external_eval"
OUT_PDF    = OUT_DIR / "popularity_proxy_academic.pdf"
OUT_PNG    = OUT_DIR / "popularity_proxy_academic.png"
PAPER_FIG  = ROOT / "_EMNLP_26__Knowledge_Updating_Ripples_into_Hubs (6)/figures/Fig1_PopularityProxy.pdf"

# ---------- palette --------------------------------------------------------
C_TAIL  = "#AFC7D9"   # muted blue   -- low-popularity bulk
C_HUB   = "#E8C9D2"   # soft pink    -- high-popularity head
C_MID   = "#D8AEB7"   # mauve        -- mid-popularity
C_NEUT  = "#C9CDD3"   # warm gray    -- axes / grid / boxes
C_TEXT  = "#33363F"   # near-black for text


def bucketize(in_deg: int) -> str:
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def load_triplet() -> pd.DataFrame:
    with open(GRAPH_PATH, "rb") as f:
        gdata = pickle.load(f)
    G = gdata["graph"] if isinstance(gdata, dict) else gdata
    side = json.loads(QID_INDEX.read_text())
    qid_to_name = side["qid_to_name"]
    pv = json.loads(PV_PATH.read_text())
    wf = json.loads(WF_PATH.read_text())

    rows = []
    for qid, node in qid_to_name.items():
        if node not in G:
            continue
        in_deg = G.in_degree(node)
        info_pv = pv.get(qid, {})
        info_wf = wf.get(qid, {})
        rows.append({
            "qid": qid,
            "title": info_pv.get("title") or node,
            "in_degree": in_deg,
            "bucket": bucketize(in_deg),
            "pageview_total": info_pv.get("pageviews_total", 0),
            "pageview_status": info_pv.get("fetch_status", "missing"),
            "wiki_freq": info_wf.get("freq", 0),
        })
    df = pd.DataFrame(rows)

    triplet = df[
        (df["pageview_status"] == "ok") &
        (df["in_degree"] > 0) &
        (df["wiki_freq"] > 0) &
        (df["pageview_total"] > 0)
    ].copy()
    return triplet


def set_academic_style():
    # Use PDF core fonts so the saved PDF embeds true Times-Roman regardless
    # of which TTFs are installed on this machine (DejaVu Serif is the only
    # serif TTF present locally). Type-42 keeps text selectable / searchable.
    plt.rcParams.update({
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Nimbus Roman No9 L",
                        "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Font sizes are tuned for a TWO-COLUMN-WIDE figure (\textwidth in
        # EMNLP/ACL is ~6.3in). The figure is rendered at 1:1 scale, so the
        # font sizes in points here are exactly what appears in the PDF.
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.edgecolor": C_TEXT,
        "axes.labelcolor": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "text.color": C_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def draw_panel(ax, df: pd.DataFrame, xcol: str, ycol: str,
               xlabel: str, ylabel: str, title: str):
    # Slightly larger scatter points to read well at \textwidth (~6.3in).
    layer_cfg = [
        ("tail", C_TAIL, 7.5,  0.22),
        ("mid",  C_MID,  13.0, 0.55),
        ("hub",  C_HUB,  36.0, 0.95),
    ]
    handles = []
    for bucket, color, size, alpha in layer_cfg:
        sub = df[df["bucket"] == bucket]
        if sub.empty:
            continue
        h = ax.scatter(
            sub[xcol], sub[ycol],
            s=size, alpha=alpha,
            facecolor=color,
            edgecolor="white" if bucket == "hub" else "none",
            linewidth=0.4 if bucket == "hub" else 0.0,
            label=f"{bucket.capitalize()} (n={len(sub):,})",
            rasterized=True,
        )
        handles.append(h)

    # Spearman on log-log (monotonic so ranks are identical; report directly).
    sr, _ = stats.spearmanr(df[xcol].values, df[ycol].values)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, loc="left", fontweight="bold")

    # Light dotted grid only on major ticks, suppress minor tick labels.
    ax.grid(True, which="major", linestyle=":", linewidth=0.5,
            color=C_NEUT, alpha=0.8, zorder=0)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))

    # Stats box (Spearman rho + n) in the top-left, sitting in empty area.
    stats_text = rf"Spearman $\rho={sr:+.3f}$" + "\n" + rf"$n={len(df):,}$"
    ax.text(
        0.035, 0.965, stats_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.34",
                  facecolor="white", edgecolor=C_NEUT,
                  linewidth=0.7, alpha=0.95),
    )
    return handles


def main():
    print("[1/3] Loading triplet ...")
    df = load_triplet()
    print(f"      triplet n = {len(df):,}")
    print(f"      bucket dist = {df['bucket'].value_counts().to_dict()}")

    set_academic_style()

    print("[2/3] Rendering 3-panel figure ...")
    # EMNLP/ACL \textwidth is ~6.3in (double-column page). We render at
    # native size 6.3 x 3.0 in so all in-figure text appears at exactly the
    # rcParams font sizes when the PDF is included with width=\textwidth.
    # Extra bottom space is reserved for the shared legend strip.
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.05))

    h_a = draw_panel(
        axes[0], df,
        xcol="in_degree", ycol="wiki_freq",
        xlabel="Graph in-degree",
        ylabel="Wiki surface-form frequency",
        title="(a)  in-degree  vs  wiki frequency",
    )
    draw_panel(
        axes[1], df,
        xcol="in_degree", ycol="pageview_total",
        xlabel="Graph in-degree",
        ylabel="Wikipedia pageviews (2024)",
        title="(b)  in-degree  vs  pageviews",
    )
    draw_panel(
        axes[2], df,
        xcol="pageview_total", ycol="wiki_freq",
        xlabel="Wikipedia pageviews (2024)",
        ylabel="Wiki surface-form frequency",
        title="(c)  pageviews  vs  wiki frequency",
    )

    # One shared legend below all three panels (no overlap with data).
    # Build proxy handles with fixed visible size so the legend dots aren't
    # constrained by the rasterized scatter alpha.
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=C_TAIL, markeredgecolor="none",
               markersize=6.5, label="Tail  (n=34,754)"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=C_MID,  markeredgecolor="none",
               markersize=7.0, label="Mid  (n=1,083)"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=C_HUB,  markeredgecolor="white",
               markeredgewidth=0.5, markersize=8.5, label="Hub  (n=31)"),
    ]
    leg = fig.legend(
        handles=proxy, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False, handletextpad=0.5, columnspacing=2.2,
        fontsize=10,
    )

    fig.subplots_adjust(wspace=0.42, left=0.075, right=0.99,
                        top=0.91, bottom=0.30)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    plt.close(fig)
    print(f"      wrote {OUT_PDF}")
    print(f"      wrote {OUT_PNG}")

    print("[3/3] Copying PDF into the paper figures/ directory ...")
    PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PDF, PAPER_FIG)
    print(f"      copied -> {PAPER_FIG}")


if __name__ == "__main__":
    main()
