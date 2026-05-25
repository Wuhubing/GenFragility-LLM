"""Academic-style plots for the EMNLP submission.

Mirrors the visual language of
    scripts/external_eval/plot_popularity_proxy_academic.py
so the cross-model figures sit next to the popularity-proxy panel without
stylistic drift.

Sizing convention (ACL acl.sty: A4 + 2.5cm margins + 0.6cm columnsep):
    \\textwidth  ~= 6.30 in   (two-column floats, figure*)
    \\columnwidth ~= 3.03 in   (single-column floats, figure)
Every figure is rendered at its exact LaTeX-target width so the document
never scales it down. With a 10 pt base font this keeps on-page tick text
in the 9-10 pt range, which matches the body text size.

Layout decision:
    Two-column floats (figure* + width=\\textwidth):
        Fig 2 (PopularityParadox, 2-panel)
        Fig 5 (FlipVsSimilarity, twin-axis 4-model)
        Fig 7 (MitigationBlastRadius, 12-bar grouped)
    Single-column floats (figure + width=\\columnwidth):
        Fig 3 (InnocentBystander, 3x3 heatmap)
        Fig 4 (BlastRadius, 4-line plot)
        Fig 6 (SemanticDrift, 3x3 heatmap)

PDFs are written to analysis_4models/figures_v3/ and copied into the paper's
figures/ directory by main().

Palette (anchored on the four user-supplied colors, matching the proxy script):
    C_TAIL = "#AFC7D9"  (muted blue)  -- Hub Src / dominant bin
    C_HUB  = "#E8C9D2"  (soft pink)   -- Tail Src / hub overlay
    C_MID  = "#D8AEB7"  (mauve)       -- Mid / Random Src
    C_NEUT = "#C9CDD3"  (warm gray)   -- grid / box border
    C_TEXT = "#33363F"  (near-black)  -- all text
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------- paths ----------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "analysis_4models" / "figures_v3"
PAPER_DIR = ROOT / "_EMNLP_26__Knowledge_Updating_Ripples_into_Hubs (6)" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ACL geometry --------------------------------------------------
# A4 paper (21 cm) with 2.5 cm margins -> textwidth = 16 cm = 6.30 in.
# columnsep = 0.6 cm -> columnwidth = (16 - 0.6)/2 = 7.7 cm = 3.03 in.
W_TEXT = 6.30   # two-column float target width  (figure* + \textwidth)
W_COL  = 3.03   # single-column float target width (figure + \columnwidth)

# ---------- palette (mirrors plot_popularity_proxy_academic.py) -----------
C_TAIL = "#AFC7D9"   # muted blue   -- low-popularity bulk / Hub Src
C_HUB  = "#E8C9D2"   # soft pink    -- high-popularity head / Tail Src
C_MID  = "#D8AEB7"   # mauve        -- mid bucket / Random Src
C_NEUT = "#C9CDD3"   # warm gray    -- axes / grid / boxes
C_TEXT = "#33363F"   # near-black for text
C_CALL = "#8C4A57"   # mauve-darkened, used for Tail-source delta callouts


def set_academic_style() -> None:
    """Match plot_popularity_proxy_academic.py exactly.

    pdf.use14corefonts=True embeds the 14 standard PostScript fonts directly
    (real Times-Roman) regardless of which TTFs are installed locally; this
    is the same trick the proxy script uses so the two figures look identical.

    Base font bumped from 9 to 10 pt so that, when the figure renders 1:1
    at its target ACL width, axis labels and tick text stay at >= 8 pt on
    the printed page, matching the body text.
    """
    mpl.rcParams.update({
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Nimbus Roman No9 L",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
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


def _stats_box(ax, text: str,
               loc: tuple[float, float] = (0.035, 0.965),
               ha: str = "left", va: str = "top") -> None:
    """Reusable rounded white box, identical to the proxy script."""
    ax.text(
        loc[0], loc[1], text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.32",
                  facecolor="white", edgecolor=C_NEUT,
                  linewidth=0.6, alpha=0.95),
    )


# ---------------------------------------------------------------------------
# Fig 2 -- Popularity Paradox: (a) d=1 Hub Flip Rate per model
#                              (b) Mean d=1..d=5 EPR per model x src class
# ---------------------------------------------------------------------------
def plot_popularity_paradox() -> Path:
    # Data from analysis_4models/v2/fig2a_flip_v2.md and fig2b_epr_v2.md
    models = ["Qwen3.5-2B", "Qwen3.5-9B", "Gemma-4-E4B-it", "Gemma-4-31B-it"]
    short  = ["Qwen-2B", "Qwen-9B", "Gemma-E4B", "Gemma-31B"]

    # (a) d=1 Hub Flip Rate
    hub_flip_d1 = [0.545, 0.849, 0.016, 0.568]

    # (b) Mean d1-d5 EPR by source group
    epr_hub  = [0.376, 0.580, 0.070, 0.311]
    epr_tail = [0.512, 0.575, 0.146, 0.255]
    epr_rand = [0.536, 0.596, 0.116, 0.186]

    fig, axes = plt.subplots(1, 2, figsize=(W_TEXT, 3.0))

    # ---------- panel (a): Hub-neighbor d=1 Flip Rate ----------------------
    ax = axes[0]
    x = np.arange(len(models))
    bars = ax.bar(x, hub_flip_d1, width=0.55, color=C_HUB,
                  edgecolor="white", lw=0.6)
    for rect, v in zip(bars, hub_flip_d1):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.018,
                f"{v:.0%}", ha="center", va="bottom",
                fontsize=9, color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Hub neighbor Flip Rate at $d{=}1$")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a)  Vulnerability: Hub flip rate",
                 loc="left", pad=8, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    # ---------- panel (b): Mean d1-d5 EPR by source class ------------------
    ax = axes[1]
    w = 0.26
    bars_h = ax.bar(x - w, epr_hub,  width=w, color=C_TAIL,
                    edgecolor="white", lw=0.5, label="Src = Hub")
    bars_t = ax.bar(x,     epr_tail, width=w, color=C_HUB,
                    edgecolor="white", lw=0.5, label="Src = Tail")
    bars_r = ax.bar(x + w, epr_rand, width=w, color=C_MID,
                    edgecolor="white", lw=0.5, label="Src = Random")
    for group in (bars_h, bars_t, bars_r):
        for rect in group:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel(r"Mean EPR  ($d{=}1$--$d{=}5$)")
    ax.set_ylim(0, max(max(epr_hub), max(epr_tail), max(epr_rand)) * 1.32)
    ax.set_title("(b)  Impact: error propagation by source class",
                 loc="left", pad=8, fontweight="bold")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
              handlelength=1.2, columnspacing=1.6, frameon=False,
              fontsize=8.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    fig.subplots_adjust(wspace=0.30, top=0.84, bottom=0.16,
                        left=0.08, right=0.99)

    out = OUT_DIR / "Fig2_PopularityParadox.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 3 -- Innocent Bystander: cross-model Δmargin Src x Nbr heatmap
# ---------------------------------------------------------------------------
def plot_innocent_bystander() -> Path:
    # Data from analysis_4models/v2/fig3_innocent_bystander/fig3_crossmodel.md
    sources   = ["Hub", "Tail", "Random"]
    neighbors = ["Hub", "Mid", "Tail"]
    dmargin = np.array([
        [-1.56, -0.78, -0.69],   # Src = Hub
        [-1.74, -0.99, -0.38],   # Src = Tail  (Innocent-Bystander row)
        [-1.57, -1.10, -0.72],   # Src = Random
    ])
    counts = np.array([
        [29302, 11822, 740],
        [13527,  6010, 424],
        [23216,  8839, 483],
    ])

    # Sequential pastel ramp on negative drift (deeper = more collapse).
    # We invert sign so the colourbar visually reads "deeper = darker".
    abs_mag = -dmargin
    vmin, vmax = abs_mag.min() * 0.95, abs_mag.max() * 1.05
    cmap = LinearSegmentedColormap.from_list(
        "bystander_pastel", ["#FFFFFF", C_TAIL, C_MID], N=256,
    )

    # Tight bbox will expand the saved PDF to include the colorbar, so we
    # shrink the inner figsize to (~2.4 in wide) so that the saved file lands
    # near W_COL after colorbar capture.
    fig, ax = plt.subplots(figsize=(2.40, 2.55))
    im = ax.imshow(abs_mag, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(dmargin.shape[0]):
        for j in range(dmargin.shape[1]):
            ax.text(j, i - 0.14, f"{dmargin[i, j]:+.2f}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=10, fontweight="bold")
            ax.text(j, i + 0.24, f"n={counts[i, j]:,}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=7, alpha=0.80)

    # Highlight Innocent-Bystander cell: Tail-src -> Hub-nbr (largest |Δmargin|)
    i_max, j_max = np.unravel_index(np.argmax(abs_mag), abs_mag.shape)
    ax.add_patch(plt.Rectangle((j_max - 0.5, i_max - 0.5), 1, 1,
                               fill=False, edgecolor=C_CALL, lw=1.8))

    ax.set_xticks(range(len(neighbors)))
    ax.set_xticklabels([f"{n}-nbr" for n in neighbors])
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([f"Src={s}" for s in sources])
    ax.set_xlabel("Neighbor class")
    ax.set_ylabel("Update source class")
    ax.set_title("Innocent Bystander: $\\Delta\\mathrm{Margin}$",
                 loc="left", pad=8, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(r"$|\Delta\mathrm{Margin}|$ (deeper = larger collapse)",
                   fontsize=8, color=C_TEXT)
    cbar.ax.tick_params(labelsize=7.5, color=C_TEXT, labelcolor=C_TEXT)
    cbar.outline.set_edgecolor(C_NEUT)
    cbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.16)

    out = OUT_DIR / "Fig3_InnocentBystander.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 4 -- Extent of Error Propagation: per-model EPR vs hop (Hub source)
# ---------------------------------------------------------------------------
def plot_blast_radius() -> Path:
    # Data from analysis_4models/v2/fig1_epr_v2.md  (Group = hub rows)
    hops = np.arange(1, 6)
    epr_hub_src = {
        "Qwen3.5-2B":     [0.545, 0.284, 0.386, 0.319, 0.348],
        "Qwen3.5-9B":     [0.849, 0.594, 0.475, 0.468, 0.515],
        "Gemma-4-E4B-it": [0.016, 0.059, 0.114, 0.083, 0.077],
        "Gemma-4-31B-it": [0.568, 0.275, 0.228, 0.236, 0.249],
    }
    model_colors = {
        "Qwen3.5-2B":     C_TAIL,
        "Qwen3.5-9B":     C_HUB,
        "Gemma-4-E4B-it": C_MID,
        "Gemma-4-31B-it": "#9DB3C0",
    }
    means = {m: float(np.mean(v)) for m, v in epr_hub_src.items()}

    fig, ax = plt.subplots(figsize=(W_COL, 2.85))
    for model, vals in epr_hub_src.items():
        c = model_colors[model]
        ax.plot(hops, vals, marker="o", color=c, lw=1.6, ms=4.6,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{model} ($\\overline{{\\mathrm{{EPR}}}}={means[model]:.2f}$)")

    # Mark the long-range Qwen-9B persistence: arrow + annotation
    qwen9b_d5 = epr_hub_src["Qwen3.5-9B"][-1]
    ax.annotate(
        f"Qwen3.5-9B  EPR$={qwen9b_d5:.2f}$  at $d{{=}}5$",
        xy=(5, qwen9b_d5), xytext=(2.6, 0.86),
        ha="left", va="bottom", fontsize=7.5, color=C_TEXT,
        arrowprops=dict(arrowstyle="->", color=C_NEUT, lw=0.7,
                        shrinkA=2, shrinkB=4),
    )

    ax.set_xticks(hops)
    ax.set_xticklabels([f"$d{{=}}{h}$" for h in hops])
    ax.set_xlabel("Hop distance from injected fact")
    ax.set_ylabel(r"EPR  (Hub source)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Extent of Error Propagation",
                 loc="left", pad=8, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    leg = ax.legend(loc="lower left", title=None,
                    frameon=True, fontsize=7,
                    handlelength=1.3, handletextpad=0.4,
                    borderpad=0.3, labelspacing=0.3)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(C_NEUT)
    leg.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)

    out = OUT_DIR / "Fig4_BlastRadius.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 6 -- Semantic Drift (Src x Nbr) heatmap, judge-free
# ---------------------------------------------------------------------------
def plot_semantic_drift() -> Path:
    # Numbers from analysis_4models/v2/strict_d0/semantic_drift_summary.md
    sources = ["Hub", "Tail", "Random"]
    neighbors = ["Hub", "Mid", "Tail"]
    mat = np.array([
        [0.2812, 0.2839, 0.2869],   # Src=Hub
        [0.2551, 0.2939, 0.3412],   # Src=Tail
        [0.3429, 0.3062, 0.3447],   # Src=Random
    ])
    counts = np.array([
        [26462, 10634, 629],
        [8803,  4067, 257],
        [4277,  1885,  97],
    ])

    # Sequential pastel ramp: white -> muted blue -> mauve.
    cmap = LinearSegmentedColormap.from_list(
        "drift_pastel", ["#FFFFFF", C_TAIL, C_MID], N=256,
    )

    # Shrink to leave room for colorbar after tight-bbox capture.
    fig, ax = plt.subplots(figsize=(2.40, 2.50))
    vmin, vmax = mat.min() * 0.96, mat.max() * 1.02
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i - 0.14, f"{mat[i, j]:.3f}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=9.5, fontweight="bold")
            ax.text(j, i + 0.24, f"n={counts[i, j]:,}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=7, alpha=0.80)

    # Highlight the largest cell (Tail-src -> Tail-nbr): mauve outline.
    i_max, j_max = np.unravel_index(np.argmax(mat), mat.shape)
    ax.add_patch(plt.Rectangle((j_max - 0.5, i_max - 0.5), 1, 1,
                               fill=False, edgecolor=C_CALL, lw=1.6))

    ax.set_xticks(range(len(neighbors)))
    ax.set_xticklabels([f"{n}-nbr" for n in neighbors])
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([f"Src={s}" for s in sources])
    ax.set_xlabel("Neighbor class")
    ax.set_ylabel("Update source class")
    ax.set_title("Judge-free semantic drift", loc="left",
                 pad=8, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(
        r"Mean drift  $1 - \cos(y_{\mathrm{pre}}, y_{\mathrm{post}})$",
        fontsize=8, color=C_TEXT,
    )
    cbar.ax.tick_params(labelsize=7.5, color=C_TEXT, labelcolor=C_TEXT)
    cbar.outline.set_edgecolor(C_NEUT)
    cbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.16)

    out = OUT_DIR / "Fig6_SemanticDrift.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 7 -- Mitigation: absolute flipped-fact count by source x anchor mode
# ---------------------------------------------------------------------------
def plot_mitigation_blast() -> Path:
    # Numbers from analysis_4models/v2/fig4_mitigation/fig4_blast_radius.md
    modes = ["none", "pop. top5", "pop. top25", "pop. top75"]
    hub  = [5728, 5342, 5350, 5230]
    tail = [1924, 1465, 1719, 1314]
    rand = [4948, 4380, 4474, 4301]

    # Reductions on Tail relative to none: -23.9, -10.7, -31.7 %
    tail_delta = [None, -23.9, -10.7, -31.7]

    x = np.arange(len(modes))
    w = 0.26

    fig, ax = plt.subplots(figsize=(W_TEXT, 3.1))
    bars_h = ax.bar(x - w, hub,  width=w, color=C_TAIL, edgecolor="white",
                    lw=0.5, label="Src = Hub")
    bars_t = ax.bar(x,     tail, width=w, color=C_HUB,  edgecolor="white",
                    lw=0.5, label="Src = Tail")
    bars_r = ax.bar(x + w, rand, width=w, color=C_MID,  edgecolor="white",
                    lw=0.5, label="Src = Random")

    # value labels on top of each bar
    for group in (bars_h, bars_t, bars_r):
        for rect in group:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 80,
                    f"{int(v):,}",
                    ha="center", va="bottom", fontsize=7.5, color=C_TEXT)

    # Delta% callouts on Tail bars (the main story)
    for rect, d in zip(bars_t, tail_delta):
        if d is None:
            continue
        ax.annotate(
            f"{d:+.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, -14), textcoords="offset points",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=C_CALL,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Absolute flipped-fact count\n"
                  "(pooled, $d{=}1$--$d{=}5$, $17$ shared targets)")
    ax.set_xlabel("Anchor mode")
    ax.set_title("Popularity Anchoring reduces propagated errors",
                 loc="left", pad=8, fontweight="bold")
    ax.set_ylim(0, max(hub) * 1.30)

    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
              handlelength=1.4, columnspacing=1.8, frameon=False,
              fontsize=9)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.16)

    out = OUT_DIR / "Fig7_MitigationBlastRadius.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 5 -- Flip Rate vs Levenshtein similarity (per-model line)
# ---------------------------------------------------------------------------
def plot_flip_vs_sim() -> Path:
    # Per-model bins from analysis_4models/v2/lexical/flip_vs_sim.md (L(subject,head))
    bins = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    models = ["Qwen3.5-2B", "Qwen3.5-9B", "Gemma-4-E4B-it", "Gemma-4-31B-it"]
    flip = {
        "Qwen3.5-2B":     [0.364, 0.389, 0.367, 0.533, 0.562],
        "Qwen3.5-9B":     [0.480, 0.471, 0.562, 0.680, 0.870],
        "Gemma-4-E4B-it": [0.083, 0.096, 0.091, 0.068, 0.041],
        "Gemma-4-31B-it": [0.176, 0.195, 0.255, 0.264, 0.520],
    }
    # Pearson r(L_subj,head) per model from correlation_summary.md
    r_vals = {
        "Qwen3.5-2B":     +0.034,
        "Qwen3.5-9B":     +0.076,
        "Gemma-4-E4B-it": +0.000,
        "Gemma-4-31B-it": +0.090,
    }
    bin_n = [24512, 60742, 7747, 475, 887]

    # Use the four palette tones so the per-model colours stay on-brand
    # while still being distinguishable.
    model_colors = {
        "Qwen3.5-2B":     C_TAIL,
        "Qwen3.5-9B":     C_HUB,
        "Gemma-4-E4B-it": C_MID,
        "Gemma-4-31B-it": "#9DB3C0",   # slightly deeper blue-gray for contrast
    }

    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(W_TEXT, 3.1))

    # bottom-axis bin frequencies (twin log bars, soft neutral)
    ax_n = ax.twinx()
    ax_n.bar(x, bin_n, color=C_NEUT, edgecolor="none", zorder=0, width=0.78,
             alpha=0.55)
    ax_n.set_yscale("log")
    ax_n.set_ylim(1, max(bin_n) * 14)
    ax_n.set_ylabel("Source-neighbor pair count (log)",
                    fontsize=9, color=C_TEXT)
    ax_n.tick_params(axis="y", labelsize=8.5, color=C_TEXT, labelcolor=C_TEXT)
    ax_n.grid(False)
    for spine in ax_n.spines.values():
        spine.set_visible(False)

    # foreground: per-model lines drawn on top of the count bars
    for model in models:
        c = model_colors[model]
        ax.plot(x, flip[model], marker="o", color=c, lw=1.7, ms=5.4,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{model}  ($r={r_vals[model]:+.3f}$)")

    ax.set_zorder(ax_n.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=0)
    ax.set_xlabel(r"Levenshtein similarity bin  "
                  r"$L(\mathrm{subject}, \mathrm{head})$")
    ax.set_ylabel("Flip rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Surface similarity does not drive the ripple",
                 loc="left", pad=8, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    leg = ax.legend(loc="upper center", title="Model (Pearson $r$)",
                    title_fontsize=8.5, frameon=True, fontsize=8,
                    ncol=4, bbox_to_anchor=(0.5, 1.02),
                    handlelength=1.3, columnspacing=1.4)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(C_NEUT)
    leg.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=0.07, right=0.93, top=0.80, bottom=0.18)

    out = OUT_DIR / "Fig5_FlipVsSimilarity.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    set_academic_style()
    paths = [
        plot_popularity_paradox(),  # Fig 2
        plot_innocent_bystander(),  # Fig 3
        plot_blast_radius(),        # Fig 4
        plot_flip_vs_sim(),         # Fig 5
        plot_semantic_drift(),      # Fig 6
        plot_mitigation_blast(),    # Fig 7
    ]
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dst = PAPER_DIR / p.name
        # overwrite paper-dir copy so the LaTeX picks up the refreshed PDF
        dst.write_bytes(p.read_bytes())
        sz = p.stat().st_size
        print(f"[ok] {p.name:38s} ({sz/1024:6.1f} KB)  ->  {dst}")


if __name__ == "__main__":
    main()
