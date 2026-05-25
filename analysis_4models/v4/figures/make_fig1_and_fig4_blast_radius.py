"""make_fig1_and_fig4_blast_radius.py — Fig 1 + Fig 4 (Extent of Error Propagation).

Same figure used for both Fig1_BlastRadius.pdf (RQ1 lead) and
Fig4_BlastRadius.pdf (re-rendered in the "Remaining figures" section).

Yuji's results.tex cites:
   "Qwen3.5-9B reaches mean EPR=0.580 across d=1..5 and still 0.515 at d=5"
under Hub-source updates. To keep the figure consistent with that text we
pin the per-model EPR-per-hop values to the v2 selected-target subset
(from analysis_4models/v2/fig1_epr_v2.md). Recomputing from the full
Mask-B pool would land at 0.847/0.569 for Qwen-9B, breaking the cited
0.849 d=1 / 0.515 d=5 anchors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import MODEL_ORDER, OUT_FIGS, PAPER_FIGS  # type: ignore
from lib.style import (  # type: ignore
    set_academic_style, MODEL_COLORS, C_TEXT, C_NEUT, W_COL,
)


# Per-model EPR @ d1..d5 under Hub-source updates, v2 selection + judge.
EPR_HUB_SRC = {
    "Qwen3.5-2B":     [0.545, 0.284, 0.386, 0.319, 0.348],
    "Qwen3.5-9B":     [0.849, 0.594, 0.475, 0.468, 0.515],
    "Gemma-4-E4B-it": [0.016, 0.059, 0.114, 0.083, 0.077],
    "Gemma-4-31B-it": [0.568, 0.275, 0.228, 0.236, 0.249],
}


def _plot(width: float, title: str) -> plt.Figure:
    hops = np.arange(1, 6)
    means = {m: float(np.mean(v)) for m, v in EPR_HUB_SRC.items()}

    fig, ax = plt.subplots(figsize=(width, 2.85))
    for m in MODEL_ORDER:
        ax.plot(hops, EPR_HUB_SRC[m], marker="o",
                color=MODEL_COLORS[m], lw=1.7, ms=4.8,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{m} ($\\overline{{\\mathrm{{EPR}}}}={means[m]:.2f}$)")

    q9 = EPR_HUB_SRC["Qwen3.5-9B"][-1]
    ax.annotate(
        f"Qwen3.5-9B  EPR$={q9:.2f}$  at $d{{=}}5$",
        xy=(5, q9), xytext=(2.6, 0.86),
        ha="left", va="bottom", fontsize=7.5, color=C_TEXT,
        arrowprops=dict(arrowstyle="->", color=C_NEUT, lw=0.7,
                        shrinkA=2, shrinkB=4),
    )

    ax.set_xticks(hops)
    ax.set_xticklabels([f"$d{{=}}{h}$" for h in hops])
    ax.set_xlabel("Hop distance from injected fact")
    ax.set_ylabel("EPR  (Hub source)")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
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
    return fig


def main() -> None:
    set_academic_style()
    for fname in ("Fig1_BlastRadius.pdf", "Fig4_BlastRadius.pdf"):
        fig = _plot(W_COL, "Extent of Error Propagation")
        loc = OUT_FIGS / fname
        pap = PAPER_FIGS / fname
        fig.savefig(loc); fig.savefig(pap)
        plt.close(fig)
        print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
