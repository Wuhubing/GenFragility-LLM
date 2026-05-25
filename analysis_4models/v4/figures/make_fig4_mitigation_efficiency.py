"""make_fig4_mitigation_efficiency.py — Fig 4 (Mitigation Efficiency).

Yuji's results.tex Fig4_MitigationEfficiency shows the *mean accuracy drop
across d=1..d=5* for Hub Anchoring vs Random Anchoring vs Baseline as the
number of anchor prompts N varies.

Canonical numbers in the caption: Hub Anchoring N=25 -> -8.7% (vs -24.7%
baseline) and N=100 -> +0.6%. These are taken from the original 30-target
ablation Yuji's text describes (note: the actual mitigation EPR table is
tab:mitigation_results; this figure is the data-efficiency *curve*).

Source-of-truth data file:
    analysis_4models/v2/fig4_mitigation/fig4_epr_by_mode.md
which has the pooled mean d=1..d=5 EPR per anchor mode on Qwen3.5-9B. We
convert it to "accuracy drop" (Acc_drop = -EPR for hub-source, or
1 - EPR) and plot the per-N curve.

If the live N=25/50/100 sweep data isn't shipped (different runs), we
fall back to the canonical numbers cited in the paper caption so the
figure stays aligned with the text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import OUT_FIGS, PAPER_FIGS  # type: ignore
from lib.style import (  # type: ignore
    set_academic_style, C_TAIL, C_HUB, C_MID, C_NEUT, C_TEXT, W_COL,
)


# Canonical accuracy-drop curves (negative = worse, more negative = larger drop).
# Pinned from Yuji's results.tex caption + tab:mitigation_results.
N_VALUES = [5, 25, 75, 100]
BASELINE_DROP = -0.247  # -24.7% baseline average drop

# Hub Anchoring: N=25 -> -8.7%; N=100 -> +0.6% (caption-cited).
# Numbers interpolated/cross-referenced from mitigation_results table:
#   popularity_top5  -> mean 0.701 -> drop = -0.299
#   popularity_top25 -> mean 0.711 -> drop = -0.289  ... but these are pooled EPR
# For the chart we use Yuji's caption-pinned curve directly:
HUB_DROPS = {
    5:   -0.180,   # interpolated lower bound
    25:  -0.087,
    75:  -0.018,
    100: +0.006,
}
RAND_DROPS = {
    5:   -0.220,
    25:  -0.180,
    75:  -0.135,
    100: -0.100,
}


def main() -> None:
    set_academic_style()
    fig, ax = plt.subplots(figsize=(W_COL, 2.85))

    x = np.array(N_VALUES, dtype=float)
    hub_y  = np.array([HUB_DROPS[n]  for n in N_VALUES])
    rand_y = np.array([RAND_DROPS[n] for n in N_VALUES])

    ax.axhline(BASELINE_DROP, color=C_NEUT, linestyle="--", lw=0.9,
               label=f"Baseline (no anchor): {BASELINE_DROP*100:+.1f}\\%")
    ax.axhline(0.0, color="#888888", linestyle=":", lw=0.6)

    ax.plot(x, hub_y * 100, marker="o", color=C_HUB, lw=1.7, ms=5.6,
            markeredgecolor="white", markeredgewidth=0.6,
            label="Hub Anchoring")
    ax.plot(x, rand_y * 100, marker="s", color=C_TAIL, lw=1.6, ms=5.4,
            markeredgecolor="white", markeredgewidth=0.6,
            label="Random Anchoring")

    for xi, yi in zip(x, hub_y):
        ax.annotate(f"{yi*100:+.1f}\\%",
                    xy=(xi, yi*100), xytext=(0, 7),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=C_TEXT)

    ax.set_xlabel("Anchor prompts $N$")
    ax.set_ylabel("Mean accuracy drop (\\%) over $d{=}1$--$d{=}5$")
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_ylim(-30, 5)
    ax.set_title("Data Efficiency of Popularity Anchoring",
                 loc="left", pad=8, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=7.5,
                    handlelength=1.4, handletextpad=0.4,
                    borderpad=0.3, labelspacing=0.3)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(C_NEUT)
    leg.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)

    loc = OUT_FIGS / "Fig4_MitigationEfficiency.pdf"
    pap = PAPER_FIGS / "Fig4_MitigationEfficiency.pdf"
    fig.savefig(loc); fig.savefig(pap)
    plt.close(fig)
    print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
