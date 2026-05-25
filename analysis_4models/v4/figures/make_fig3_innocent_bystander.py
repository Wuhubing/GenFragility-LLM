"""make_fig3_innocent_bystander.py — Fig 3 (Innocent Bystander).

Source x Neighbor Delta-Margin heatmap, pooled across the 4 models.

Yuji's results.tex CITES the specific cell values
  Tail-source -> Hub-neighbor = -1.74
  Tail-source -> Tail-neighbor = -0.38
  Hub-source  -> Tail-neighbor = -0.69
from analysis_4models/v2/fig3_innocent_bystander/fig3_crossmodel.md.
That file was computed with:
   - top-5% / bot-5% in-degree thresholds (Hub / Tail)
   - v2 selected_targets.json (10 targets per group)
   - GPT-4o-mini judge overturns for flip
Recomputing from a different subset would break the paper text, so we
pin to those values here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import OUT_FIGS, PAPER_FIGS  # type: ignore
from lib.style import (  # type: ignore
    set_academic_style, C_TAIL, C_MID, C_CALL, C_NEUT, C_TEXT,
)


SOURCES   = ["Hub", "Tail", "Random"]
NEIGHBORS = ["Hub", "Mid", "Tail"]
DMARGIN = np.array([
    [-1.56, -0.78, -0.69],   # Src=Hub
    [-1.74, -0.99, -0.38],   # Src=Tail  -- Innocent Bystander row
    [-1.57, -1.10, -0.72],   # Src=Random
])
COUNTS = np.array([
    [29302, 11822, 740],
    [13527,  6010, 424],
    [23216,  8839, 483],
])


def main() -> None:
    set_academic_style()
    abs_mag = -DMARGIN
    vmin, vmax = abs_mag.min() * 0.95, abs_mag.max() * 1.05
    cmap = LinearSegmentedColormap.from_list(
        "bystander_pastel", ["#FFFFFF", C_TAIL, C_MID], N=256,
    )

    fig, ax = plt.subplots(figsize=(2.40, 2.55))
    im = ax.imshow(abs_mag, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.14, f"{DMARGIN[i, j]:+.2f}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=10, fontweight="bold")
            ax.text(j, i + 0.24, f"n={COUNTS[i, j]:,}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=7, alpha=0.80)

    i_max, j_max = np.unravel_index(np.argmax(abs_mag), abs_mag.shape)
    ax.add_patch(plt.Rectangle((j_max - 0.5, i_max - 0.5), 1, 1,
                               fill=False, edgecolor=C_CALL, lw=1.8))

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{n}-nbr" for n in NEIGHBORS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Src={s}" for s in SOURCES])
    ax.set_xlabel("Neighbor class")
    ax.set_ylabel("Update source class")
    ax.set_title("Innocent Bystander: $\\Delta\\mathrm{Margin}$",
                 loc="left", pad=8, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(r"$|\Delta\mathrm{Margin}|$ (deeper = larger collapse)",
                   fontsize=8, color=C_TEXT)
    cbar.ax.tick_params(labelsize=7.5, color=C_TEXT, labelcolor=C_TEXT)
    cbar.outline.set_edgecolor(C_NEUT)
    cbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.16)
    loc = OUT_FIGS / "Fig3_InnocentBystander.pdf"
    pap = PAPER_FIGS / "Fig3_InnocentBystander.pdf"
    fig.savefig(loc); fig.savefig(pap)
    plt.close(fig)
    print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
