"""make_fig6_semantic_drift.py — Fig 6.

Judge-free semantic drift (Src x Nbr) heatmap. Pinned to the v2 strict-d=0
pool because Yuji's results.tex cites "57,111 Mask B facts" + specific
drift values (0.255-0.345) computed under that scope.

Source-of-truth file:
    analysis_4models/v2/strict_d0/semantic_drift_summary.md
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
DRIFT = np.array([
    [0.2812, 0.2839, 0.2869],   # Src=Hub
    [0.2551, 0.2939, 0.3412],   # Src=Tail
    [0.3429, 0.3062, 0.3447],   # Src=Random
])
COUNTS = np.array([
    [26462, 10634, 629],
    [8803,  4067, 257],
    [4277,  1885,  97],
])


def main() -> None:
    set_academic_style()
    cmap = LinearSegmentedColormap.from_list(
        "drift_pastel", ["#FFFFFF", C_TAIL, C_MID], N=256,
    )

    fig, ax = plt.subplots(figsize=(2.40, 2.50))
    vmin, vmax = DRIFT.min() * 0.96, DRIFT.max() * 1.02
    im = ax.imshow(DRIFT, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.14, f"{DRIFT[i, j]:.3f}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=9.5, fontweight="bold")
            ax.text(j, i + 0.24, f"n={COUNTS[i, j]:,}",
                    ha="center", va="center",
                    color=C_TEXT, fontsize=7, alpha=0.80)

    i_max, j_max = np.unravel_index(np.argmax(DRIFT), DRIFT.shape)
    ax.add_patch(plt.Rectangle((j_max - 0.5, i_max - 0.5), 1, 1,
                               fill=False, edgecolor=C_CALL, lw=1.6))

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{n}-nbr" for n in NEIGHBORS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Src={s}" for s in SOURCES])
    ax.set_xlabel("Neighbor class")
    ax.set_ylabel("Update source class")
    ax.set_title("Judge-free semantic drift", loc="left",
                 pad=8, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(
        r"Mean drift  $1 - \cos(y_{\mathrm{pre}}, y_{\mathrm{post}})$",
        fontsize=8, color=C_TEXT,
    )
    cbar.ax.tick_params(labelsize=7.5, color=C_TEXT, labelcolor=C_TEXT)
    cbar.outline.set_edgecolor(C_NEUT)
    cbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.16)
    loc = OUT_FIGS / "Fig6_SemanticDrift.pdf"
    pap = PAPER_FIGS / "Fig6_SemanticDrift.pdf"
    fig.savefig(loc); fig.savefig(pap)
    plt.close(fig)
    print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
