"""make_fig5_flip_vs_similarity.py — Fig 5.

Per-model Flip Rate as a function of Levenshtein(subject, head) bins, with
log-scale background bars for bin frequency.

Recomputed from analysis_4models/v2/lexical/per_fact_lev.csv.gz (the
94,363-row CSV with is_flip_judge + L_sh per fact) — this is the same
source Yuji's text cites.
"""
from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import REPO_ROOT, MODEL_ORDER, OUT_FIGS, PAPER_FIGS  # type: ignore
from lib.style import (  # type: ignore
    set_academic_style, MODEL_COLORS, C_NEUT, C_TEXT, W_TEXT,
)


LEV_CSV = REPO_ROOT / "analysis_4models/v2/lexical/per_fact_lev.csv.gz"

BINS = [
    ("[0.0,0.2)", 0.0,  0.2),
    ("[0.2,0.4)", 0.2,  0.4),
    ("[0.4,0.6)", 0.4,  0.6),
    ("[0.6,0.8)", 0.6,  0.8),
    ("[0.8,1.0]", 0.8,  1.001),
]


def main() -> None:
    set_academic_style()

    # Per-model per-bin n / flip
    pm = {m: [[0, 0] for _ in BINS] for m in MODEL_ORDER}
    bin_n_total = [0] * len(BINS)

    # Pearson r per model accumulators (x = L_sh, y = flip)
    acc = {m: {"n": 0, "sx": 0.0, "sy": 0.0,
               "sxx": 0.0, "syy": 0.0, "sxy": 0.0} for m in MODEL_ORDER}

    with gzip.open(LEV_CSV, "rt") as f:
        rd = csv.DictReader(f)
        for row in rd:
            m = row["model"]
            if m not in pm: continue
            try:
                x = float(row["L_sh"])
                y = int(row["is_flip_judge"])
            except (KeyError, ValueError):
                continue
            for i, (_, lo, hi) in enumerate(BINS):
                if lo <= x < hi:
                    pm[m][i][0] += 1
                    pm[m][i][1] += y
                    bin_n_total[i] += 1
                    break
            a = acc[m]
            a["n"]   += 1
            a["sx"]  += x
            a["sy"]  += y
            a["sxx"] += x * x
            a["syy"] += y * y
            a["sxy"] += x * y

    def pearson(a):
        n = a["n"]
        if n == 0: return 0.0
        num   = n * a["sxy"] - a["sx"] * a["sy"]
        den_x = (n * a["sxx"] - a["sx"] ** 2) ** 0.5
        den_y = (n * a["syy"] - a["sy"] ** 2) ** 0.5
        if den_x == 0 or den_y == 0: return 0.0
        return num / (den_x * den_y)

    r_vals = {m: pearson(acc[m]) for m in MODEL_ORDER}

    bin_labels = [b[0] for b in BINS]
    x = np.arange(len(BINS))

    fig, ax = plt.subplots(figsize=(W_TEXT, 3.1))
    ax_n = ax.twinx()
    ax_n.bar(x, bin_n_total, color=C_NEUT, edgecolor="none",
             zorder=0, width=0.78, alpha=0.55)
    ax_n.set_yscale("log")
    ax_n.set_ylim(1, max(bin_n_total) * 14)
    ax_n.set_ylabel("Source-neighbor pair count (log)",
                    fontsize=9, color=C_TEXT)
    ax_n.tick_params(axis="y", labelsize=8.5, color=C_TEXT, labelcolor=C_TEXT)
    ax_n.grid(False)
    for spine in ax_n.spines.values(): spine.set_visible(False)

    for m in MODEL_ORDER:
        rates = [(pm[m][i][1] / pm[m][i][0]) if pm[m][i][0] else 0
                 for i in range(len(BINS))]
        ax.plot(x, rates, marker="o", color=MODEL_COLORS[m], lw=1.7,
                ms=5.4, markeredgecolor="white", markeredgewidth=0.6,
                label=f"{m}  ($r={r_vals[m]:+.3f}$)")

    ax.set_zorder(ax_n.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=0)
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

    loc = OUT_FIGS / "Fig5_FlipVsSimilarity.pdf"
    pap = PAPER_FIGS / "Fig5_FlipVsSimilarity.pdf"
    fig.savefig(loc); fig.savefig(pap)
    plt.close(fig)
    print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
