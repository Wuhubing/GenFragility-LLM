"""make_fig2_popularity_paradox.py — Fig 2 (a) Vulnerability + (b) Impact.

Reproduces the canonical Fig 2 from
analysis_4models/scripts/plot_v3_paper_figs.py.

Panel (a): per-target macro hard_flip rate (Hub / Mid / Tail neighbors)
under the S2 scope (random-source updates, hops d=2..d=4, default
in-degree thresholds, min target n=5). Numbers come from
v4/lib.loader.load_mask_b_rows + s2_filter — live recompute,
matches tab:flip_by_nbr_class_s2 exactly. 4/4 models show
Hub > Tail; Qwen3.5-9B is strict Hub > Mid > Tail (+7.1 pp gap).

Panel (b): mean d=1..d=5 EPR broken down by update source class,
pinned to the v2 selection (10 targets/group + GPT-4o-mini judge
overturns) because Yuji's results.tex cites Qwen-9B mean=0.580.

Data source (provenance):
    (a) live recompute from comparison_reports/ (S2 scope)
    (b) analysis_4models/v2/fig1_epr_v2.md
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import (  # type: ignore
    MODEL_ORDER, SHORT_NAME, OUT_FIGS, PAPER_FIGS,
    load_mask_b_rows, s2_filter,
)
from lib.style import (  # type: ignore
    set_academic_style, C_HUB, C_TAIL, C_MID, C_NEUT, C_TEXT, W_TEXT,
)


# (a) Live-recomputed below from S2 scope (per-target macro hard_flip).
CLASSES = ["Hub", "Mid", "Tail"]


def _s2_flip_by_class(min_n: int = 5):
    """Return {model -> {class -> flip_rate}} under S2 (per-target macro)."""
    rows = load_mask_b_rows()
    s2 = s2_filter(rows)
    per_tgt = defaultdict(list)
    for r in s2:
        per_tgt[(r["model"], r["target"], r["nbr_default"])].append(
            1 if r["is_flip"] else 0
        )
    per_grp = defaultdict(list)
    for (m, _t, c), vs in per_tgt.items():
        if len(vs) < min_n: continue
        per_grp[(m, c)].append(sum(vs) / len(vs))
    out = defaultdict(dict)
    for (m, c), rates in per_grp.items():
        out[m][c] = sum(rates) / len(rates)
    return out


# (b) Mean d1-d5 EPR by source group (post-judge, v2 selection)
EPR_HUB_SRC  = {"Qwen3.5-2B": 0.376, "Qwen3.5-9B": 0.580,
                "Gemma-4-E4B-it": 0.070, "Gemma-4-31B-it": 0.311}
EPR_TAIL_SRC = {"Qwen3.5-2B": 0.512, "Qwen3.5-9B": 0.575,
                "Gemma-4-E4B-it": 0.146, "Gemma-4-31B-it": 0.255}
EPR_RAND_SRC = {"Qwen3.5-2B": 0.536, "Qwen3.5-9B": 0.596,
                "Gemma-4-E4B-it": 0.116, "Gemma-4-31B-it": 0.186}


def main() -> None:
    set_academic_style()
    print("[load] computing S2 per-class flip rate for panel (a)...")
    flip_a = _s2_flip_by_class()
    for m in MODEL_ORDER:
        c = flip_a.get(m, {})
        print(f"  {m:18s}  H={c.get('Hub',0):.4f}  "
              f"M={c.get('Mid',0):.4f}  T={c.get('Tail',0):.4f}")

    short = [SHORT_NAME[m] for m in MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(W_TEXT, 3.0))

    # ---- (a) Hub / Mid / Tail neighbor flip rate (S2 scope) ----
    ax = axes[0]
    w = 0.26
    vals_h = [flip_a.get(m, {}).get("Hub", 0)  for m in MODEL_ORDER]
    vals_m = [flip_a.get(m, {}).get("Mid", 0)  for m in MODEL_ORDER]
    vals_t = [flip_a.get(m, {}).get("Tail", 0) for m in MODEL_ORDER]
    bars_h = ax.bar(x - w, vals_h, width=w, color=C_HUB,
                    edgecolor="white", lw=0.5, label="Hub nbr")
    bars_m = ax.bar(x,     vals_m, width=w, color=C_MID,
                    edgecolor="white", lw=0.5, label="Mid nbr")
    bars_t = ax.bar(x + w, vals_t, width=w, color=C_TAIL,
                    edgecolor="white", lw=0.5, label="Tail nbr")
    ymax = max(max(vals_h), max(vals_m), max(vals_t)) * 1.35
    for group in (bars_h, bars_m, bars_t):
        for rect in group:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + ymax * 0.02,
                    f"{v*100:.0f}", ha="center", va="bottom",
                    fontsize=6.8, color=C_TEXT)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Macro flip rate (per-target)")
    ax.set_ylim(0, ymax)
    ax.set_title("(a)  Vulnerability: Hub vs Mid vs Tail flip rate "
                 r"(S2: src=Random, $d{=}2$--$d{=}4$)",
                 loc="left", pad=8, fontweight="bold", fontsize=8.5)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
              handlelength=1.2, columnspacing=1.4, frameon=False,
              fontsize=8.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    # ---- (b) ----
    ax = axes[1]
    w = 0.26
    epr_h = [EPR_HUB_SRC[m]  for m in MODEL_ORDER]
    epr_t = [EPR_TAIL_SRC[m] for m in MODEL_ORDER]
    epr_r = [EPR_RAND_SRC[m] for m in MODEL_ORDER]
    bars_h = ax.bar(x - w, epr_h, width=w, color=C_TAIL,
                    edgecolor="white", lw=0.5, label="Src = Hub")
    bars_t = ax.bar(x,     epr_t, width=w, color=C_HUB,
                    edgecolor="white", lw=0.5, label="Src = Tail")
    bars_r = ax.bar(x + w, epr_r, width=w, color=C_MID,
                    edgecolor="white", lw=0.5, label="Src = Random")
    for group in (bars_h, bars_t, bars_r):
        for rect in group:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=C_TEXT)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel(r"Mean EPR  ($d{=}1$--$d{=}5$)")
    ax.set_ylim(0, max(max(epr_h), max(epr_t), max(epr_r)) * 1.35)
    ax.set_title("(b)  Impact: error propagation by source class",
                 loc="left", pad=8, fontweight="bold", fontsize=8.5)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
              handlelength=1.2, columnspacing=1.4, frameon=False,
              fontsize=8.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_NEUT, linestyle=":", lw=0.6, alpha=0.8)
    ax.xaxis.grid(False)

    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.16,
                        left=0.08, right=0.99)
    loc = OUT_FIGS / "Fig2_PopularityParadox.pdf"
    pap = PAPER_FIGS / "Fig2_PopularityParadox.pdf"
    fig.savefig(loc); fig.savefig(pap)
    plt.close(fig)
    print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
