"""make_fig7_mitigation_blast.py — Fig 7.

Absolute flipped-fact count by (source group x anchor mode) on
Qwen3.5-9B's anchor experiment (17 shared targets, pooled d=1..5).

We recompute the absolute flip counts directly from
main_output/Qwen3.5-9B_anchor_full30_experiment/<mode>/.

Anchor modes: none, popularity_top5, popularity_top25, popularity_top75.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import MAIN_OUT, OUT_FIGS, PAPER_FIGS  # type: ignore
from lib.style import (  # type: ignore
    set_academic_style, C_TAIL, C_HUB, C_MID, C_NEUT, C_TEXT, C_CALL, W_TEXT,
)


ANCHOR_ROOT = MAIN_OUT / "Qwen3.5-9B_anchor_full30_experiment"
MODES = [
    ("none",              "none"),
    ("popularity_top5",   "pop. top5"),
    ("popularity_top25",  "pop. top25"),
    ("popularity_top75",  "pop. top75"),
]
SOURCES = ["hub", "tail", "random"]
HOPS    = ["d1", "d2", "d3", "d4", "d5"]


def collect_flips_per_mode() -> dict:
    """Return {mode_code -> {src -> total_flips_d1_to_d5}}."""
    out = {}
    for code, _ in MODES:
        per_src = {s: 0 for s in SOURCES}
        d = ANCHOR_ROOT / code
        if not d.exists():
            out[code] = per_src; continue
        for tdir in sorted(d.iterdir()):
            if not tdir.is_dir(): continue
            nm = tdir.name
            src = nm.split("_")[0]
            if src not in SOURCES: continue
            fp = tdir / "comparison_reports" / f"{nm}_vllm_comparison.json"
            if not fp.exists(): continue
            try:
                j = json.loads(fp.read_text())
            except Exception:
                continue
            for r in j.get("unified_results", []):
                if r.get("distance") not in HOPS: continue
                if r.get("clean_accuracy") != 1.0: continue
                if bool(r.get("is_flip")):
                    per_src[src] += 1
        out[code] = per_src
        print(f"  mode={code:18s}  hub={per_src['hub']:5d}  "
              f"tail={per_src['tail']:5d}  random={per_src['random']:5d}")
    return out


def main() -> None:
    set_academic_style()
    print("[load] counting absolute flip counts per anchor mode...")
    flips = collect_flips_per_mode()

    hub  = [flips[c]["hub"]    for c, _ in MODES]
    tail = [flips[c]["tail"]   for c, _ in MODES]
    rand = [flips[c]["random"] for c, _ in MODES]

    base_tail = tail[0] if tail[0] > 0 else 1
    tail_delta = [None] + [100 * (t - base_tail) / base_tail for t in tail[1:]]

    mode_labels = [lbl for _, lbl in MODES]
    x = np.arange(len(MODES))
    w = 0.26

    fig, ax = plt.subplots(figsize=(W_TEXT, 3.1))
    bars_h = ax.bar(x - w, hub,  width=w, color=C_TAIL, edgecolor="white",
                    lw=0.5, label="Src = Hub")
    bars_t = ax.bar(x,     tail, width=w, color=C_HUB,  edgecolor="white",
                    lw=0.5, label="Src = Tail")
    bars_r = ax.bar(x + w, rand, width=w, color=C_MID,  edgecolor="white",
                    lw=0.5, label="Src = Random")

    for group in (bars_h, bars_t, bars_r):
        for rect in group:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 80,
                    f"{int(v):,}", ha="center", va="bottom",
                    fontsize=7.5, color=C_TEXT)

    for rect, d in zip(bars_t, tail_delta):
        if d is None: continue
        ax.annotate(
            f"{d:+.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, -14), textcoords="offset points",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=C_CALL,
        )

    ax.set_xticks(x); ax.set_xticklabels(mode_labels)
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

    loc = OUT_FIGS / "Fig7_MitigationBlastRadius.pdf"
    pap = PAPER_FIGS / "Fig7_MitigationBlastRadius.pdf"
    fig.savefig(loc); fig.savefig(pap)
    plt.close(fig)
    print(f"[fig] {loc}\n  -> {pap}")


if __name__ == "__main__":
    main()
