"""make_delta_margin_per_model.py

Regenerate tables/delta_margin_per_model.tex (Hub-neighbor margin collapse,
pooled d1-d5 on Mask B) — 4/4 monotone Hub<Tail across the 4 models.

This faithfully reproduces the table that already lives in the paper at
tables/delta_margin_per_model.tex, recomputing the numbers directly from
main_output/<model>/<target>/comparison_reports/*_vllm_comparison.json so
the artifact is provably runnable.

Output: v4/outputs/tables/delta_margin_per_model.tex
        + mirror to <paper>/tables/delta_margin_per_model.tex
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# allow `from v4.lib.loader import ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import (  # type: ignore
    MODEL_ORDER, OUT_TABLES, PAPER_TABLES, load_mask_b_rows,
)
from lib.latex import write_paired


def main() -> None:
    rows = load_mask_b_rows()
    print(f"[load] {len(rows):,} Mask-B rows")

    agg = defaultdict(lambda: [0, 0.0])  # (model, nbr) -> [n, sum_dm]
    for r in rows:
        agg[(r["model"], r["nbr_default"])][0] += 1
        agg[(r["model"], r["nbr_default"])][1] += r["margin_change"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Hub-neighbor margin collapse is consistent across all four "
        r"evaluated model families.} Mean correct-token margin change "
        r"$\Delta\mathrm{Margin} = \mathrm{Margin}_{\text{post}} - "
        r"\mathrm{Margin}_{\text{pre}}$, pooled over $d{=}1$ to $d{=}5$ on the "
        r"Mask-B subset (pre-update-correct facts; see Section~\ref{sec:setup}). "
        r"More negative $=$ deeper confidence collapse. The Hub-neighbor column "
        r"suffers the deepest drop in \textbf{4/4} models (Hub--Tail gap shown in "
        r"the last column). The signal also persists on the heavily "
        r"instruction-tuned Gemma-4-E4B-it, where output-level flip behaviour is "
        r"otherwise heavily suppressed.}",
        r"\label{tab:delta_margin_per_model}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Hub} & \textbf{Mid} & \textbf{Tail} & "
        r"\textbf{Hub $-$ Tail} \\",
        r"\midrule",
    ]

    hub_lt_tail_count = 0
    for m in MODEL_ORDER:
        h_n, h_s = agg[(m, "Hub")]
        mid_n, mid_s = agg[(m, "Mid")]
        t_n, t_s = agg[(m, "Tail")]
        h = h_s / h_n if h_n else 0.0
        md = mid_s / mid_n if mid_n else 0.0
        t = t_s / t_n if t_n else 0.0
        gap = h - t
        if h < t: hub_lt_tail_count += 1

        def cell(v, n):
            sign = "" if v < 0 else "+"
            return rf"$\mathbf{{{sign}{v:.2f}}}$ \tiny{{(n{{=}}{n:,})}}" \
                if v < 0 else rf"${sign}{v:.2f}$ \tiny{{(n{{=}}{n:,})}}"

        # Hub cell always bolded per the paper's existing style
        h_cell  = rf"$\mathbf{{{h:+.2f}}}$ \tiny{{(n{{=}}{h_n:,})}}".replace("+", "")
        h_cell  = rf"$\mathbf{{{h:.2f}}}$ \tiny{{(n{{=}}{h_n:,})}}" if h < 0 \
            else rf"$\mathbf{{+{h:.2f}}}$ \tiny{{(n{{=}}{h_n:,})}}"
        md_cell = (rf"${md:.2f}$ \tiny{{(n{{=}}{mid_n:,})}}" if md < 0
                   else rf"$+{md:.2f}$ \tiny{{(n{{=}}{mid_n:,})}}")
        t_cell  = (rf"${t:.2f}$ \tiny{{(n{{=}}{t_n:,})}}" if t < 0
                   else rf"$+{t:.2f}$ \tiny{{(n{{=}}{t_n:,})}}")
        gap_cell = rf"$\mathbf{{{gap:.2f}}}$" if gap < 0 \
            else rf"$\mathbf{{+{gap:.2f}}}$"

        lines.append(f"{m:14s} & {h_cell} & {md_cell} & {t_cell} & {gap_cell} \\\\")

    lines += [
        r"\midrule",
        rf"\textbf{{Hub $>$ Tail collapse}} & \multicolumn{{4}}{{c}}{{\textbf{{{hub_lt_tail_count} / 4 models}}}} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    content = "\n".join(lines)

    write_paired(
        OUT_TABLES / "delta_margin_per_model.tex",
        PAPER_TABLES / "delta_margin_per_model.tex",
        content,
    )


if __name__ == "__main__":
    main()
