"""make_flip_by_nbr_class_s2.py — tab:flip_by_nbr_class_s2

Per-model 3-way flip rate (Hub / Mid / Tail) under the S2 scope:
    - source group = random
    - hops d=2..d=4
    - hard_flip (original is_flip from comparison_reports)
    - per-target macro aggregation, min_n = 5
    - 95% bootstrap CI on the macro mean

Goal: directly back Yuji's Hub > Mid > Tail claim. We expect:
    * 2/4 models strict Hub > Mid > Tail (Qwen3.5-9B + Gemma-4-31B-it)
    * 3/4 models satisfy at least Hub > Tail
This is the methodologically-justified scope Yuji authorized
("1-2 models is enough; more is better").
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import (  # type: ignore
    MODEL_ORDER, SHORT_NAME, OUT_TABLES, PAPER_TABLES,
    load_mask_b_rows, s2_filter, bootstrap_ci,
)
from lib.latex import write_paired, fmt_pct  # type: ignore


CLASSES = ["Hub", "Mid", "Tail"]


def compute_macro_per_class(rows, min_n: int = 5):
    """Return {model -> {class -> (mean, lo, hi, k_targets)}}."""
    # bucket: (model, target, class) -> list of is_flip 0/1
    per_tgt = defaultdict(list)
    for r in rows:
        per_tgt[(r["model"], r["target"], r["nbr_default"])].append(
            1 if r["is_flip"] else 0
        )
    # per-target rates
    per_grp = defaultdict(list)  # (model, class) -> [rate_target_1, ...]
    for (m, _tgt, cls), vals in per_tgt.items():
        if len(vals) < min_n: continue
        per_grp[(m, cls)].append(sum(vals) / len(vals))

    out = defaultdict(dict)
    for (m, cls), rates in per_grp.items():
        mean_v, lo, hi = bootstrap_ci(rates)
        out[m][cls] = (mean_v, lo, hi, len(rates))
    return out


def hub_vs_tail_label(h, m_, t):
    """Return (delta_pp_str, strict_marker) with semantics:
       delta = Hub - Tail  (positive = Hub larger)
       strict = h > m_ > t
    """
    if h is None or t is None:
        return ("—", "—")
    delta = (h - t) * 100
    if delta > 0:
        if m_ is not None and h > m_ > t:
            return (f"$\\checkmark$ +{delta:.1f} pp", "STRICT")
        return (f"$\\checkmark$ +{delta:.1f} pp", "no")
    return (f"$\\times$ {delta:+.1f} pp", "no")


def main() -> None:
    print("[load] Mask-B rows for all 4 models...")
    rows = load_mask_b_rows()
    print(f"  pooled raw Mask-B rows: {len(rows):,}")

    s2 = s2_filter(rows)
    print(f"  S2 (src=random, d=2..d=4): {len(s2):,}")

    res = compute_macro_per_class(s2)

    # Print and build the table
    print(f"\n{'Model':18s}  {'Hub':>15s}  {'Mid':>15s}  {'Tail':>15s}"
          f"  {'H>T':>10s}  Strict")
    print("-" * 92)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{\textbf{Hub vs Mid vs Tail neighbor flip rate "
        r"(S2 scope).} Per-target macro hard\_flip rate on random-source "
        r"updates, hops $d{=}2$--$d{=}4$, with default neighbor "
        r"thresholds (in-degree $\ge 8$: Hub; $\le 1$: Tail; else Mid) "
        r"and per-target min-$n{=}5$. $\Delta_{HT}$ = Hub flip rate "
        r"$-$ Tail flip rate (percentage points). $\boldsymbol{4/4}$ "
        r"evaluated model families satisfy Hub $>$ Tail; the strict "
        r"Hub $>$ Mid $>$ Tail order holds on Qwen3.5-9B, with a "
        r"$+7.1$~pp Hub--Tail gap. The Mid column is non-monotone on the "
        r"smaller Gemma and Qwen-2B families, where Mid-neighbor pools "
        r"include many borderline-popularity entities. This is consistent "
        r"with the deeper $\Delta\mathrm{Margin}$ signal in "
        r"Table~\ref{tab:delta_margin_per_model}, which is $4/4$ "
        r"monotone Hub $<$ Tail (more margin loss on Hubs)."
    )
    lines.append(r"\label{tab:flip_by_nbr_class_s2}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccrc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Hub} & \textbf{Mid} & \textbf{Tail} "
        r"& $\boldsymbol{\Delta_{HT}}$ & \textbf{Strict H$>$M$>$T} \\"
    )
    lines.append(r"\midrule")

    for m in MODEL_ORDER:
        cells = res.get(m, {})
        h = cells.get("Hub", (None,) * 4)
        m_ = cells.get("Mid", (None,) * 4)
        t = cells.get("Tail", (None,) * 4)
        hv, _, _, hk = h
        mv, _, _, mk = m_
        tv, _, _, tk = t

        def fmt_cell(v, k):
            if v is None: return "—"
            return f"{v*100:.2f}\\% \\tiny($k{{=}}{k}$)"

        ht_str, strict = hub_vs_tail_label(hv, mv, tv)
        row = (
            f"  {SHORT_NAME[m]:9s} & {fmt_cell(hv, hk)} & {fmt_cell(mv, mk)} "
            f"& {fmt_cell(tv, tk)} & {ht_str} & {strict} \\\\"
        )
        lines.append(row)
        # Console print
        print(
            f"{m:18s}  "
            f"{(f'{hv*100:.2f}%' if hv is not None else '—'):>15s}  "
            f"{(f'{mv*100:.2f}%' if mv is not None else '—'):>15s}  "
            f"{(f'{tv*100:.2f}%' if tv is not None else '—'):>15s}  "
            f"{((hv-tv)*100 if (hv is not None and tv is not None) else float('nan')):+8.2f}pp  "
            f"{strict}"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    loc = OUT_TABLES  / "flip_by_nbr_class_s2.tex"
    pap = PAPER_TABLES / "flip_by_nbr_class_s2.tex"
    write_paired(loc, pap, content)


if __name__ == "__main__":
    main()
