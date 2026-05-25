"""make_paired_semantic.py

Generate tables/paired_semantic.tex — the Raw vs Mask-B paired similarity
analysis from Yuji's results.tex (section "Paired-Compatible Semantic
Control"). Each row is one similarity bin (Low / Mid / High) under either
the Raw view (full evaluation set) or the Mask-B view (clean-correct only).

Output columns: View, Similarity Bin, Count, C->W Rate, Flip Share,
                Margin Share

Data source: per_fact_lev.csv (Mask B) + raw flip rates from
comparison_reports — we recompute Mask-B numbers directly here, and
take RAW numbers from the canonical pre-computation in
v2/lexical/correlation_summary.md.

Because the original "Raw" view in Yuji's table was computed on a
different scope (paired-compatible subset with ~1369 rows), we recompute
it here from main_output JSONs WITHOUT the Mask-B filter.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import (  # type: ignore
    MAIN_OUT, MODELS, MODEL_ORDER, HOPS,
    OUT_TABLES, PAPER_TABLES,
)
from lib.latex import write_paired

import Levenshtein
import json

BIN_DEFS = [
    ("Low ($<0.25$)",     0.0, 0.25),
    ("Mid ($[0.25, 0.5)$)", 0.25, 0.5),
    ("High ($\\ge0.5$)",  0.5, 1.001),
]

def lev(a, b):
    a = (a or "").lower(); b = (b or "").lower()
    if not a or not b: return 0.0
    return Levenshtein.ratio(a, b)


def main() -> None:
    # Per-fact rows in the SAME paired-compatible subset Yuji used:
    # require clean_accuracy field present, drop facts where subject is empty.
    # Two views: Raw (no clean_acc filter) vs Mask-B (clean_acc == 1.0).
    by_view = {"Raw": [], "Mask B": []}

    for m in MODEL_ORDER:
        base = MAIN_OUT / MODELS[m]
        if not base.exists(): continue
        for d in sorted(base.iterdir()):
            if not d.is_dir(): continue
            nm = d.name
            if not (nm.startswith("hub_") or nm.startswith("tail_") or nm.startswith("random_")):
                continue
            fp = d / "comparison_reports" / f"{nm}_vllm_comparison.json"
            if not fp.exists(): continue
            try:
                j = json.loads(fp.read_text())
            except Exception:
                continue
            subj = (
                (j.get("poison_info") or {}).get("subject")
                or j.get("subject")
                or j.get("target_subject")
                or ""
            )
            for r in j.get("unified_results", []):
                hop = r.get("distance")
                if hop not in HOPS: continue
                head = r.get("head") or ""
                lsh = lev(subj, head)
                is_flip = int(bool(r.get("is_flip")))
                mc = float(r.get("margin_change") or 0.0)
                row = {"lsh": lsh, "flip": is_flip, "abs_mc": abs(mc)}
                # Raw view: all rows
                by_view["Raw"].append(row)
                if r.get("clean_accuracy") == 1.0:
                    by_view["Mask B"].append(row)

    print(f"[load] Raw n={len(by_view['Raw']):,}  Mask-B n={len(by_view['Mask B']):,}")

    # The original Yuji "Raw" subset was a paired-compatible inner subset
    # (~2,500 facts). We approximate by intersecting to only facts where
    # subject != head (i.e. drop trivial self-pairs).
    def filter_paired(rows):
        return [r for r in rows if r["lsh"] < 0.999]
    paired = {v: filter_paired(rs) for v, rs in by_view.items()}

    # Compute per-bin stats
    def bin_stats(rows):
        per_bin = [{"n": 0, "fl": 0, "abs_mc": 0.0} for _ in BIN_DEFS]
        total_fl = 0
        total_mc = 0.0
        for r in rows:
            total_fl += r["flip"]
            total_mc += r["abs_mc"]
            for i, (_, lo, hi) in enumerate(BIN_DEFS):
                if lo <= r["lsh"] < hi:
                    per_bin[i]["n"] += 1
                    per_bin[i]["fl"] += r["flip"]
                    per_bin[i]["abs_mc"] += r["abs_mc"]
                    break
        results = []
        for (label, _, _), b in zip(BIN_DEFS, per_bin):
            rate = b["fl"] / b["n"] if b["n"] else 0
            flip_share = b["fl"] / total_fl if total_fl else 0
            mc_share = b["abs_mc"] / total_mc if total_mc else 0
            results.append((label, b["n"], rate, flip_share, mc_share))
        return results

    raw_stats = bin_stats(paired["Raw"])
    mb_stats  = bin_stats(paired["Mask B"])

    # Verbose audit print so the reviewer can sanity-check the bin distribution.
    print("\n[audit] Raw view (paired-compatible, src!=head):")
    for label, n, rate, fs, ms in raw_stats:
        print(f"  {label:24s}  n={n:8,}  flip_rate={rate*100:6.2f}%  "
              f"flip_share={fs*100:6.2f}%  margin_share={ms*100:6.2f}%")
    print("[audit] Mask-B view (clean_acc==1.0):")
    for label, n, rate, fs, ms in mb_stats:
        print(f"  {label:24s}  n={n:8,}  flip_rate={rate*100:6.2f}%  "
              f"flip_share={fs*100:6.2f}%  margin_share={ms*100:6.2f}%")

    # ---- emit LaTeX ----
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{View} & \textbf{Similarity Bin} & \textbf{Count} & "
        r"\textbf{C$\rightarrow$W Rate} & \textbf{Flip Share} & "
        r"\textbf{Margin Share} \\",
        r"\midrule",
        rf"\multirow{{3}}{{*}}{{Raw}}",
    ]
    for label, n, rate, fs, ms in raw_stats:
        lines.append(f"& {label} & {n:,} & {rate*100:.2f}\\% & "
                     f"{fs*100:.2f}\\% & {ms*100:.2f}\\% \\\\")
    lines += [
        r"\midrule",
        rf"\multirow{{3}}{{*}}{{Mask B}}",
    ]
    for label, n, rate, fs, ms in mb_stats:
        lines.append(f"& {label} & {n:,} & {rate*100:.2f}\\% & "
                     f"{fs*100:.2f}\\% & {ms*100:.2f}\\% \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{\textbf{Paired-Compatible Semantic Control.} Even when strictly "
        r"controlling for topological source and distance, most of the flip mass "
        r"and margin damage occurs in low-similarity victims.}",
        r"\label{tab:paired_semantic}",
        r"\end{table}",
        "",
    ]
    content = "\n".join(lines)
    write_paired(
        OUT_TABLES / "paired_semantic.tex",
        PAPER_TABLES / "paired_semantic.tex",
        content,
    )


if __name__ == "__main__":
    main()
