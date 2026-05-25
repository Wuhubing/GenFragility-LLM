"""make_broad_similarity.py

Generate tables/broad_similarity.tex — the 5-bin pooled Mask-B flip rate
vs Levenshtein L(subject, head) similarity for the 94,363 source-neighbor
pairs across the 4 evaluated models.

Source-of-truth CSV: analysis_4models/v2/lexical/per_fact_lev.csv.gz
columns: model, src_group, target, hop, subject, head, tail,
         is_flip_raw, is_flip_judge, L_sh, L_sq, L_aR, L_tR
"""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import REPO_ROOT, OUT_TABLES, PAPER_TABLES  # type: ignore
from lib.latex import write_paired


LEV_CSV = REPO_ROOT / "analysis_4models/v2/lexical/per_fact_lev.csv.gz"

BINS = [
    ("$[0.0, 0.2)$ (Low)",   0.0,  0.2,  "Dominant Failure Mode"),
    ("$[0.2, 0.4)$ (Mid)",   0.2,  0.4,  "Frequent Propagation"),
    ("$[0.4, 0.6)$ (High)",  0.4,  0.6,  "Moderate Correlation"),
    ("$[0.6, 0.8)$",          0.6,  0.8,  "High Risk, Rare"),
    ("$[0.8, 1.0]$ (Exact)", 0.8,  1.001, "High Risk, Rare Occurrence"),
]


def main() -> None:
    n_bin = [0] * len(BINS)
    f_bin = [0] * len(BINS)
    n_total = 0

    with gzip.open(LEV_CSV, "rt") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                lsh = float(row["L_sh"])
                isf = int(row["is_flip_judge"])
            except (KeyError, ValueError):
                continue
            n_total += 1
            for i, (_, lo, hi, _) in enumerate(BINS):
                if lo <= lsh < hi:
                    n_bin[i] += 1
                    f_bin[i] += isf
                    break

    print(f"[load] {n_total:,} rows from {LEV_CSV.name}")
    for (label, lo, hi, obs), n, fl in zip(BINS, n_bin, f_bin):
        rate = fl / n if n else 0.0
        print(f"  {label:24s} n={n:6,}  flip={rate*100:5.2f}%")

    # Locate the bin with the maximum flip rate, bold it.
    rates = [f / n if n else 0 for f, n in zip(f_bin, n_bin)]
    max_i = max(range(len(rates)), key=lambda i: rates[i])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Broad impact of surface similarity (four-model Mask B "
        rf"pool).}} In the ${n_total:,}$ Mask B source-neighbor pairs pooled across "
        r"the four evaluated models, highly similar entity pairs ($\ge 0.8$) flip at "
        r"$\sim$$50\%$ but represent under $1\%$ of the data. Most errors occur in "
        r"the low-similarity range with a near-constant flip rate.}",
        r"\label{tab:broad_similarity}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l c c l}",
        r"\toprule",
        r"\textbf{Similarity Range} & \textbf{Count} & \textbf{Flip Rate (\%)} & \textbf{Observation} \\",
        r"\midrule",
    ]
    for i, ((label, _, _, obs), n, fl) in enumerate(zip(BINS, n_bin, f_bin)):
        rate = fl / n if n else 0.0
        rate_str = f"\\textbf{{{rate*100:.2f}\\%}}" if i == max_i \
                   else f"{rate*100:.2f}\\%"
        lines.append(f"{label} & {n:,} & {rate_str} & {obs} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    content = "\n".join(lines)

    write_paired(
        OUT_TABLES / "broad_similarity.tex",
        PAPER_TABLES / "broad_similarity.tex",
        content,
    )


if __name__ == "__main__":
    main()
