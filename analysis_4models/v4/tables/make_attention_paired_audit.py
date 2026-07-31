"""make_attention_paired_audit.py — replaces tab:attention_lift_by_hop.

Source-of-truth: the Qwen3.5-9B paired audit CSV produced by
`scripts/attention_paired_audit.py`:

    analysis_4models/v4/outputs/attention_paired/qwen3.5-9b_paired.csv

Per (source_class, hop) bucket we report:
    n
    mean |delta_att_lift_abs|      (using the "last_full" metric by default)
    std, se = std/sqrt(n)
    bootstrap 95% CI (percentile, 10 000 resamples, seed 42)
    C->W rate = (#flips among clean-correct) / (#clean-correct)

The "last_full" metric = absolute pre/post change in attention lift measured
on the last full-attention layer at the first generated token.
Qwen3.5 is hybrid (linear_attention + full_attention every 4 layers); the
auditor also dumps a `mean_full` (mean over all 8 full-attn layers); both
columns are present in the CSV so reviewers can re-aggregate either way.

Deliverables (4):
  1. v4/outputs/tables/attention_lift_by_hop.tex       (main paper table)
  2. v4/outputs/tables/attention_lift_by_hop_appendix.tex (n + CI + C->W)
  3. v4/outputs/figures/attention_lift_by_hop.{pdf,png}
  4. CSV remains unchanged at the audit output path
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import REPO_ROOT, OUT_TABLES, OUT_FIGS, PAPER_TABLES, PAPER_FIGS  # type: ignore
from lib.latex import write_paired  # type: ignore


# --------------------------- config -----------------------------------------

AUDIT_CSV = (
    REPO_ROOT
    / "analysis_4models/v4/outputs/attention_paired/qwen3.5-9b_paired.csv"
)

# Which metric to use for the main paper table. "last_full" matches the
# paper-traditional "final transformer layer" convention. "mean_full" is
# kept as an ablation column in the appendix.
PRIMARY_METRIC = "last_full"

HOPS_ORDER  = ["d1", "d2", "d3", "d4", "d5"]
CLASS_ORDER = ["Popular", "Rare"]

N_BOOT = 10_000
BOOT_SEED = 42


# --------------------------- load CSV ---------------------------------------

def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def parse_float(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------- statistics -------------------------------------

def bootstrap_ci(samples: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                 alpha: float = 0.05) -> tuple[float | None, float | None]:
    if len(samples) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


def aggregate(rows: list[dict], metric: str) -> dict[tuple[str, str], dict]:
    """{(class, hop) -> {n, mean, std, se, ci_lo, ci_hi, cw_rate}}."""
    col = f"delta_att_lift_abs_{metric}"
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    # For C->W: count rows where clean_accuracy == 1 and is_flip true,
    # divided by rows with clean_accuracy == 1, within the bucket.
    cw_num: dict[tuple[str, str], int] = defaultdict(int)
    cw_den: dict[tuple[str, str], int] = defaultdict(int)

    for r in rows:
        cls = r["source_class"]
        hop = r["hop"]
        if cls not in CLASS_ORDER or hop not in HOPS_ORDER:
            continue
        v = parse_float(r[col])
        if v is not None:
            buckets[(cls, hop)].append(v)
        # C->W aggregation: every row already passed the Mask B filter at
        # selection time (clean_accuracy == 1.0). Defensive check anyway.
        if parse_float(r["clean_accuracy"]) == 1.0:
            cw_den[(cls, hop)] += 1
            if str(r["is_flip"]).strip() in ("1", "True", "true"):
                cw_num[(cls, hop)] += 1

    out = {}
    for cls in CLASS_ORDER:
        for hop in HOPS_ORDER:
            vals = buckets[(cls, hop)]
            n = len(vals)
            if n == 0:
                out[(cls, hop)] = dict(n=0, mean=None, std=None, se=None,
                                       ci_lo=None, ci_hi=None,
                                       cw_rate=None, cw_n=cw_den[(cls, hop)])
                continue
            mu = float(mean(vals))
            sd = float(pstdev(vals)) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n > 0 else None
            lo, hi = bootstrap_ci(vals)
            cw_rate = (cw_num[(cls, hop)] / cw_den[(cls, hop)]) if cw_den[(cls, hop)] else None
            out[(cls, hop)] = dict(n=n, mean=mu, std=sd, se=se,
                                   ci_lo=lo, ci_hi=hi,
                                   cw_rate=cw_rate, cw_n=cw_den[(cls, hop)])
    return out


# --------------------------- emit main table --------------------------------

def fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def emit_main_tex(agg: dict[tuple[str, str], dict]) -> str:
    """Main paper table — class x hop mean |Δ AttLift|, n in subscript caption."""
    lines = [
        "% generated by analysis_4models/v4/tables/make_attention_paired_audit.py",
        "% source: analysis_4models/v4/outputs/attention_paired/qwen3.5-9b_paired.csv",
        f"% metric: delta_att_lift_abs_{PRIMARY_METRIC}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(HOPS_ORDER) + r"}",
        r"\toprule",
        (r"\textbf{Source class} & " +
         " & ".join(rf"\textbf{{{h}}}" for h in HOPS_ORDER) + r" \\"),
        r"\midrule",
    ]
    for cls in CLASS_ORDER:
        cells = []
        for hop in HOPS_ORDER:
            cells.append(fmt(agg[(cls, hop)]["mean"]))
        lines.append(f"{cls} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{\label{tab:attention_lift_by_hop} \textbf{Attention "
         r"perturbation across hop distance.} "
         r"We report the mean absolute pre/post change in attention lift on "
         r"neighbor entity spans in the Qwen3.5-9B paired audit "
         r"($|\Delta \mathrm{AttLift}|$, last full-attention layer, "
         r"first generated token). "
         r"Popular-source updates induce stronger early-hop perturbations, "
         r"while distant-hop effects are more mixed. "
         r"Bootstrap 95\% confidence intervals and per-cell sample counts "
         r"are reported in Table~\ref{tab:attention_lift_by_hop_appendix}.}"),
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def emit_appendix_tex(agg: dict[tuple[str, str], dict]) -> str:
    """Appendix table — same cells plus n, 95% CI, and C->W rate per bucket."""
    lines = [
        "% generated by analysis_4models/v4/tables/make_attention_paired_audit.py",
        "% appendix: mean, 95% bootstrap CI, n, and Clean->Wrong rate per bucket",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{ll" + "c" * 5 + r"}",
        r"\toprule",
        (r"\textbf{Class} & \textbf{Hop} & "
         r"\textbf{$n$} & \textbf{mean} & "
         r"\textbf{95\% CI low} & \textbf{95\% CI high} & "
         r"\textbf{C$\to$W} \\"),
        r"\midrule",
    ]
    for cls in CLASS_ORDER:
        for hop in HOPS_ORDER:
            d = agg[(cls, hop)]
            cw = f"{d['cw_rate']*100:.1f}\\%" if d["cw_rate"] is not None else "—"
            lines.append(
                f"{cls} & {hop} & {d['n']} & "
                f"{fmt(d['mean'])} & {fmt(d['ci_lo'])} & {fmt(d['ci_hi'])} & "
                f"{cw} \\\\"
            )
        if cls != CLASS_ORDER[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        (r"\caption{\label{tab:attention_lift_by_hop_appendix} \textbf{Per-bucket "
         r"statistics for the Qwen3.5-9B paired attention audit.} "
         r"$n$ is the number of (source, neighbor) pairs in the bucket "
         r"(soft-cap 2 per source, hard-cap 3; 95\% CI from $10{,}000$ "
         r"bootstrap resamples, seed 42). C$\to$W is the fraction of "
         r"clean-correct rows whose post-update prediction flipped wrong; "
         r"identical to the flip-rate reported in the main analysis.}"),
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# --------------------------- emit figure ------------------------------------

def emit_figure(agg: dict[tuple[str, str], dict], png_path: Path, pdf_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hops = HOPS_ORDER
    x = list(range(len(hops)))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = {"Popular": "#c0392b", "Rare": "#2980b9"}
    markers = {"Popular": "o", "Rare": "s"}

    for cls in CLASS_ORDER:
        ys, lo_err, hi_err, ns = [], [], [], []
        for h in hops:
            d = agg[(cls, h)]
            ys.append(d["mean"])
            if d["mean"] is None:
                lo_err.append(0.0); hi_err.append(0.0)
            else:
                lo_err.append(d["mean"] - (d["ci_lo"] if d["ci_lo"] is not None else d["mean"]))
                hi_err.append((d["ci_hi"] if d["ci_hi"] is not None else d["mean"]) - d["mean"])
            ns.append(d["n"])
        # mask Nones
        valid = [i for i, y in enumerate(ys) if y is not None]
        if not valid:
            continue
        xs_v = [x[i] for i in valid]
        ys_v = [ys[i] for i in valid]
        lo_v = [lo_err[i] for i in valid]
        hi_v = [hi_err[i] for i in valid]
        ax.errorbar(
            xs_v, ys_v, yerr=[lo_v, hi_v],
            label=cls, color=colors[cls], marker=markers[cls],
            markersize=6, linewidth=1.6, capsize=3,
        )
        # n annotations
        for xi, yi, ni in zip(xs_v, ys_v, [ns[i] for i in valid]):
            ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=colors[cls])

    ax.set_xticks(x)
    ax.set_xticklabels(hops)
    ax.set_xlabel("Hop distance from updated source")
    ax.set_ylabel(r"mean $|\Delta\,\mathrm{AttLift}|$")
    ax.set_title("Attention Perturbation after Popular vs Rare Source Updates")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


# --------------------------- main -------------------------------------------

def main() -> None:
    if not AUDIT_CSV.exists():
        print(f"ERROR: audit CSV not found at {AUDIT_CSV}", file=sys.stderr)
        print("Run scripts/attention_paired_audit.py first.", file=sys.stderr)
        sys.exit(2)

    rows = load_rows(AUDIT_CSV)
    print(f"[load] {len(rows)} rows from {AUDIT_CSV}")

    agg = aggregate(rows, PRIMARY_METRIC)

    # Print summary to stdout for quick eyeball check
    print(f"\n[summary] metric = delta_att_lift_abs_{PRIMARY_METRIC}")
    print("class    hop   n   mean      CI_lo     CI_hi     C->W")
    for cls in CLASS_ORDER:
        for hop in HOPS_ORDER:
            d = agg[(cls, hop)]
            cw = f"{d['cw_rate']*100:5.1f}%" if d["cw_rate"] is not None else "  —  "
            print(f"{cls:8s} {hop}  {d['n']:3d}  "
                  f"{fmt(d['mean'])}  {fmt(d['ci_lo'])}  {fmt(d['ci_hi'])}  {cw}")

    # 1. main paper .tex
    main_tex = emit_main_tex(agg)
    write_paired(
        OUT_TABLES / "attention_lift_by_hop.tex",
        PAPER_TABLES / "attention_lift_by_hop.tex",
        main_tex,
    )

    # 2. appendix .tex
    app_tex = emit_appendix_tex(agg)
    write_paired(
        OUT_TABLES / "attention_lift_by_hop_appendix.tex",
        PAPER_TABLES / "attention_lift_by_hop_appendix.tex",
        app_tex,
    )

    # 3. figure (png + pdf), mirrored to paper figures
    png_local = OUT_FIGS / "attention_lift_by_hop.png"
    pdf_local = OUT_FIGS / "attention_lift_by_hop.pdf"
    emit_figure(agg, png_local, pdf_local)
    print(f"[fig] wrote {png_local}")
    print(f"[fig] wrote {pdf_local}")
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    (PAPER_FIGS / "attention_lift_by_hop.png").write_bytes(png_local.read_bytes())
    (PAPER_FIGS / "attention_lift_by_hop.pdf").write_bytes(pdf_local.read_bytes())
    print(f"[fig]  -> {PAPER_FIGS / 'attention_lift_by_hop.png'}")
    print(f"[fig]  -> {PAPER_FIGS / 'attention_lift_by_hop.pdf'}")


if __name__ == "__main__":
    main()
