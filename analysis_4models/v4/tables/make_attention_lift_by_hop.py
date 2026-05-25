"""make_attention_lift_by_hop.py

Regenerate tables/attention_lift_by_hop.tex.

Source-of-truth values:
    Llama-2 paired audit (n=30 evaluation samples per hop, hub-source 006
    vs tail-source 007). The raw attention-dump JSONs live in:

        scripts/external_eval/llama_attention_paired/
            attn_lift_006_d1.json … attn_lift_006_d5.json   (Hub source)
            attn_lift_007_d1.json … attn_lift_007_d5.json   (Tail source)

If those JSONs are NOT present (mechanistic-deep-dive artifacts not always
shipped with the repo), this script falls back to the canonical numbers
already in the paper. We mark whether the table came from raw recomputation
or canonical pinned values via a comment in the .tex preamble.

Output: v4/outputs/tables/attention_lift_by_hop.tex
        + mirror to <paper>/tables/attention_lift_by_hop.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.loader import REPO_ROOT, OUT_TABLES, PAPER_TABLES  # type: ignore
from lib.latex import write_paired


# Canonical pinned values from the existing paper table.
CANONICAL = {
    "d1": (0.07246, 0.06912),
    "d2": (0.05836, 0.04996),
    "d3": (0.05970, 0.04318),
}

DUMP_DIR = REPO_ROOT / "scripts/external_eval/llama_attention_paired"


def try_load_raw() -> dict[str, tuple[float, float]] | None:
    if not DUMP_DIR.exists():
        return None
    out = {}
    for h in ("d1", "d2", "d3", "d4", "d5"):
        fh = DUMP_DIR / f"attn_lift_006_{h}.json"
        ft = DUMP_DIR / f"attn_lift_007_{h}.json"
        if not (fh.exists() and ft.exists()):
            return None
        try:
            h_val = json.loads(fh.read_text())["delta_attlift_abs_mean"]
            t_val = json.loads(ft.read_text())["delta_attlift_abs_mean"]
        except Exception:
            return None
        out[h] = (float(h_val), float(t_val))
    return out


def main() -> None:
    raw = try_load_raw()
    if raw is not None:
        data = raw
        provenance = "% recomputed from llama_attention_paired/attn_lift_*.json"
    else:
        data = CANONICAL
        provenance = ("% canonical pinned values (Llama-2 paired audit, "
                      "n=30 per hop, last-layer first-step head-token span)")

    hops = sorted(data.keys())

    lines = [
        provenance,
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Hop} & \textbf{Hub $|\Delta \mathrm{AttLift}|$} & "
        r"\textbf{Tail $|\Delta \mathrm{AttLift}|$} & "
        r"\textbf{Hub $>$ Tail?} \\",
        r"\midrule",
    ]
    for h in hops:
        hv, tv = data[h]
        cmp_ = "Yes" if hv > tv else "No"
        lines.append(f"{h} & {hv:.5f} & {tv:.5f} & {cmp_} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\label{tab:attention_lift_by_hop} \textbf{Attention Lift "
        r"Perturbation by Hop in the Llama-2 Paired Audit.} Absolute post-update "
        r"attention-lift change ($|\Delta \mathrm{AttLift}|$) for the hub-sourced "
        r"update (006) and the tail-sourced update (007), measured on the paired "
        r"audit with $n=30$ evaluation samples per hop. The statistic is computed "
        r"from the final transformer layer at the first generated token step, "
        r"averaged over all heads and query positions, and evaluated on the matched "
        r"neighbor head-token span. Hubs show larger perturbations at $d1$--$d3$.}",
        r"\end{table}",
        "",
    ]
    content = "\n".join(lines)

    write_paired(
        OUT_TABLES / "attention_lift_by_hop.tex",
        PAPER_TABLES / "attention_lift_by_hop.tex",
        content,
    )


if __name__ == "__main__":
    main()
