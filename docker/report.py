"""Report orchestrator — load a user JSONL dataset, segment its subjects (and
optional answers) against the GenFragility 100k graph's popularity proxy,
and emit a Markdown report + JSON summary + per-row segmented JSONL.

Public entrypoint: ``build_report(api, dataset_path, out_dir, top_n=5)``.

Popularity = QID-aggregated in-degree (paper §3.5). Bucket thresholds are
copied from ``scripts/external_eval/link_public_datasets.py`` so the output
is directly comparable to the bundled examples (e.g. trivia_bucketed.jsonl):

    hub  : in-degree >= 500
    mid  : in-degree >= 20
    tail : in-degree <  20
    unlinkable : subject did not resolve to any graph node
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from graph_api import GraphAPI


# Bucket cutoffs — mirror link_public_datasets.py::bucketize.
HUB_MIN = 500
MID_MIN = 20

# Path to the embedded paper-claims blob. Tests can override.
DEFAULT_CLAIMS = Path(__file__).resolve().parent / "paper_claims.md"


# ----------------------------------------------------------------------
# 1. Dataset loader
# ----------------------------------------------------------------------
def iter_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream a JSONL dataset, skipping blank lines.

    Each yielded dict carries the original ``id`` (or an auto-assigned one)
    so the report can be cross-referenced with the input.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{p}: invalid JSON on line {i}: {e.msg}"
                ) from None
            row.setdefault("id", f"row_{i:06d}")
            yield row


# ----------------------------------------------------------------------
# 2. Per-row resolution + bucketing
# ----------------------------------------------------------------------
def bucketize(in_degree: int | None) -> str:
    if in_degree is None:
        return "unlinkable"
    if in_degree >= HUB_MIN:
        return "hub"
    if in_degree >= MID_MIN:
        return "mid"
    return "tail"


def _resolve_side(api: GraphAPI, qid: str | None, text: str | None) -> dict[str, Any]:
    """Resolve one side (subject or answer) of a row.

    Tries QID first, then exact text, then case-insensitive text via
    ``GraphAPI.resolve``. Returns the node, the (possibly inferred) QID,
    the per-node in-degree, the popularity (QID-agg in-degree), and the
    bucket — using *node-level* in-degree as the bucketing input since the
    paper defines popularity on the object entity.
    """
    node = qid_out = None
    source = "miss"

    if qid:
        r = api.resolve(qid)
        if r["node"] is not None:
            node, qid_out, source = r["node"], r["qid"], "qid"

    if node is None and text:
        if text in api.graph:
            node, source = text, "text_exact"
            qid_out = api.name_to_qid.get(text)
        else:
            r = api.resolve(text)  # falls through to case-insensitive
            if r["node"] is not None:
                node = r["node"]
                qid_out = r["qid"]
                source = r["source"]  # exact_node / case_insensitive_node

    in_degree = api._in_degree_by_node.get(node) if node else None
    popularity = (
        api._in_degree_by_qid.get(qid_out)
        if qid_out is not None
        else in_degree
    )
    return {
        "qid_input": qid,
        "text_input": text,
        "resolved_node": node,
        "qid_resolved": qid_out,
        "resolution_source": source,
        "in_degree": in_degree,
        "popularity": popularity,
        "bucket": bucketize(in_degree),
    }


def segment(api: GraphAPI, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich every row with subject + optional answer resolution.

    Output schema is intentionally close to ``trivia_bucketed.jsonl`` so
    existing downstream scripts can consume the file unchanged.
    """
    out = []
    for row in rows:
        subj = _resolve_side(
            api, row.get("subject_qid"), row.get("subject_text")
        )
        ans = _resolve_side(
            api, row.get("answer_qid"), row.get("answer_text")
        )
        linkable = subj["resolved_node"] is not None
        out.append({
            "id": row.get("id"),
            "question": row.get("question"),
            "subject_qid": subj["qid_resolved"],
            "subject_text": row.get("subject_text"),
            "subject_node": subj["resolved_node"],
            "subject_in_degree": subj["in_degree"],
            "subject_popularity": subj["popularity"],
            "subject_resolution": subj["resolution_source"],
            "bucket": subj["bucket"],
            "linkable": linkable,
            "answer_qid": ans["qid_resolved"],
            "answer_text": row.get("answer_text"),
            "answer_node": ans["resolved_node"],
            "answer_in_degree": ans["in_degree"],
            "answer_popularity": ans["popularity"],
        })
    return out


# ----------------------------------------------------------------------
# 3. Statistics
# ----------------------------------------------------------------------
def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _log10_histogram(values: list[int], n_bins: int = 8) -> list[dict[str, Any]]:
    """Bin values on a log10(x+1) axis. Returns [{lo, hi, count, label}]."""
    if not values:
        return []
    log_vals = [math.log10(v + 1) for v in values]
    lo, hi = min(log_vals), max(log_vals)
    if lo == hi:
        # Single distinct value — return a degenerate bin so the report
        # still renders something useful.
        return [{
            "lo_log10": lo, "hi_log10": hi,
            "lo": int(10 ** lo - 1), "hi": int(10 ** hi - 1),
            "count": len(values),
            "label": f"={int(round(10 ** lo - 1))}",
        }]
    step = (hi - lo) / n_bins
    bins = []
    for i in range(n_bins):
        b_lo = lo + i * step
        b_hi = lo + (i + 1) * step
        # Include right edge in the last bin.
        if i == n_bins - 1:
            count = sum(1 for v in log_vals if b_lo <= v <= b_hi)
        else:
            count = sum(1 for v in log_vals if b_lo <= v < b_hi)
        bins.append({
            "lo_log10": b_lo, "hi_log10": b_hi,
            "lo": int(round(10 ** b_lo - 1)),
            "hi": int(round(10 ** b_hi - 1)),
            "count": count,
            "label": f"[{int(round(10 ** b_lo - 1)):>5}, {int(round(10 ** b_hi - 1)):>5}]",
        })
    return bins


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    linkable_rows = [r for r in rows if r["linkable"]]
    pops = [r["subject_popularity"] for r in linkable_rows
            if r["subject_popularity"] is not None]

    bucket_counts = {"hub": 0, "mid": 0, "tail": 0, "unlinkable": 0}
    for r in rows:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    quantiles = {
        "min": min(pops) if pops else None,
        "p25": _quantile(pops, 0.25) if pops else None,
        "median": _quantile(pops, 0.50) if pops else None,
        "p75": _quantile(pops, 0.75) if pops else None,
        "max": max(pops) if pops else None,
        "mean": (sum(pops) / len(pops)) if pops else None,
    }

    return {
        "n_rows": n,
        "n_linkable": len(linkable_rows),
        "linkable_rate": (len(linkable_rows) / n) if n else 0.0,
        "bucket_counts": bucket_counts,
        "popularity_quantiles": quantiles,
        "popularity_histogram": _log10_histogram(pops),
    }


# ----------------------------------------------------------------------
# 4. Markdown rendering
# ----------------------------------------------------------------------
def _ascii_bar(count: int, max_count: int, width: int = 40) -> str:
    if max_count <= 0:
        return ""
    filled = int(round(width * count / max_count))
    return "█" * filled + "·" * (width - filled)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_md(api: GraphAPI, summary: dict[str, Any], top_hubs: list[dict[str, Any]],
              dataset_path: str, claims_text: str) -> str:
    stats = api.stats()
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Section 1: graph overview + top-N hubs ---
    hub_rows = [
        [str(i + 1), h["qid"], h["canonical"] or "—",
         f"{h['popularity']:,}",
         ", ".join(h["aliases"][:3]) + (
             f", … (+{len(h['aliases']) - 3})" if len(h["aliases"]) > 3 else ""
         )]
        for i, h in enumerate(top_hubs)
    ]
    sec1 = (
        "## 1. Graph Overview\n\n"
        f"- Nodes: **{stats['num_nodes']:,}**, Edges: **{stats['num_edges']:,}**\n"
        f"- QID coverage: **{stats['num_nodes_with_qid']:,} / "
        f"{stats['num_nodes']:,}** "
        f"({100.0 * stats['num_nodes_with_qid'] / stats['num_nodes']:.1f}%)\n"
        f"- Unique QIDs after alias merging: **{stats['num_unique_qids']:,}**\n\n"
        f"### Top-{len(top_hubs)} most popular entities (QID-aggregated in-degree)\n\n"
        + _md_table(
            ["#", "QID", "Canonical name", "Popularity", "Aliases"],
            hub_rows,
        )
        + "\n"
    )

    # --- Section 2: short popularity-meaning paragraph ---
    sec2 = (
        "## 2. What “popularity” means in this report\n\n"
        "Popularity is the **in-degree of the entity** in the verified factual\n"
        "graph `G_fact`, after collapsing alias node-names (e.g. `USA`,\n"
        "`United States`, `U.S.A.`) onto a single Wikidata QID (`Q30`). This is\n"
        "the metric the paper uses throughout (see §3 *Factual Connectivity as\n"
        "Popularity*). Buckets are `hub` (≥ 500), `mid` (≥ 20), `tail` (< 20),\n"
        "or `unlinkable` (subject not found in `G_fact`).\n"
    )

    # --- Section 3: dataset summary ---
    bc = summary["bucket_counts"]
    qs = summary["popularity_quantiles"]
    sec3_lines = [
        "## 3. Dataset Summary\n",
        f"- Source file: `{dataset_path}`",
        f"- Generated: {ts}",
        f"- Rows: **{summary['n_rows']:,}**",
        f"- Linkable subjects: **{summary['n_linkable']:,}** "
        f"({summary['linkable_rate'] * 100:.1f}%)",
        "",
        "### Bucket distribution",
        "",
        _md_table(
            ["bucket", "count", "% of all rows"],
            [
                [b, f"{bc.get(b, 0):,}",
                 f"{(bc.get(b, 0) / summary['n_rows'] * 100) if summary['n_rows'] else 0:.1f}%"]
                for b in ("hub", "mid", "tail", "unlinkable")
            ],
        ),
        "",
        "### Popularity quantiles (linkable subjects only)",
        "",
        _md_table(
            ["min", "p25", "median", "mean", "p75", "max"],
            [[
                "—" if qs["min"] is None else f"{int(qs['min']):,}",
                "—" if qs["p25"] is None else f"{qs['p25']:,.1f}",
                "—" if qs["median"] is None else f"{qs['median']:,.1f}",
                "—" if qs["mean"] is None else f"{qs['mean']:,.1f}",
                "—" if qs["p75"] is None else f"{qs['p75']:,.1f}",
                "—" if qs["max"] is None else f"{int(qs['max']):,}",
            ]],
        ),
    ]
    sec3 = "\n".join(sec3_lines) + "\n"

    # --- Section 4: ASCII histogram of popularity ---
    hist = summary["popularity_histogram"]
    if hist:
        max_count = max(b["count"] for b in hist)
        bar_lines = [
            f"`{b['label']}` | {_ascii_bar(b['count'], max_count):<40} | "
            f"{b['count']:>6,}"
            for b in hist
        ]
    else:
        bar_lines = ["_(no linkable rows — nothing to plot)_"]

    sec4 = (
        "## 4. Popularity Distribution (linkable subjects, log10-binned)\n\n"
        "```\n"
        "range of in-degree                   | bar"
        + " " * (40 - len("bar")) + " | count\n"
        + "-" * 78 + "\n"
        + "\n".join(bar_lines)
        + "\n```\n"
    )

    # --- Section 5: paper claims (verbatim) ---
    sec5 = (
        "## 5. Reference: paper claims at a glance\n\n"
        "_Embedded from `paper_claims.md` so the report is self-contained._\n\n"
        + claims_text
    )

    header = (
        "# GenFragility Popularity Report\n\n"
        f"_Generated_: {ts}  ·  _Dataset_: `{dataset_path}`  ·  "
        f"_Graph_: `{api.graph_path.name}`\n\n"
        "---\n"
    )
    return "\n".join([header, sec1, sec2, sec3, sec4, sec5])


# ----------------------------------------------------------------------
# 5. Output writer + public entrypoint
# ----------------------------------------------------------------------
def write_outputs(out_dir: str | Path, md: str,
                  summary: dict[str, Any],
                  segmented_rows: list[dict[str, Any]],
                  top_hubs: list[dict[str, Any]]) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "report.md"
    json_path = out / "summary.json"
    seg_path = out / "segmented.jsonl"

    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"summary": summary, "top_hubs": top_hubs},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with seg_path.open("w", encoding="utf-8") as f:
        for row in segmented_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "report_md": str(md_path),
        "summary_json": str(json_path),
        "segmented_jsonl": str(seg_path),
    }


def build_report(api: GraphAPI, dataset_path: str | Path, out_dir: str | Path,
                 top_n: int = 5,
                 claims_path: str | Path | None = None) -> dict[str, str]:
    """End-to-end: load dataset → segment → summarize → render → write.

    Returns a dict of {artifact_name: absolute_path} for the three files.
    """
    claims_path = Path(claims_path) if claims_path else DEFAULT_CLAIMS
    if not claims_path.exists():
        # Fall back to a stub so missing paper_claims.md does not crash the run.
        claims_text = (
            "_(paper_claims.md was not bundled with this image — see the\n"
            "EMNLP submission for full claim text.)_\n"
        )
    else:
        claims_text = claims_path.read_text(encoding="utf-8")

    rows_raw = list(iter_rows(dataset_path))
    segmented = segment(api, rows_raw)
    summary = summarize(segmented)
    top_hubs = api.top_hubs(top_n, by="qid")

    md = render_md(api, summary, top_hubs, str(dataset_path), claims_text)
    paths = write_outputs(out_dir, md, summary, segmented, top_hubs)

    print(f"[report] wrote {paths['report_md']}", flush=True)
    print(f"[report] wrote {paths['summary_json']}", flush=True)
    print(f"[report] wrote {paths['segmented_jsonl']}", flush=True)
    return paths
