"""Thin CLI wrapper around GraphAPI — handy for shell/CI use.

Usage (inside the container):
  python /app/cli.py query    "Brooklyn"
  python /app/cli.py query    "Q30"
  python /app/cli.py search   "jay-z"          [--limit 20]
  python /app/cli.py top      [--n 20] [--by qid|node]
  python /app/cli.py info     "United States"
  python /app/cli.py stats
  python /app/cli.py report   [--dataset DATA.jsonl] [--out DIR] [--top-n 5]

All read-only commands print JSON to stdout (one object/array per call).
`report` writes report.md / summary.json / segmented.jsonl into --out and
prints the artifact paths to stderr.

Override default paths with --graph / --qid-index, or set env vars
GENFRAG_GRAPH_PATH / GENFRAG_QID_INDEX_PATH.
"""

from __future__ import annotations

import json
import os

import click

from graph_api import GraphAPI


DEFAULT_GRAPH = os.environ.get("GENFRAG_GRAPH_PATH", "/app/data/final.pkl")
DEFAULT_QID = os.environ.get(
    "GENFRAG_QID_INDEX_PATH", "/app/data/graph_qid_index.json"
)


def _emit(obj) -> None:
    click.echo(json.dumps(obj, indent=2, ensure_ascii=False))


@click.group()
@click.option("--graph", default=DEFAULT_GRAPH, show_default=True,
              help="Path to final.pkl")
@click.option("--qid-index", default=DEFAULT_QID, show_default=True,
              help="Path to graph_qid_index.json")
@click.pass_context
def cli(ctx, graph, qid_index):
    """GenFragility graph CLI — popularity & in-degree lookups."""
    ctx.ensure_object(dict)
    ctx.obj["api"] = GraphAPI(graph, qid_index)


@cli.command()
@click.argument("key")
@click.pass_context
def query(ctx, key):
    """Look up popularity + in-degree for KEY (node name or QID)."""
    api: GraphAPI = ctx.obj["api"]
    indeg = api.get_in_degree(key)
    pop = api.get_popularity(key)
    _emit({
        "key": key,
        "resolved_node": indeg["resolved_node"],
        "qid": indeg["qid"],
        "source": indeg["source"],
        "in_degree": indeg["in_degree"],
        "popularity": pop["popularity"],
        "popularity_metric": pop["metric"],
        "aliases": pop["aliases"],
    })


@cli.command()
@click.argument("substr")
@click.option("--limit", default=20, show_default=True)
@click.pass_context
def search(ctx, substr, limit):
    """Case-insensitive substring search over node names."""
    api: GraphAPI = ctx.obj["api"]
    _emit(api.search(substr, limit=limit))


@cli.command()
@click.option("--n", default=20, show_default=True)
@click.option("--by", type=click.Choice(["qid", "node"]), default="qid",
              show_default=True)
@click.pass_context
def top(ctx, n, by):
    """Top-N entities by popularity."""
    api: GraphAPI = ctx.obj["api"]
    _emit(api.top_hubs(n=n, by=by))


@cli.command()
@click.argument("key")
@click.pass_context
def info(ctx, key):
    """Detailed view of one node (degrees + outgoing relation histogram)."""
    api: GraphAPI = ctx.obj["api"]
    _emit(api.node_info(key))


@cli.command()
@click.pass_context
def stats(ctx):
    """Summary stats for the loaded graph."""
    api: GraphAPI = ctx.obj["api"]
    _emit(api.stats())


@cli.command()
@click.option("--dataset", default="/app/demo_dataset.jsonl", show_default=True,
              help="Path to input JSONL. Falls back to bundled demo if omitted.")
@click.option("--out", "out_dir", default="./report_out", show_default=True,
              help="Directory to write report.md / summary.json / segmented.jsonl.")
@click.option("--top-n", default=5, show_default=True,
              help="Top-N hubs to list in the report's Graph Overview.")
@click.pass_context
def report(ctx, dataset, out_dir, top_n):
    """Segment DATASET against the graph and write a popularity report."""
    # Imported lazily so `query`/`search`/etc. don't pay the report.py import.
    from report import build_report
    api: GraphAPI = ctx.obj["api"]
    paths = build_report(api, dataset, out_dir, top_n=top_n)
    _emit(paths)


if __name__ == "__main__":
    cli(obj={})
