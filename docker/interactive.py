"""Interactive REPL entrypoint for the GenFragility graph container.

Pre-loads final.pkl + the QID side-index into a GraphAPI instance, then drops
the user into an IPython shell with two pre-bound names:

  api  : GraphAPI    (helpers: get_in_degree / get_popularity / search / ...)
  G    : nx.DiGraph  (the raw graph, for nx.pagerank / nx.shortest_path / ...)

Override default paths via env vars GENFRAG_GRAPH_PATH / GENFRAG_QID_INDEX_PATH.
"""

from __future__ import annotations

import os
import sys

from graph_api import GraphAPI


GRAPH_PATH = os.environ.get("GENFRAG_GRAPH_PATH", "/app/data/final.pkl")
QID_INDEX_PATH = os.environ.get(
    "GENFRAG_QID_INDEX_PATH", "/app/data/graph_qid_index.json"
)


def main() -> None:
    print(f"[interactive] loading graph from {GRAPH_PATH} ...", file=sys.stderr)
    api = GraphAPI(GRAPH_PATH, QID_INDEX_PATH)
    G = api.graph  # noqa: F841 — exposed in user namespace
    print(api.banner())

    try:
        from IPython import embed
    except ImportError:
        print(
            "[interactive] IPython not installed; falling back to plain Python REPL.",
            file=sys.stderr,
        )
        import code

        code.interact(local={"api": api, "G": G})
        return

    embed(
        colors="neutral",
        header="(GenFragility graph REPL — `api` and `G` are pre-loaded)",
        user_ns={"api": api, "G": G},
    )


if __name__ == "__main__":
    main()
