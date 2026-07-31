# GenFragility Graph — Docker (in-degree, popularity, & dataset report)

Self-contained Docker image that bundles the GenFragility 100k knowledge graph
(`results/checkpoints/final.pkl`) together with the QID side-index, and exposes:

- A tiny Python API for **in-degree** and **popularity** queries (REPL & CLI).
- A `report` subcommand that ingests **your own JSONL dataset**, segments it
  against our graph's popularity proxy, and emits a Markdown/JSON report
  grounded in the EMNLP 2026 paper's claims.

> Popularity here is the paper §5.1 metric: **QID-aggregated in-degree** —
> aliases like `USA` and `United States` are first collapsed to a single
> Wikidata QID (`Q30`), then their in-degrees are summed. This matches
> `scripts/external_eval/graph_indegree_vs_external.py::load_graph_indegree_by_qid`.

The image carries **no model weights and no LLM dependencies** — just
networkx, click, and IPython. Final image is ~600 MB, dominated by the pickle
itself (~125 MB) plus the QID index (~4 MB).

---

## 1. Build

Run from the **repo root** (so `COPY results/...` and `COPY data/...` resolve):

```bash
cd /home/weibing_wang/GenFragility-LLM
docker build -f docker/Dockerfile -t genfrag-graph:latest .
```

The repo-root `.dockerignore` already excludes `LLaMA-Factory/`, `main_output/`,
`saves/`, model checkpoints, etc., so the build context stays small.

---

## 2. Three ways to use it

### A. Interactive Python REPL (default)

```bash
docker run -it --rm genfrag-graph:latest
```

Drops you straight into IPython with two pre-loaded names:

| Name  | Type                | Purpose                                              |
|-------|---------------------|------------------------------------------------------|
| `api` | `GraphAPI`          | Helpers: `get_popularity`, `get_in_degree`, `search` |
| `G`   | `networkx.DiGraph`  | Raw graph — use for PageRank, BFS, custom analytics  |

Examples once inside the REPL:

```python
api.get_in_degree("Brooklyn")
# {'key': 'Brooklyn', 'resolved_node': 'Brooklyn', 'qid': 'Q18419', ..., 'in_degree': 42}

api.get_popularity("Q30")
# {'qid': 'Q30', 'popularity': 1873, 'aliases': ['United States', 'USA', 'U.S.'], ...}

api.search("jay-z")             # case-insensitive substring match
api.top_hubs(10)                # top 10 entities by aggregated in-degree
api.node_info("United States")  # in/out degree + outgoing-relation histogram

# Raw graph still available for anything fancier:
import networkx as nx
nx.pagerank(G, alpha=0.85)
```

### B. One-shot CLI (JSON output)

Good for shell scripts, CI, or piping into `jq`:

```bash
docker run --rm genfrag-graph:latest python /app/cli.py query "Brooklyn"
docker run --rm genfrag-graph:latest python /app/cli.py query "Q30"
docker run --rm genfrag-graph:latest python /app/cli.py search "jay-z" --limit 10
docker run --rm genfrag-graph:latest python /app/cli.py top --n 20 --by qid
docker run --rm genfrag-graph:latest python /app/cli.py info "United States"
docker run --rm genfrag-graph:latest python /app/cli.py stats
```

All commands print a single JSON document to stdout.

### C. Dataset → Popularity Report  (`report` subcommand)

Feed in **your own JSONL dataset** and get back a Markdown report + JSON
summary + segmented per-row JSONL — useful for understanding how
hub-heavy / tail-heavy your dataset looks against our 100k graph
*before* you run any model edits.

```bash
# Default: runs against the bundled 20-row demo dataset
docker run --rm -v "$PWD/out":/data/out genfrag-graph:latest \
  python /app/cli.py report --out /data/out

# Mount your own dataset
docker run --rm \
  -v "$PWD/my_dataset.jsonl":/data/in.jsonl \
  -v "$PWD/out":/data/out \
  genfrag-graph:latest \
  python /app/cli.py report --dataset /data/in.jsonl --out /data/out
```

Outputs (three files written into `--out`):

| File              | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `report.md`       | Human-readable report (graph overview, top-5 hubs, dataset bucket breakdown, ASCII popularity histogram, paper-claim reference) |
| `summary.json`    | Every number that appears in the report, machine-readable    |
| `segmented.jsonl` | Your dataset with `subject_node`, `subject_in_degree`, `subject_popularity`, `bucket`, `linkable` (+answer fields) attached — schema-compatible with `data/external_eval/trivia_bucketed.jsonl` |

#### Dataset schema (JSONL — one object per line)

Only `subject_text` *or* `subject_qid` is required. Everything else is
optional and improves resolution or downstream readability.

```jsonc
// minimum (entity-only)
{"id": "row_1", "subject_text": "Brooklyn"}

// preferred: QID-tagged (matches PopQA / EQ / TempLAMA conventions)
{"id": "row_2", "subject_qid": "Q30", "subject_text": "United States"}

// full QA form — both subject and answer get resolved
{"id": "row_3",
 "subject_qid": "Q30", "subject_text": "United States",
 "question":   "What is the capital of the United States?",
 "answer_qid": "Q61",  "answer_text":   "Washington, D.C."}
```

Resolution order per row: `subject_qid` → exact `subject_text` →
case-insensitive `subject_text`. Buckets follow the paper convention:
**hub** ≥ 500 in-degree, **mid** ≥ 20, **tail** < 20, **unlinkable** if no
match. Optional `--top-n` (default 5) controls how many hub rows the
Graph Overview lists.

### D. Jupyter Lab

```bash
docker run -it --rm -p 8888:8888 genfrag-graph:latest \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser \
              --NotebookApp.token='' --NotebookApp.password=''
```

Then open the printed `http://127.0.0.1:8888/lab` URL. Start with
`examples.ipynb` if you mount it in:

```bash
docker run -it --rm -p 8888:8888 \
  -v "$PWD/docker/examples.ipynb:/app/examples.ipynb" \
  genfrag-graph:latest \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser \
              --NotebookApp.token='' --NotebookApp.password=''
```

---

## 3. Query semantics — the three accepted key forms

| Input form                | Example          | How it's resolved                                  |
|---------------------------|------------------|----------------------------------------------------|
| Exact node name           | `"Brooklyn"`     | Direct hit on `G.nodes`                            |
| Wikidata QID              | `"Q30"` / `"q30"`| Looked up in `name_to_qid` reverse map             |
| Case-insensitive name     | `"brooklyn"`     | Falls back to lowercase match over node names      |

Every result carries `source` ∈ `{exact_node, qid, case_insensitive_node, unresolved}`
so you can tell which path matched.

---

## 4. Returned fields — what they mean

```jsonc
// api.get_popularity("Brooklyn")
{
  "key": "Brooklyn",                  // what you asked for, verbatim
  "resolved_node": "Brooklyn",        // which graph node we matched
  "qid": "Q18419",                    // Wikidata QID (may be null)
  "source": "exact_node",             // resolution path used
  "popularity": 87,                   // QID-aggregated in-degree (paper §5.1)
  "aliases": ["Brooklyn"],            // every node-name pointing at this QID
  "metric": "qid_aggregated_in_degree"
}
```

If the key has no QID mapping, `metric` becomes `node_in_degree_fallback` and
`popularity` is just the raw per-node in-degree — useful, but **not** the
metric reported in the paper.

---

## 5. Overriding data paths

Both `interactive.py` and `cli.py` read these env vars:

```bash
GENFRAG_GRAPH_PATH       # default: /app/data/final.pkl
GENFRAG_QID_INDEX_PATH   # default: /app/data/graph_qid_index.json
```

So you can mount a newer graph and point at it without rebuilding:

```bash
docker run -it --rm \
  -v /path/to/newer/final.pkl:/data/final.pkl \
  -e GENFRAG_GRAPH_PATH=/data/final.pkl \
  genfrag-graph:latest
```

---

## 6. What's *not* in this image (by design)

- **No external popularity signals** (QRank, Wikipedia pageviews). They live
  in `data/external_eval/qrank.csv.gz` and `data/external_eval/graph_pageviews_2024_user.json`
  and can be mounted in separately if a downstream pipeline needs them.
- **No torch / transformers / vLLM.** This image is for graph analytics, not
  model probing. If you need both, build the model-side image separately.
- **No re-generation of the graph.** The graph pipeline (DeepSeek API + 31-relation
  ontology) lives in `graph_builder/` and `GRAPH_GENERATION_KNOWLEDGE.md`.
  This image only *consumes* `final.pkl`.

---

## 7. Smoke tests (after build)

```bash
# Sanity: the image loads and reports expected stats.
docker run --rm genfrag-graph:latest python /app/cli.py stats

# Popularity ≥ in_degree for any node with a QID (aggregation only adds).
docker run --rm genfrag-graph:latest python /app/cli.py query "Brooklyn"

# QID-entry works.
docker run --rm genfrag-graph:latest python /app/cli.py query "Q30"

# Top hubs match what scripts/external_eval/graph_indegree_vs_external.py prints.
docker run --rm genfrag-graph:latest python /app/cli.py top --n 5 --by qid
```
