"""
Enrich final.pkl: write Wikidata QID as a node attribute for every entity-like node.

Adds two attributes to each node:
  G.nodes[n]['qid']        : 'Q30' | None
  G.nodes[n]['qid_status'] : 'resolved' | 'unresolved' | 'literal'

Also dumps a sidecar fast-lookup index to
    data/external_eval/graph_qid_index.json
so downstream scripts can avoid loading the 120MB pkl just to look up names→QIDs.

Behavior:
  - Skips literal nodes (dates, pure numbers, junk tokens) — never queries them.
  - Batches Wikipedia API at 50 titles/call, follows redirects (USA→US).
  - Incremental: if a node already has G.nodes[n]['qid_status'] set, it is skipped
    (so re-runs are cheap).
  - Writes to a NEW file results/checkpoints/final_qid.pkl by default; pass --in-place
    to overwrite final.pkl directly.

Run:
  python scripts/external_eval/enrich_graph_with_qid.py
  python scripts/external_eval/enrich_graph_with_qid.py --in-place
"""
from __future__ import annotations
import argparse
import json
import pickle
import re
import time
from pathlib import Path
import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
DEFAULT_OUT_PATH = ROOT / "results/checkpoints/final_qid.pkl"
SIDECAR_DIR = ROOT / "data/external_eval"

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "GenFragility-LLM/0.1 (research; contact: wuhubing19@gmail.com)"

DATE_NUM_RE = re.compile(r"^-?\d{1,4}(-\d{1,2}){0,2}$")
JUNK_TOKENS = {"None", "none", "Null", "null", "N/A", "n/a", "TBD",
               "tbd", "Unknown", "unknown", ""}


def is_literal(node: str) -> bool:
    if not isinstance(node, str):
        return True
    if node in JUNK_TOKENS:
        return True
    if DATE_NUM_RE.match(node):
        return True
    return False


def batch_resolve(titles, sess, sleep=0.2, retries=2):
    """Returns {title: qid_or_None} for a batch."""
    params = {
        "action": "query", "format": "json", "prop": "pageprops",
        "titles": "|".join(titles), "redirects": 1, "ppprop": "wikibase_item",
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = sess.get(WIKI_API, params=params,
                         headers={"User-Agent": UA}, timeout=20)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    else:
        raise RuntimeError(f"Wikipedia batch failed after retries: {last_err}")

    normalized = {n["from"]: n["to"]
                  for n in data.get("query", {}).get("normalized", [])}
    redirects = {r["from"]: r["to"]
                 for r in data.get("query", {}).get("redirects", [])}
    title_to_qid: dict[str, str | None] = {}
    for page in data.get("query", {}).get("pages", {}).values():
        t = page.get("title")
        qid = page.get("pageprops", {}).get("wikibase_item")
        if t and qid:
            title_to_qid[t] = qid
    out: dict[str, str | None] = {}
    for orig in titles:
        canonical = redirects.get(
            normalized.get(orig, orig), normalized.get(orig, orig))
        out[orig] = title_to_qid.get(canonical)
    time.sleep(sleep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite final.pkl directly. Otherwise writes final_qid.pkl.")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None,
                    help="(debug) only process first N entity nodes")
    ap.add_argument("--report-every", type=int, default=50,
                    help="Print progress every N batches")
    args = ap.parse_args()

    out_path = GRAPH_PATH if args.in_place else DEFAULT_OUT_PATH
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading graph from {GRAPH_PATH} ...")
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    print(f"      Loaded: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # Step 1: classify nodes
    print("[2/5] Classifying nodes (literal vs entity)...")
    literals = []
    entities = []
    already_done = []
    for n in G.nodes():
        if "qid_status" in G.nodes[n]:        # incremental: skip done
            already_done.append(n)
            continue
        if is_literal(n):
            G.nodes[n]["qid"] = None
            G.nodes[n]["qid_status"] = "literal"
            literals.append(n)
        else:
            entities.append(n)
    print(f"      already-done : {len(already_done):,}")
    print(f"      literals     : {len(literals):,}")
    print(f"      to-resolve   : {len(entities):,}")

    if args.limit:
        entities = entities[: args.limit]
        print(f"      --limit applied: only {len(entities):,} entities will be queried")

    # Step 2: API resolution
    n_batches = (len(entities) + args.batch - 1) // args.batch
    print(f"[3/5] Resolving via Wikipedia API "
          f"({n_batches:,} batches of {args.batch}, sleep={args.sleep}s)")
    print(f"      ETA: ~{n_batches * (args.sleep + 0.35) / 60:.1f} min")

    sess = requests.Session()
    resolved = 0
    unresolved = 0
    t0 = time.time()
    for i in range(0, len(entities), args.batch):
        chunk = entities[i: i + args.batch]
        try:
            res = batch_resolve(chunk, sess, args.sleep)
        except Exception as e:
            print(f"      [warn] batch {i//args.batch} failed: {e}, marking unresolved")
            for n in chunk:
                G.nodes[n]["qid"] = None
                G.nodes[n]["qid_status"] = "unresolved"
                unresolved += 1
            continue
        for n, qid in res.items():
            if qid:
                G.nodes[n]["qid"] = qid
                G.nodes[n]["qid_status"] = "resolved"
                resolved += 1
            else:
                G.nodes[n]["qid"] = None
                G.nodes[n]["qid_status"] = "unresolved"
                unresolved += 1
        b_done = i // args.batch + 1
        if b_done % args.report_every == 0 or b_done == n_batches:
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(elapsed, 1e-3)
            eta_sec = (len(entities) - (i + len(chunk))) / max(rate, 1e-3)
            print(f"      [{i+len(chunk):,}/{len(entities):,}]  "
                  f"resolved={resolved:,}  miss={unresolved:,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_sec:.0f}s")
    api_time = time.time() - t0
    print(f"      API done in {api_time:.0f}s "
          f"({resolved:,} resolved / {unresolved:,} unresolved)")

    # Step 3: write enriched graph
    print(f"[4/5] Saving enriched graph to {out_path} ...")
    if isinstance(data, dict):
        data["graph"] = G
        payload = data
    else:
        payload = G
    # Pickle to a tmp file then rename so an interruption never corrupts the target.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(out_path)
    print(f"      Wrote: {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")

    # Step 4: sidecar index for fast lookup
    print(f"[5/5] Writing sidecar QID index ...")
    name_to_qid = {}
    qid_to_name = {}
    status_counts = {"resolved": 0, "unresolved": 0, "literal": 0}
    for n in G.nodes():
        st = G.nodes[n].get("qid_status", "unresolved")
        status_counts[st] = status_counts.get(st, 0) + 1
        q = G.nodes[n].get("qid")
        if q:
            name_to_qid[n] = q
            # If multiple nodes share a QID (shouldn't, but guard), keep first
            qid_to_name.setdefault(q, n)

    sidecar = SIDECAR_DIR / "graph_qid_index.json"
    sidecar.write_text(json.dumps({
        "meta": {
            "source_graph": str(GRAPH_PATH),
            "enriched_graph": str(out_path),
            "total_nodes": G.number_of_nodes(),
            "status_counts": status_counts,
            "resolved_rate": round(
                status_counts["resolved"] / G.number_of_nodes(), 4),
        },
        "name_to_qid": name_to_qid,
        "qid_to_name": qid_to_name,
    }, ensure_ascii=False))
    print(f"      Wrote: {sidecar} "
          f"({sidecar.stat().st_size/1024/1024:.1f} MB)")

    print("\n=== Summary ===")
    print(f"  Total nodes  : {G.number_of_nodes():,}")
    print(f"  Resolved     : {status_counts['resolved']:,}  "
          f"({status_counts['resolved']/G.number_of_nodes()*100:.1f}%)")
    print(f"  Unresolved   : {status_counts['unresolved']:,}  "
          f"({status_counts['unresolved']/G.number_of_nodes()*100:.1f}%)")
    print(f"  Literal      : {status_counts['literal']:,}  "
          f"({status_counts['literal']/G.number_of_nodes()*100:.1f}%)")
    print(f"  Wall time    : {time.time()-t0:.0f}s")
    print(f"\n  Enriched graph -> {out_path}")
    print(f"  Sidecar index  -> {sidecar}")


if __name__ == "__main__":
    main()
