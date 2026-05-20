"""
Build a graph_node -> Wikidata QID index for the full 100k graph.

Strategy
--------
1. Filter out literal nodes (dates, pure numbers) — ~11% of nodes — they will
   never have a QID by definition.
2. For the remaining ~89k entity-like nodes, batch-query the Wikipedia
   MediaWiki API (50 titles per request) for the wikibase_item pageprop.
   This returns the QID and follows redirects/normalization (e.g. USA→US).
3. Persist to data/external_eval/graph_qid_index.json:
     { qid -> graph_node }  and  { graph_node -> qid }
4. Also dump unresolved list so we can choose to enrich via DeepSeek API
   in a second pass.

Performance
-----------
~89k nodes / 50 per batch / 0.2s sleep ≈ 360s wall clock (~6 min).

Usage
-----
    python scripts/external_eval/build_graph_qid_index.py [--dry-run] [--limit N]
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
OUT_DIR = ROOT / "data/external_eval"

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "GenFragility-LLM-pilot/0.1 (research; contact: wuhubing19@gmail.com)"

DATE_NUM_RE = re.compile(r"^-?\d{1,4}(-\d{1,2}){0,2}$")
JUNK_TOKENS = {"None", "none", "Null", "N/A", "TBD", "Unknown", "unknown", ""}


def is_literal(node: str) -> bool:
    if node in JUNK_TOKENS:
        return True
    if DATE_NUM_RE.match(node):
        return True
    return False


def batch_resolve(titles, sess, sleep=0.2):
    params = {
        "action": "query", "format": "json", "prop": "pageprops",
        "titles": "|".join(titles), "redirects": 1, "ppprop": "wikibase_item",
    }
    r = sess.get(WIKI_API, params=params, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    data = r.json()
    normalized = {n["from"]: n["to"] for n in data.get("query", {}).get("normalized", [])}
    redirects = {r["from"]: r["to"] for r in data.get("query", {}).get("redirects", [])}
    title_to_qid: dict[str, str | None] = {}
    for page in data.get("query", {}).get("pages", {}).values():
        t = page.get("title")
        qid = page.get("pageprops", {}).get("wikibase_item")
        if t and qid:
            title_to_qid[t] = qid
    out: dict[str, str | None] = {}
    for orig in titles:
        canonical = redirects.get(normalized.get(orig, orig), normalized.get(orig, orig))
        out[orig] = title_to_qid.get(canonical)
    time.sleep(sleep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats and ETA, do not query API.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N entity-like nodes.")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading graph...")
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    all_nodes = list(G.nodes())
    entity_nodes = [n for n in all_nodes if not is_literal(n)]
    literals = len(all_nodes) - len(entity_nodes)
    print(f"  total nodes      : {len(all_nodes):,}")
    print(f"  literals (skipped): {literals:,} ({literals/len(all_nodes)*100:.1f}%)")
    print(f"  entity-like nodes: {len(entity_nodes):,}")

    if args.limit:
        entity_nodes = entity_nodes[: args.limit]
        print(f"  --limit applied  : {len(entity_nodes):,}")

    n_batches = (len(entity_nodes) + args.batch - 1) // args.batch
    eta_sec = n_batches * (args.sleep + 0.4)  # ~0.4s avg request
    print(f"  batches          : {n_batches:,}  (size={args.batch}, sleep={args.sleep}s)")
    print(f"  estimated wall   : {eta_sec/60:.1f} min")

    if args.dry_run:
        print("\n[DRY RUN] No API calls. Re-run without --dry-run to execute.")
        return

    sess = requests.Session()
    name_to_qid: dict[str, str] = {}
    unresolved: list[str] = []
    t0 = time.time()
    for i in range(0, len(entity_nodes), args.batch):
        chunk = entity_nodes[i: i + args.batch]
        try:
            res = batch_resolve(chunk, sess, args.sleep)
        except Exception as e:
            print(f"[warn] batch {i//args.batch} failed: {e}")
            for n in chunk:
                unresolved.append(n)
            continue
        for n, q in res.items():
            if q:
                name_to_qid[n] = q
            else:
                unresolved.append(n)
        if (i // args.batch) % 50 == 0:
            elapsed = time.time() - t0
            done = i + len(chunk)
            print(f"  [{done:,}/{len(entity_nodes):,}] "
                  f"resolved={len(name_to_qid):,} "
                  f"elapsed={elapsed:.0f}s")

    qid_to_name = {v: k for k, v in name_to_qid.items()}
    (OUT_DIR / "graph_qid_index.json").write_text(
        json.dumps({"name_to_qid": name_to_qid,
                    "qid_to_name": qid_to_name}, ensure_ascii=False)
    )
    (OUT_DIR / "graph_unresolved.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2)
    )

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  resolved : {len(name_to_qid):,} / {len(entity_nodes):,} "
          f"({len(name_to_qid)/len(entity_nodes)*100:.1f}%)")
    print(f"  unresolved: {len(unresolved):,}")
    print(f"  index    : {OUT_DIR/'graph_qid_index.json'}")


if __name__ == "__main__":
    main()
