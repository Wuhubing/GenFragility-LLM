"""
Phase 0 Pilot Part B (v2) — RippleEdits → Graph via QID bridge.

Direction reversed: for each RippleEdits sample we resolve its subject_qid
and target_id to their canonical English labels via Wikidata, then check
whether those labels exist as graph nodes (and also try common aliases).

This is the real coverage test that matters for the public-benchmark plan.
"""
from __future__ import annotations
import json
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
RIPPLE_PATH = Path("/tmp/RippleEdits/data/benchmark/popular.json")
OUT = ROOT / "data/external_eval/pilot_partB_v2.json"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
UA = "GenFragility-LLM-pilot/0.1 (research; contact: wuhubing19@gmail.com)"

random.seed(42)


def load_graph_node_set():
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    nodes = set(G.nodes())
    lower = {n.lower(): n for n in nodes}
    return nodes, lower


def batch_qid_to_label(qids, batch_size=50, sleep=0.2):
    """Wikidata wbgetentities batched. Returns {qid: {'label': str, 'aliases': [..]}}."""
    out = {}
    sess = requests.Session()
    headers = {"User-Agent": UA}
    qids = list(qids)
    for i in range(0, len(qids), batch_size):
        chunk = qids[i : i + batch_size]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels|aliases",
            "languages": "en",
            "format": "json",
        }
        try:
            r = sess.get(WIKIDATA_API, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[warn] batch failed: {e}")
            for q in chunk:
                out[q] = None
            continue
        for q, ent in data.get("entities", {}).items():
            label = ent.get("labels", {}).get("en", {}).get("value")
            aliases = [a["value"] for a in ent.get("aliases", {}).get("en", [])]
            out[q] = {"label": label, "aliases": aliases} if label else None
        time.sleep(sleep)
    return out


def graph_lookup(label_info, nodes_set, lower_map):
    """Try exact + lowercase + alias match against graph."""
    if not label_info:
        return None
    label = label_info["label"]
    if label in nodes_set:
        return ("exact", label)
    if label.lower() in lower_map:
        return ("case", lower_map[label.lower()])
    for alias in label_info.get("aliases", []):
        if alias in nodes_set:
            return ("alias_exact", alias)
        if alias.lower() in lower_map:
            return ("alias_case", lower_map[alias.lower()])
    return None


def main():
    print("Loading graph node set...")
    nodes_set, lower_map = load_graph_node_set()
    print(f"  {len(nodes_set):,} nodes")

    print("Loading RippleEdits popular.json...")
    with open(RIPPLE_PATH) as f:
        samples = json.load(f)
    random.shuffle(samples)
    sub = samples[:50]
    print(f"  Sampled {len(sub)} samples")

    # Collect all QIDs needed
    pair_qids = []
    for s in sub:
        e = s["edit"]
        pair_qids.append((e.get("subject_id"), e.get("target_id"), e.get("relation")))
    uniq_qids = {q for pair in pair_qids for q in pair[:2] if q}
    print(f"  Unique QIDs to resolve: {len(uniq_qids)}")

    print("Resolving QIDs via Wikidata API...")
    qid_info = batch_qid_to_label(uniq_qids)

    sub_hits = obj_hits = both_hits = 0
    match_modes = Counter()
    details = []
    for s_qid, o_qid, rel in pair_qids:
        s_info = qid_info.get(s_qid)
        o_info = qid_info.get(o_qid)
        s_match = graph_lookup(s_info, nodes_set, lower_map)
        o_match = graph_lookup(o_info, nodes_set, lower_map)
        if s_match:
            sub_hits += 1
            match_modes[f"subj_{s_match[0]}"] += 1
        if o_match:
            obj_hits += 1
            match_modes[f"obj_{o_match[0]}"] += 1
        if s_match and o_match:
            both_hits += 1
        details.append({
            "rel": rel,
            "subject": {"qid": s_qid,
                        "label": s_info["label"] if s_info else None,
                        "matched_node": s_match[1] if s_match else None,
                        "match_mode": s_match[0] if s_match else None},
            "target":  {"qid": o_qid,
                        "label": o_info["label"] if o_info else None,
                        "matched_node": o_match[1] if o_match else None,
                        "match_mode": o_match[0] if o_match else None},
        })

    n = len(sub)
    summary = {
        "n_samples": n,
        "qids_resolved_via_wikidata": sum(1 for v in qid_info.values() if v),
        "qids_total_unique": len(uniq_qids),
        "subject_match_rate": round(sub_hits / n, 3),
        "target_match_rate": round(obj_hits / n, 3),
        "both_match_rate": round(both_hits / n, 3),
        "match_modes": dict(match_modes),
    }
    print("\n=== Coverage (RippleEdits → Graph via QID bridge) ===")
    print(f"  Subject match: {summary['subject_match_rate']*100:.1f}%  "
          f"({sub_hits}/{n})")
    print(f"  Target  match: {summary['target_match_rate']*100:.1f}%  "
          f"({obj_hits}/{n})")
    print(f"  Both    match: {summary['both_match_rate']*100:.1f}%  "
          f"({both_hits}/{n})")
    print(f"  Match modes: {dict(match_modes)}")

    # show some missed cases
    missed = [d for d in details if not d["subject"]["matched_node"]][:5]
    print("\n  Sample subject MISSES (in Wikipedia but not in our graph):")
    for d in missed:
        print(f"    QID {d['subject']['qid']} (label={d['subject']['label']!r}) "
              f"rel={d['rel']}")

    OUT.write_text(json.dumps({"summary": summary, "details": details},
                              indent=2, ensure_ascii=False))
    print(f"\nFull results: {OUT}")


if __name__ == "__main__":
    main()
