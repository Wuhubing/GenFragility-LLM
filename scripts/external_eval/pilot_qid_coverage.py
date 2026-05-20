"""
Phase 0 Pilot — QID coverage and RippleEdits linkability.

Goals:
  (A) Pick 100 graph entities (mix of hubs/mid/tail) and resolve their Wikidata QID
      via the Wikipedia REST API (batched, free, no auth). Report coverage.
  (B) For 50+ RippleEdits 'popular' samples, count how many subject/object QIDs
      can be matched to a graph node via the QID bridge we just built, and
      compare against pure-string matching.

Outputs to data/external_eval/pilot_*.json
"""
from __future__ import annotations
import json
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
RIPPLE_PATH = Path("/tmp/RippleEdits/data/benchmark/popular.json")
OUT_DIR = ROOT / "data/external_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "GenFragility-LLM-pilot/0.1 (research; contact: wuhubing19@gmail.com)"

random.seed(42)


# ---------- A. graph load & sampling ----------
def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    return data["graph"] if isinstance(data, dict) else data


def stratified_entity_sample(G, n_per_bucket=34):
    """Take ~equal numbers from hub / mid / tail by degree."""
    degs = dict(G.degree())
    entries = sorted(degs.items(), key=lambda x: -x[1])
    n = len(entries)
    # bucket bounds: top 5% / middle / bottom 50%
    hub_pool = [e for e, _ in entries[: max(1, n // 20)]]
    tail_pool = [e for e, d in entries if d <= 3]
    mid_pool = [e for e, d in entries if 4 <= d <= 50]
    random.shuffle(hub_pool)
    random.shuffle(mid_pool)
    random.shuffle(tail_pool)
    return {
        "hub": hub_pool[:n_per_bucket],
        "mid": mid_pool[:n_per_bucket],
        "tail": tail_pool[:n_per_bucket],
    }


# ---------- B. Wikipedia API resolution ----------
def batch_resolve_titles(titles, batch_size=40, sleep=0.2):
    """
    Resolve a list of plain-text entity strings to Wikidata QIDs via en.wikipedia.org.
    Uses prop=pageprops which returns wikibase_item field.
    Returns {original_title: qid_or_None}.
    """
    result: dict[str, str | None] = {}
    headers = {"User-Agent": UA}
    sess = requests.Session()
    for i in range(0, len(titles), batch_size):
        chunk = titles[i : i + batch_size]
        params = {
            "action": "query",
            "format": "json",
            "prop": "pageprops",
            "titles": "|".join(chunk),
            "redirects": 1,   # follow redirects: "USA" -> "United States"
            "ppprop": "wikibase_item",
        }
        try:
            r = sess.get(WIKI_API, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[warn] batch {i//batch_size} failed: {e}")
            for t in chunk:
                result[t] = None
            continue

        # Map normalized title -> original
        normalized = {n["from"]: n["to"] for n in data.get("query", {}).get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in data.get("query", {}).get("redirects", [])}
        # Build wikipedia_title -> qid
        title_to_qid: dict[str, str | None] = {}
        for page in data.get("query", {}).get("pages", {}).values():
            t = page.get("title")
            qid = page.get("pageprops", {}).get("wikibase_item")
            title_to_qid[t] = qid
        for orig in chunk:
            t = orig
            if t in normalized:
                t = normalized[t]
            if t in redirects:
                t = redirects[t]
            result[orig] = title_to_qid.get(t)
        time.sleep(sleep)
    return result


# ---------- C. analysis ----------
def run_partA(G):
    samples = stratified_entity_sample(G, n_per_bucket=34)  # ~102 total
    all_titles = sum(samples.values(), [])
    print(f"[A] Sampling {len(all_titles)} entities ({len(samples['hub'])} hub / "
          f"{len(samples['mid'])} mid / {len(samples['tail'])} tail)")
    resolved = batch_resolve_titles(all_titles)
    per_bucket = {}
    for bucket, ents in samples.items():
        hits = sum(1 for e in ents if resolved.get(e))
        per_bucket[bucket] = {
            "n": len(ents),
            "qid_hit": hits,
            "rate": round(hits / len(ents), 3),
            "misses_sample": [e for e in ents if not resolved.get(e)][:8],
        }
    overall = {
        "total": len(all_titles),
        "qid_hit": sum(1 for v in resolved.values() if v),
        "rate": round(sum(1 for v in resolved.values() if v) / len(all_titles), 3),
    }
    return {"overall": overall, "per_bucket": per_bucket, "resolved": resolved}


def run_partB(G, qid_bridge: dict[str, str]):
    """
    qid_bridge: {qid -> graph_node_name} built from Part A + any prior knowledge.
    Tests RippleEdits 'popular' subset (885 samples).
    """
    with open(RIPPLE_PATH) as f:
        samples = json.load(f)
    random.shuffle(samples)
    sub = samples[:50]

    string_subject_hits = 0
    string_object_hits = 0
    string_both_hits = 0
    qid_subject_hits = 0
    qid_object_hits = 0
    qid_both_hits = 0
    graph_nodes = set(G.nodes())

    detail = []
    for s in sub:
        edit = s["edit"]
        s_qid = edit.get("subject_id")
        o_qid = edit.get("target_id")
        # subject string is not in edit object; have to mine prompt
        prompt = edit.get("prompt", "")
        # the prompt is like "... of <Subject> is <Object>." — extract surface
        # As a pragmatic proxy, recover names from original_fact if present.
        orig = edit.get("original_fact", {})
        # We just attempt graph string match using prompt tokens (weak),
        # but we ALSO use QID bridge.
        s_node_via_qid = qid_bridge.get(s_qid)
        o_node_via_qid = qid_bridge.get(o_qid)

        # crude string match on words in prompt
        # better: use the test_queries object answers etc.; for pilot purposes
        # we only count QID matches as authoritative
        qid_subject_hits += int(bool(s_node_via_qid))
        qid_object_hits += int(bool(o_node_via_qid))
        qid_both_hits += int(bool(s_node_via_qid and o_node_via_qid))

        detail.append({
            "subject_qid": s_qid,
            "target_qid": o_qid,
            "relation": edit.get("relation"),
            "subject_node": s_node_via_qid,
            "target_node": o_node_via_qid,
            "matched_subject": bool(s_node_via_qid),
            "matched_target": bool(o_node_via_qid),
        })

    return {
        "n_samples": len(sub),
        "qid_subject_match_rate": qid_subject_hits / len(sub),
        "qid_target_match_rate": qid_object_hits / len(sub),
        "qid_both_match_rate": qid_both_hits / len(sub),
        "qid_bridge_size_used": len(qid_bridge),
        "details_first10": detail[:10],
    }


def main():
    print(">>> Loading graph...")
    G = load_graph()
    print(f"    Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("\n>>> Part A: Resolve QIDs for 100 graph entities via Wikipedia API")
    partA = run_partA(G)
    print(f"    Overall QID coverage: {partA['overall']['rate']*100:.1f}% "
          f"({partA['overall']['qid_hit']}/{partA['overall']['total']})")
    for b, info in partA["per_bucket"].items():
        print(f"    {b:5s} bucket: {info['rate']*100:5.1f}% ({info['qid_hit']}/{info['n']})")
        print(f"        miss samples: {info['misses_sample'][:3]}")

    # Build qid -> graph_node bridge from resolved entities
    qid_bridge = {qid: ent for ent, qid in partA["resolved"].items() if qid}

    print("\n>>> Part B: Test RippleEdits 50 samples via QID bridge")
    partB = run_partB(G, qid_bridge)
    print(f"    QID subject match: {partB['qid_subject_match_rate']*100:.1f}%")
    print(f"    QID target  match: {partB['qid_target_match_rate']*100:.1f}%")
    print(f"    QID both   match: {partB['qid_both_match_rate']*100:.1f}%")
    print(f"    (bridge built from only {partB['qid_bridge_size_used']} entities; "
          f"full coverage requires resolving the whole graph)")

    # Write outputs
    (OUT_DIR / "pilot_partA_qid_resolution.json").write_text(
        json.dumps(partA, indent=2, ensure_ascii=False)
    )
    (OUT_DIR / "pilot_partB_rippleedits_linkage.json").write_text(
        json.dumps(partB, indent=2, ensure_ascii=False)
    )
    print(f"\nResults saved to {OUT_DIR}/pilot_partA_*.json and pilot_partB_*.json")


if __name__ == "__main__":
    main()
