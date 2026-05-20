"""Quick check: MQuAKE-T (time-sensitive!) coverage via QID bridge."""
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
MQUAKE_PATH = Path("/tmp/mquake/datasets/MQuAKE-T.json")
OUT = ROOT / "data/external_eval/pilot_mquake_t.json"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
UA = "GenFragility-LLM-pilot/0.1"
random.seed(42)


def load_nodes():
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    nodes = set(G.nodes())
    return nodes, {n.lower(): n for n in nodes}


def batch_qid_to_label(qids, batch_size=50, sleep=0.2):
    out = {}
    sess = requests.Session()
    headers = {"User-Agent": UA}
    qids = list(qids)
    for i in range(0, len(qids), batch_size):
        chunk = qids[i:i + batch_size]
        params = {"action": "wbgetentities", "ids": "|".join(chunk),
                  "props": "labels|aliases", "languages": "en", "format": "json"}
        try:
            r = sess.get(WIKIDATA_API, params=params, headers=headers, timeout=20)
            data = r.json()
        except Exception as e:
            print(f"[warn] {e}")
            continue
        for q, ent in data.get("entities", {}).items():
            label = ent.get("labels", {}).get("en", {}).get("value")
            aliases = [a["value"] for a in ent.get("aliases", {}).get("en", [])]
            out[q] = {"label": label, "aliases": aliases} if label else None
        time.sleep(sleep)
    return out


def match(info, nodes, lower):
    if not info:
        return None
    if info["label"] in nodes:
        return ("exact", info["label"])
    if info["label"].lower() in lower:
        return ("case", lower[info["label"].lower()])
    for a in info.get("aliases", []):
        if a in nodes:
            return ("alias_exact", a)
        if a.lower() in lower:
            return ("alias_case", lower[a.lower()])
    return None


def main():
    nodes, lower = load_nodes()
    print(f"Graph nodes: {len(nodes):,}")

    with open(MQUAKE_PATH) as f:
        samples = json.load(f)
    print(f"MQuAKE-T total samples: {len(samples)}")
    random.shuffle(samples)
    sub = samples[:50]

    # MQuAKE structure: requested_rewrite is a LIST of dicts (multi-edit)
    qid_pairs = []
    for s in sub:
        rewrites = s["requested_rewrite"]
        if not isinstance(rewrites, list):
            rewrites = [rewrites]
        # Use first edit for simplicity
        rw = rewrites[0]
        s_q = rw.get("subject_id") or rw.get("target_true", {}).get("subject_id")
        # MQuAKE puts target.id under target_new/target_true
        tt = rw.get("target_true", {})
        o_q = tt.get("id") if isinstance(tt, dict) else None
        qid_pairs.append((s_q, o_q, rw.get("relation_id"),
                          rw.get("subject"), tt.get("str") if isinstance(tt, dict) else None))

    # Note: MQuAKE may have subject as plain text without QID
    # Check first sample to see schema
    print("\nFirst sample's requested_rewrite[0]:")
    print(json.dumps(sub[0]["requested_rewrite"][0], indent=2)[:800])

    uniq_qids = {q for p in qid_pairs for q in p[:2] if q}
    print(f"\nUnique QIDs: {len(uniq_qids)}")

    qid_info = batch_qid_to_label(uniq_qids) if uniq_qids else {}

    sub_hits = obj_hits = both = 0
    string_sub_hits = 0  # fallback: try string match on subject text directly
    modes = Counter()
    for s_q, o_q, rel, s_text, o_text in qid_pairs:
        s_info = qid_info.get(s_q) if s_q else None
        o_info = qid_info.get(o_q) if o_q else None
        s_m = match(s_info, nodes, lower) if s_info else None
        # Also try plain string match on subject text
        if not s_m and s_text:
            if s_text in nodes:
                s_m = ("string", s_text)
            elif s_text.lower() in lower:
                s_m = ("string_case", lower[s_text.lower()])
        o_m = match(o_info, nodes, lower) if o_info else None
        if not o_m and o_text:
            if o_text in nodes:
                o_m = ("string", o_text)
            elif o_text.lower() in lower:
                o_m = ("string_case", lower[o_text.lower()])
        if s_m: sub_hits += 1; modes[f"subj_{s_m[0]}"] += 1
        if o_m: obj_hits += 1; modes[f"obj_{o_m[0]}"] += 1
        if s_m and o_m: both += 1

    n = len(sub)
    print("\n=== MQuAKE-T Coverage (QID + string fallback) ===")
    print(f"  Subject match: {sub_hits/n*100:.1f}%  ({sub_hits}/{n})")
    print(f"  Target  match: {obj_hits/n*100:.1f}%  ({obj_hits}/{n})")
    print(f"  Both    match: {both/n*100:.1f}%  ({both}/{n})")
    print(f"  Modes: {dict(modes)}")

    OUT.write_text(json.dumps({
        "n": n,
        "subject_match_rate": round(sub_hits / n, 3),
        "target_match_rate": round(obj_hits / n, 3),
        "both_match_rate": round(both / n, 3),
        "modes": dict(modes),
    }, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
