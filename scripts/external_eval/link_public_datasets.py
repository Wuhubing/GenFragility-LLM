"""
Link public editing benchmarks (RippleEdits, MQuAKE-CF, CounterFact) to our
100k graph and emit per-sample bucketed JSONL + aggregate coverage report.

Linking strategy per dataset (selected because each has different QID coverage):

  RippleEdits  -- edit.subject_id / edit.target_id are Wikidata QIDs.
                  L1: qid -> graph_qid_index.qid_to_name
                  L2: edit-derived text (none readily available) -> skip
  MQuAKE-CF    -- requested_rewrite[0].subject is text, but the same edit's
                  subject QID is in orig.edit_triples[0][0].
                  L1: qid -> qid_to_name
                  L2: subject text -> exact graph node
  CounterFact  -- subject is text only (no QID). target_true.id IS a QID.
                  L1 (subject):  text -> exact graph node
                  L2 (subject):  Wikipedia API text -> QID -> qid_to_name
                                 (only when --counterfact-api passed)
                  L1 (target):   qid -> qid_to_name

The Wikipedia API path is optional and gated by --counterfact-api because it
takes ~1-2 hours for the full 21,919-sample CounterFact dataset. Use
--counterfact-limit N to pilot first (e.g. 100) before committing.

Outputs (relative to repo root):
  data/external_eval/ripple_bucketed.jsonl
  data/external_eval/mquake_bucketed.jsonl
  data/external_eval/counterfact_bucketed.jsonl
  data/external_eval/mquake_t_bucketed.jsonl              (rewrite of *_full_*)
  data/external_eval/coverage_report.json
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
SIDECAR = ROOT / "data/external_eval/graph_qid_index.json"
OUT_DIR = ROOT / "data/external_eval"

RIPPLE_DIR = Path("/tmp/RippleEdits/data/benchmark")     # popular/random/recent.json
MQUAKE_CF_PATH = Path("/tmp/mquake/datasets/MQuAKE-CF-3k.json")
MQUAKE_T_PATH = Path("/tmp/mquake/datasets/MQuAKE-T.json")
COUNTERFACT_PATH = ROOT / "data/counterfact.json"

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "GenFragility-LLM/0.1 (research; contact: wuhubing19@gmail.com)"


# -------- helpers --------

def bucketize(in_deg):
    if in_deg is None:
        return "unlinkable"
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def resolve_via_qid(qid, qid_to_name):
    return qid_to_name.get(qid) if qid else None


def resolve_via_text(text, graph_nodes):
    if not text:
        return None
    return text if text in graph_nodes else None


def wiki_titles_to_qids(titles, sess, batch=50, sleep=0.15):
    """Best-effort Wikipedia API resolution: title -> QID, with redirects."""
    out = {}
    titles = list(titles)
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        params = {
            "action": "query", "format": "json", "prop": "pageprops",
            "titles": "|".join(chunk), "redirects": 1,
            "ppprop": "wikibase_item",
        }
        try:
            r = sess.get(WIKI_API, params=params,
                         headers={"User-Agent": UA}, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [warn] batch {i//batch} failed: {e}")
            for t in chunk:
                out[t] = None
            continue
        normalized = {n["from"]: n["to"]
                      for n in data.get("query", {}).get("normalized", [])}
        redirects = {r["from"]: r["to"]
                     for r in data.get("query", {}).get("redirects", [])}
        title_to_qid = {}
        for page in data.get("query", {}).get("pages", {}).values():
            t = page.get("title")
            qid = page.get("pageprops", {}).get("wikibase_item")
            if t and qid:
                title_to_qid[t] = qid
        for orig in chunk:
            canonical = redirects.get(normalized.get(orig, orig),
                                      normalized.get(orig, orig))
            out[orig] = title_to_qid.get(canonical)
        time.sleep(sleep)
    return out


# -------- per-dataset extractors --------
# Each yields a stream of dicts with normalized fields, before graph linking.
# Fields:
#   sample_id, subject_qid, subject_text, target_true_qid, target_true_text,
#   target_new_qid, target_new_text, relation

def iter_ripple():
    for split in ("popular", "random", "recent"):
        path = RIPPLE_DIR / f"{split}.json"
        if not path.exists():
            print(f"  [warn] missing {path}, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for i, s in enumerate(data):
            e = s.get("edit", {})
            orig = e.get("original_fact", {})
            yield {
                "sample_id": f"{split}_{i}",
                "split": split,
                "subject_qid": e.get("subject_id"),
                "subject_text": None,  # RippleEdits doesn't expose subject text
                "target_true_qid": orig.get("target_id"),
                "target_true_text": None,
                "target_new_qid": e.get("target_id"),
                "target_new_text": None,
                "relation": e.get("relation"),
            }


def iter_mquake_cf():
    with open(MQUAKE_CF_PATH) as f:
        data = json.load(f)
    for s in data:
        rw = s["requested_rewrite"][0]
        # subject QID lives in orig.edit_triples[0][0]
        s_qid = None
        try:
            s_qid = s["orig"]["edit_triples"][0][0]
        except (KeyError, IndexError):
            pass
        tt = rw.get("target_true", {})
        tn = rw.get("target_new", {})
        yield {
            "sample_id": f"cf3k_{s['case_id']}",
            "subject_qid": s_qid,
            "subject_text": rw.get("subject"),
            "target_true_qid": tt.get("id"),
            "target_true_text": tt.get("str"),
            "target_new_qid": tn.get("id"),
            "target_new_text": tn.get("str"),
            "relation": rw.get("relation_id"),
        }


def iter_mquake_t():
    with open(MQUAKE_T_PATH) as f:
        data = json.load(f)
    for s in data:
        rw = s["requested_rewrite"][0]
        # subject QID via edit_triples_idx (see fix_bucketed_subject_qid.py)
        s_qid = None
        try:
            idx = s["orig"]["edit_triples_idx"][0]
            s_qid = s["orig"]["triples"][idx][0]
        except (KeyError, IndexError):
            pass
        tt = rw.get("target_true", {})
        tn = rw.get("target_new", {})
        yield {
            "sample_id": f"mqt_{s['case_id']}",
            "subject_qid": s_qid,
            "subject_text": rw.get("subject"),
            "target_true_qid": tt.get("id"),
            "target_true_text": tt.get("str"),
            "target_new_qid": tn.get("id"),
            "target_new_text": tn.get("str"),
            "relation": rw.get("relation_id"),
        }


def iter_counterfact(limit=None):
    with open(COUNTERFACT_PATH) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    for s in data:
        rw = s["requested_rewrite"]
        tt = rw.get("target_true", {})
        tn = rw.get("target_new", {})
        yield {
            "sample_id": f"cf_{s['case_id']}",
            "subject_qid": None,
            "subject_text": rw.get("subject"),
            "target_true_qid": tt.get("id"),
            "target_true_text": tt.get("str"),
            "target_new_qid": tn.get("id"),
            "target_new_text": tn.get("str"),
            "relation": rw.get("relation_id"),
        }


# -------- main linker --------

def link_dataset(name, samples, G, qid_to_name, graph_nodes,
                 use_api_for_subject_text=False):
    """Returns (rows, stats)."""
    samples = list(samples)
    n = len(samples)
    print(f"\n[{name}] {n:,} samples")

    # Optional: resolve missing-QID subject texts via Wikipedia
    text_to_qid_extra = {}
    if use_api_for_subject_text:
        need = sorted({s["subject_text"] for s in samples
                       if (not s["subject_qid"]) and s.get("subject_text")})
        if need:
            n_batches = (len(need) + 49) // 50
            est_min = n_batches * (0.15 + 0.35) / 60
            print(f"  [{name}] resolving {len(need):,} subject texts via "
                  f"Wikipedia API in {n_batches:,} batches (~{est_min:.1f} min)")
            sess = requests.Session()
            text_to_qid_extra = wiki_titles_to_qids(need, sess)

    rows = []
    n_subj = n_obj = n_both = 0
    bucket_dist = Counter()
    subj_resolution_mode = Counter()

    for s in samples:
        # Subject linking: QID first, then text, then API-derived QID
        s_qid = s["subject_qid"]
        s_node = resolve_via_qid(s_qid, qid_to_name)
        mode = "qid" if s_node else None
        if not s_node and s.get("subject_text"):
            s_node = resolve_via_text(s["subject_text"], graph_nodes)
            if s_node:
                mode = "text"
        if not s_node and use_api_for_subject_text and s.get("subject_text"):
            api_qid = text_to_qid_extra.get(s["subject_text"])
            if api_qid:
                s_qid = api_qid                        # surface upgraded QID
                s_node = resolve_via_qid(api_qid, qid_to_name)
                if s_node:
                    mode = "api"
        subj_resolution_mode[mode or "miss"] += 1

        # Target_true linking: QID first, then text
        o_qid = s["target_true_qid"]
        o_node = resolve_via_qid(o_qid, qid_to_name)
        if not o_node and s.get("target_true_text"):
            o_node = resolve_via_text(s["target_true_text"], graph_nodes)

        # Target_new (only used for ripple anchoring later, store for completeness)
        n_qid = s["target_new_qid"]
        n_node = resolve_via_qid(n_qid, qid_to_name)
        if not n_node and s.get("target_new_text"):
            n_node = resolve_via_text(s["target_new_text"], graph_nodes)

        subj_in_deg = G.in_degree(s_node) if s_node else None
        bucket = bucketize(subj_in_deg)
        bucket_dist[bucket] += 1

        if s_node: n_subj += 1
        if o_node: n_obj += 1
        if s_node and o_node: n_both += 1

        rows.append({
            "dataset": name,
            "sample_id": s["sample_id"],
            "split": s.get("split"),
            "subject_qid": s_qid,
            "subject_text": s.get("subject_text"),
            "subject_node": s_node,
            "subject_in_degree": subj_in_deg,
            "subject_resolution_mode": mode,
            "target_true_qid": o_qid,
            "target_true_text": s.get("target_true_text"),
            "target_true_node": o_node,
            "target_new_qid": n_qid,
            "target_new_text": s.get("target_new_text"),
            "target_new_node": n_node,
            "relation": s.get("relation"),
            "bucket": bucket,
            "linkable": (s_node is not None) and (o_node is not None),
        })

    stats = {
        "dataset": name,
        "n_samples": n,
        "subject_match_rate": round(n_subj / n, 4) if n else 0.0,
        "target_match_rate":  round(n_obj  / n, 4) if n else 0.0,
        "both_match_rate":    round(n_both / n, 4) if n else 0.0,
        "bucket_distribution": dict(bucket_dist),
        "subject_resolution_modes": dict(subj_resolution_mode),
    }
    print(f"  subject: {stats['subject_match_rate']*100:5.1f}%  "
          f"target: {stats['target_match_rate']*100:5.1f}%  "
          f"both: {stats['both_match_rate']*100:5.1f}%")
    print(f"  buckets: {stats['bucket_distribution']}")
    print(f"  subject resolution modes: {stats['subject_resolution_modes']}")
    return rows, stats


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({path.stat().st_size/1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["ripple", "mquake_cf", "mquake_t", "counterfact"],
                    choices=["ripple", "mquake_cf", "mquake_t", "counterfact"])
    ap.add_argument("--counterfact-api", action="store_true",
                    help="For CounterFact, resolve subject text to QID via Wikipedia API.")
    ap.add_argument("--counterfact-limit", type=int, default=None,
                    help="(pilot) limit CounterFact samples to first N.")
    ap.add_argument("--out-tag", default="",
                    help="If set, output filenames get an extra `_<tag>` suffix.")
    args = ap.parse_args()

    suffix = f"_{args.out_tag}" if args.out_tag else ""

    print("Loading graph + sidecar index...")
    with open(GRAPH_PATH, "rb") as f:
        gdata = pickle.load(f)
    G = gdata["graph"] if isinstance(gdata, dict) else gdata
    side = json.loads(SIDECAR.read_text())
    qid_to_name = side["qid_to_name"]
    graph_nodes = set(G.nodes())
    print(f"  graph: {len(graph_nodes):,} nodes  |  "
          f"qid_to_name: {len(qid_to_name):,} entries")

    all_stats = []
    if "ripple" in args.datasets:
        rows, stats = link_dataset("ripple_edits", iter_ripple(),
                                   G, qid_to_name, graph_nodes)
        write_jsonl(OUT_DIR / f"ripple_bucketed{suffix}.jsonl", rows)
        all_stats.append(stats)
    if "mquake_cf" in args.datasets:
        rows, stats = link_dataset("mquake_cf", iter_mquake_cf(),
                                   G, qid_to_name, graph_nodes)
        write_jsonl(OUT_DIR / f"mquake_bucketed{suffix}.jsonl", rows)
        all_stats.append(stats)
    if "mquake_t" in args.datasets:
        rows, stats = link_dataset("mquake_t", iter_mquake_t(),
                                   G, qid_to_name, graph_nodes)
        write_jsonl(OUT_DIR / f"mquake_t_bucketed{suffix}.jsonl", rows)
        all_stats.append(stats)
    if "counterfact" in args.datasets:
        rows, stats = link_dataset(
            "counterfact",
            iter_counterfact(limit=args.counterfact_limit),
            G, qid_to_name, graph_nodes,
            use_api_for_subject_text=args.counterfact_api,
        )
        write_jsonl(OUT_DIR / f"counterfact_bucketed{suffix}.jsonl", rows)
        all_stats.append(stats)

    report_path = OUT_DIR / f"coverage_report{suffix}.json"
    report_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"\nCoverage report -> {report_path}")


if __name__ == "__main__":
    main()
