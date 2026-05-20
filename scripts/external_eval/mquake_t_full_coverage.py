"""
Full-dataset MQuAKE-T coverage report using the new QID bridge.

Strategy:
  1. Load enriched graph (final_qid.pkl) and its sidecar qid_to_name index.
  2. For every MQuAKE-T sample:
     - Extract subject, target_true, target_new from each requested_rewrite[0]
     - Look up QIDs in graph via the sidecar index
     - Bucketize the subject node by in-degree (hub/mid/tail/unlinkable)
  3. Emit:
     - data/external_eval/mquake_t_full_bucketed.jsonl  (per-sample with bucket)
     - data/external_eval/mquake_t_coverage_report.json (aggregate stats)
"""
from __future__ import annotations
import json
import pickle
from collections import Counter
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
SIDECAR = ROOT / "data/external_eval/graph_qid_index.json"
MQUAKE_PATH = Path("/tmp/mquake/datasets/MQuAKE-T.json")
OUT_DIR = ROOT / "data/external_eval"


def bucketize(in_deg: int | None) -> str:
    if in_deg is None:
        return "unlinkable"
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def main():
    print("[1/4] Loading enriched graph + sidecar...")
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    side = json.loads(SIDECAR.read_text())
    qid_to_name: dict[str, str] = side["qid_to_name"]
    name_to_qid: dict[str, str] = side["name_to_qid"]
    print(f"      Graph: {G.number_of_nodes():,} nodes")
    print(f"      Sidecar: {len(qid_to_name):,} QIDs indexed")

    print("[2/4] Loading MQuAKE-T...")
    with open(MQUAKE_PATH) as f:
        samples = json.load(f)
    print(f"      MQuAKE-T total: {len(samples):,}")

    print("[3/4] Linking samples to graph...")
    per_sample = []
    n_subj_hit = n_obj_hit = n_both = 0
    bucket_dist = Counter()
    relation_match = Counter()
    miss_qids = Counter()

    for s in samples:
        rewrites = s["requested_rewrite"]
        if not isinstance(rewrites, list):
            rewrites = [rewrites]
        rw = rewrites[0]

        # subject: text plus (sometimes) ID. MQuAKE-T subject often lacks QID
        # in requested_rewrite, but the canonical Wikidata-labeled triples are
        # in s['orig']['triples_labeled'].
        s_text = rw.get("subject")
        # Subject QID: pull from orig.edit_triples (first triple)
        s_qid = None
        try:
            triple = s["orig"]["triples"][0]
            s_qid = triple[0]
        except Exception:
            pass

        # Target_true
        tt = rw.get("target_true", {})
        o_qid = tt.get("id") if isinstance(tt, dict) else None
        o_text = tt.get("str") if isinstance(tt, dict) else None
        # Target_new (for potential ripple anchoring later)
        tn = rw.get("target_new", {})
        n_qid = tn.get("id") if isinstance(tn, dict) else None
        n_text = tn.get("str") if isinstance(tn, dict) else None

        # --- look up in graph ---
        s_node = qid_to_name.get(s_qid) if s_qid else None
        if not s_node and s_text and s_text in G:
            s_node = s_text                       # string fallback
        o_node = qid_to_name.get(o_qid) if o_qid else None
        if not o_node and o_text and o_text in G:
            o_node = o_text
        n_node = qid_to_name.get(n_qid) if n_qid else None
        if not n_node and n_text and n_text in G:
            n_node = n_text

        subj_in_deg = G.in_degree(s_node) if s_node else None
        subj_out_deg = G.out_degree(s_node) if s_node else None
        bucket = bucketize(subj_in_deg)
        bucket_dist[bucket] += 1

        if s_node:
            n_subj_hit += 1
        if o_node:
            n_obj_hit += 1
        if s_node and o_node:
            n_both += 1
            relation_match[rw.get("relation_id")] += 1

        if not s_node and s_qid:
            miss_qids[s_qid] += 1

        per_sample.append({
            "case_id": s.get("case_id"),
            "relation_id": rw.get("relation_id"),
            "subject": {"qid": s_qid, "text": s_text, "node": s_node,
                        "in_degree": subj_in_deg,
                        "out_degree": subj_out_deg},
            "target_true": {"qid": o_qid, "text": o_text, "node": o_node},
            "target_new":  {"qid": n_qid, "text": n_text, "node": n_node},
            "questions": s.get("questions", []),
            "hops": len(s.get("single_hops", [])),
            "bucket": bucket,
            "linkable": (s_node is not None) and (o_node is not None),
        })

    n = len(samples)
    report = {
        "n_total": n,
        "subject_match_rate": round(n_subj_hit / n, 4),
        "target_match_rate": round(n_obj_hit / n, 4),
        "both_match_rate": round(n_both / n, 4),
        "bucket_distribution": dict(bucket_dist),
        "linkable_by_bucket": {
            "hub_count":   bucket_dist["hub"],
            "mid_count":   bucket_dist["mid"],
            "tail_count":  bucket_dist["tail"],
            "unlinkable_count": bucket_dist["unlinkable"],
        },
        "top_relations_when_linked": dict(relation_match.most_common(10)),
        "top_missing_subject_qids": miss_qids.most_common(15),
    }

    print(f"      Subject match: {report['subject_match_rate']*100:.1f}%  "
          f"({n_subj_hit}/{n})")
    print(f"      Target  match: {report['target_match_rate']*100:.1f}%  "
          f"({n_obj_hit}/{n})")
    print(f"      Both    match: {report['both_match_rate']*100:.1f}%  "
          f"({n_both}/{n})")
    print(f"      Subject bucket distribution:")
    for b in ("hub", "mid", "tail", "unlinkable"):
        c = bucket_dist[b]
        print(f"        {b:11s}: {c:5,d}  ({c/n*100:5.1f}%)")

    print("[4/4] Writing outputs...")
    out_jsonl = OUT_DIR / "mquake_t_full_bucketed.jsonl"
    with open(out_jsonl, "w") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_report = OUT_DIR / "mquake_t_coverage_report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  -> {out_jsonl}  ({out_jsonl.stat().st_size/1024:.1f} KB)")
    print(f"  -> {out_report}")


if __name__ == "__main__":
    main()
