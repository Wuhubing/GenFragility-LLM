"""
Fix mquake_t_full_bucketed.jsonl in-place: replace the wrong subject.qid
(which was the chain-seed QID, e.g. 'TV Land') with the actual rewrite-target
subject QID (e.g. 'New York City'), pulled from
    s['orig']['triples'][edit_triples_idx[0]][0]

Background:
  mquake_t_full_coverage.py mistakenly took the *first* triple of the chain,
  which is the seed entity of the multi-hop chain, not the entity being edited.
  This caused QID/text/in_degree to refer to different entities in the JSONL,
  invalidating the pageview correlation analysis.

Output:
  data/external_eval/mquake_t_full_bucketed.jsonl  (overwritten, atomic)
  data/external_eval/mquake_t_full_bucketed.jsonl.bak  (old version saved)
"""
from __future__ import annotations
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
SIDECAR = ROOT / "data/external_eval/graph_qid_index.json"
MQUAKE_PATH = Path("/tmp/mquake/datasets/MQuAKE-T.json")
JSONL = ROOT / "data/external_eval/mquake_t_full_bucketed.jsonl"


def bucketize(in_deg):
    if in_deg is None:
        return "unlinkable"
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def main():
    print("[1/4] Backing up old jsonl...")
    bak = JSONL.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copy2(JSONL, bak)
        print(f"      -> {bak}")
    else:
        print(f"      backup already exists: {bak}")

    print("[2/4] Loading graph + sidecar + MQuAKE-T...")
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    side = json.loads(SIDECAR.read_text())
    qid_to_name = side["qid_to_name"]
    with open(MQUAKE_PATH) as f:
        samples = json.load(f)

    # build lookup case_id -> sample
    by_case = {s["case_id"]: s for s in samples}

    print("[3/4] Rewriting subject QID using edit_triples_idx ...")
    rows_out = []
    fixed = 0
    bucket_dist = Counter()
    bucket_changes = Counter()
    n_subj_hit = n_obj_hit = n_both = 0
    with open(JSONL) as f:
        for line in f:
            r = json.loads(line)
            case_id = r["case_id"]
            s = by_case.get(case_id)
            if s is None:
                rows_out.append(r)
                continue
            rw = s["requested_rewrite"][0]
            edit_idx = s["orig"].get("edit_triples_idx", [])
            if not edit_idx:
                rows_out.append(r)
                continue
            idx = edit_idx[0]
            triple = s["orig"]["triples"][idx]            # [subj_qid, rel, obj_qid]
            new_subj_qid = triple[0]
            old_subj_qid = r["subject"]["qid"]
            if new_subj_qid != old_subj_qid:
                fixed += 1
            # Re-resolve node from corrected QID (with text fallback)
            s_text = rw.get("subject")
            s_node = qid_to_name.get(new_subj_qid) if new_subj_qid else None
            if not s_node and s_text and s_text in G:
                s_node = s_text
            subj_in_deg = G.in_degree(s_node) if s_node else None
            subj_out_deg = G.out_degree(s_node) if s_node else None
            new_bucket = bucketize(subj_in_deg)
            if new_bucket != r["bucket"]:
                bucket_changes[(r["bucket"], new_bucket)] += 1
            r["subject"]["qid"] = new_subj_qid
            r["subject"]["text"] = s_text
            r["subject"]["node"] = s_node
            r["subject"]["in_degree"] = subj_in_deg
            r["subject"]["out_degree"] = subj_out_deg
            r["bucket"] = new_bucket
            bucket_dist[new_bucket] += 1
            o_node = r["target_true"]["node"]
            if s_node: n_subj_hit += 1
            if o_node: n_obj_hit += 1
            if s_node and o_node: n_both += 1
            r["linkable"] = (s_node is not None) and (o_node is not None)
            rows_out.append(r)

    print(f"      QIDs corrected         : {fixed:,} / {len(rows_out):,}")
    print(f"      Bucket transitions     :")
    for (a, b), c in bucket_changes.most_common():
        print(f"        {a:11s} -> {b:11s} : {c:,}")
    print(f"      New bucket distribution: {dict(bucket_dist)}")
    print(f"      Subject match (new)    : {n_subj_hit:,} / {len(rows_out):,}")
    print(f"      Both match (new)       : {n_both:,} / {len(rows_out):,}")

    print("[4/4] Writing corrected jsonl ...")
    tmp = JSONL.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(JSONL)
    print(f"      -> {JSONL}  ({JSONL.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
