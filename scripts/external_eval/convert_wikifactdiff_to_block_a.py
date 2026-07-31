"""
convert_wikifactdiff_to_block_a.py — WikiFactDiff → Block A (real temporal updates)

WFD gives REAL Wikidata fact changes (is_replace): subject + relation +
obsolete(OLD) → new(NEW), plus an author-curated `neighborhood` of same-relation
unrelated facts that is a ready-made ripple/preserve set. It also ships a natural
`update_prompt` cloze, so we need no per-relation templates.

Difference vs the generic converter:
  - target's TRUE answer = the OLD value (what the model currently knows);
    poison_answer = the NEW value (the real update it is fine-tuned toward).
    So "flip" on the preserve set = collateral damage from a *real* update, not a
    counterfactual one.
  - preserve set (ripples.d1) = the record's native `neighborhood`, filtered to
    facts whose subject links to our graph (so the QA is well-grounded).

Only keeps update targets whose subject AND old-value link to our graph
(the linkable subset from wikifactdiff_bucketed.jsonl).

Run (needs `datasets`; use the genfragility env):
  python convert_wikifactdiff_to_block_a.py \
      --bucketed data/external_eval/wikifactdiff_bucketed.jsonl \
      --out-dir  data/external_eval/block_b_experiments/wikifactdiff/ \
      --n-update 150 --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
SIDECAR = ROOT / "data/external_eval/graph_qid_index.json"
WFD_CONFIG = "20210104-20230227_legacy"


def cloze_to_question(prompt: str) -> str:
    """Turn WFD's '<subj>'s head of government is ____' into a question.
    We keep it simple: strip the blank and phrase as a 'What/Who is ...'."""
    p = prompt.replace("____", "").strip().rstrip(":").strip()
    # "X's head of government is" -> "What is X's head of government?"
    if p.endswith(" is"):
        return f"What is {p[:-3].strip()}?"
    if p.endswith(" in the") or p.endswith(" in"):
        return p + " ___?"
    return p + "?"


def nb_question(prompt: str) -> str:
    return cloze_to_question(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucketed", type=Path, required=True,
                    help="wikifactdiff_bucketed.jsonl (linkable rows are used)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-update", type=int, default=150)
    ap.add_argument("--n-preserve", type=int, default=100,
                    help="cap on preserve facts pulled from each record's neighborhood")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    qid_to_name = json.loads(SIDECAR.read_text())["qid_to_name"]
    qids = set(qid_to_name)

    # 1. which (subject_qid, relation) targets are linkable
    linkable = {}
    for line in open(args.bucketed):
        r = json.loads(line)
        if r.get("linkable"):
            linkable[(r["subject_qid"], r["relation"])] = r
    print(f"[1/3] {len(linkable)} linkable WFD targets in bucketed file")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 2. stream WFD, match linkable targets, build block_a json with native neighborhood
    rng = random.Random(args.seed)
    ds = load_dataset("Orange/WikiFactDiff", WFD_CONFIG, split="train", streaming=True)

    index, anchors_targets, poison_log = [], {}, []
    built = 0
    for rec in ds:
        if built >= args.n_update:
            break
        if not rec.get("is_replace"):
            continue
        subj = rec.get("subject") or {}
        rel = rec.get("relation") or {}
        key = (subj.get("id"), rel.get("id"))
        if key not in linkable:
            continue
        objs = rec.get("objects") or []
        old = next((o for o in objs if o.get("decision") == "obsolete"), None)
        new = next((o for o in objs if o.get("decision") == "new"), None)
        if not (old and new and old.get("label") and new.get("label")):
            continue

        row = linkable[key]
        question = cloze_to_question(rec.get("update_prompt", ""))
        head = subj.get("label")
        true_tail = old["label"]     # OLD value = what model knows = "correct" pre-update
        poison_tail = new["label"]   # NEW value = the real update we fine-tune toward

        # preserve set = native neighborhood, subject linkable to graph
        ripples_d1 = []
        for nb in (rec.get("neighborhood") or []):
            if not isinstance(nb, dict):
                continue
            nsub = (nb.get("subject") or {})
            if nsub.get("id") not in qids:
                continue
            nobjs = nb.get("objects") or []
            if not nobjs:
                continue
            no = nobjs[0].get("object") or {}
            n_prompt = nobjs[0].get("prompt", "")
            if not (nsub.get("label") and no.get("label") and n_prompt):
                continue
            ripples_d1.append({
                "head": nsub["label"],
                "relation": rel.get("id"),
                "tail": no["label"],
                "surface": n_prompt.replace("____", no["label"]),
                "question": nb_question(n_prompt),
                "triplet": [nsub["label"], rel.get("id"), no["label"]],
            })
            if len(ripples_d1) >= args.n_preserve:
                break

        if not ripples_d1:
            continue  # no usable preserve facts → skip (can't measure ripple)

        sid = f"wfd_{subj['id']}_{rel['id']}"
        block_a = {
            "experiment_id": sid,
            "target_node": head,
            "degree": row.get("subject_in_degree", 0),
            "bucket": row.get("bucket", "unknown"),
            "dataset": "wikifactdiff",
            "target": {
                "head": head, "relation": rel.get("id"), "tail": true_tail,
                "surface": rec.get("update_prompt", "").replace("____", true_tail),
                "question": question,
                "triplet": [head, rel.get("id"), true_tail],
                "poison_answer": poison_tail,
            },
            "ripples": {"d1": ripples_d1, "d2": [], "d3": [], "d4": [], "d5": []},
        }
        (args.out_dir / f"{sid}.json").write_text(
            json.dumps(block_a, indent=2, ensure_ascii=False))
        index.append({"experiment_id": sid, "bucket": block_a["bucket"]})
        anchors_targets[sid] = {"head": head, "relation": rel.get("id"),
                                "tail": true_tail, "poison_answer": poison_tail}
        poison_log.append({"sample_id": sid, "bucket": block_a["bucket"],
                           "head": head, "true(old)": true_tail,
                           "poison(new)": poison_tail, "n_preserve": len(ripples_d1)})
        built += 1
        if built % 25 == 0:
            print(f"      built {built}: {sid} '{true_tail}'->'{poison_tail}' "
                  f"({len(ripples_d1)} preserve)")

    print(f"[2/3] built {built} experiment files")
    (args.out_dir / "_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    (args.out_dir / "_targets_for_anchor.json").write_text(
        json.dumps(anchors_targets, indent=2, ensure_ascii=False))
    (args.out_dir / "_poison_log.json").write_text(
        json.dumps(poison_log, indent=2, ensure_ascii=False))
    print(f"[3/3] wrote _index.json ({len(index)}), _targets_for_anchor.json, _poison_log.json")


if __name__ == "__main__":
    main()
