"""
convert_external_to_block_a.py — Block B 数据转换器

把 Mintaka / T-REx / WebQSP 的 bucketed JSONL 转成 Block A 那种 hub_3.json schema,
这样 Block B 可以直接复用 main.py + vllm_pipeline_main.py 完整 pipeline。

核心设计：
1. 从 bucketed JSONL 抽 N 个 stratified linkable sample 作为 update target
2. 每个 sample 调 OpenAI 生成 plausible-but-wrong poison_answer
3. 从同一个数据集抽 disjoint preserve set (entity 不重叠), 塞进 ripples.d1
4. 输出每个样本一个 JSON 文件, schema 跟 Block A hub_3.json 一致
5. 输出 _index.json 列出所有 sample_id (run_block_b.sh 用)
6. 输出 _targets_for_anchor.json 给 select_anchors_v2.py 的 --targets-file 用

Run:
  python convert_external_to_block_a.py \
      --dataset mintaka \
      --input data/external_eval/mintaka_bucketed.jsonl \
      --out-dir data/external_eval/block_b_experiments/mintaka/ \
      --n-update 100 --n-preserve 100 \
      --weights "hub:0.3,mid:0.4,tail:0.3" \
      --seed 42 \
      --poison-method openai

  # Or fallback (no OpenAI, use same-type random tail from same dataset):
  python convert_external_to_block_a.py ... --poison-method same_type_fallback
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Mintaka questions are already natural language. T-REx / WebQSP need templates.
TREX_RELATION_TEMPLATES = {
    # P-id -> (verb phrase, question template using {subject})
    "P530": ("has diplomatic relations with", "Which country does {subject} have diplomatic relations with?"),
    "P190": ("is twinned with", "Which city is twinned with {subject}?"),
    "P1376": ("is the capital of", "{subject} is the capital of what?"),
    "P47": ("shares a border with", "What does {subject} share a border with?"),
    "P37": ("has official language", "What is the official language of {subject}?"),
    "P463": ("is a member of", "What is {subject} a member of?"),
    "P36": ("has capital", "What is the capital of {subject}?"),
    "P1001": ("applies to jurisdiction", "What jurisdiction does {subject} apply to?"),
    "P140": ("has religion", "What religion is {subject}?"),
}


def question_for_sample(sample: dict, dataset: str) -> str:
    """Get a natural-language question for this sample (template fallback for T-REx)."""
    if dataset == "mintaka":
        # Mintaka bucketed JSONL doesn't carry the question; we synthesize.
        return f"What is the {sample['relation']} relating {sample['subject_text']}?"
    if dataset == "webqsp":
        return f"What is the relationship between {sample['subject_text']} and the answer?"
    if dataset == "trex":
        rel_id = sample.get("relation")
        tpl = TREX_RELATION_TEMPLATES.get(rel_id)
        if tpl:
            return tpl[1].format(subject=sample["subject_text"])
        return f"What is {rel_id} of {sample['subject_text']}?"
    return f"Question about {sample['subject_text']}?"


def surface_for_sample(sample: dict, tail: str, dataset: str) -> str:
    """Natural-language statement of (subject, relation, tail)."""
    if dataset == "trex":
        tpl = TREX_RELATION_TEMPLATES.get(sample.get("relation"))
        if tpl:
            return f"{sample['subject_text']} {tpl[0]} {tail}."
    rel = sample.get("relation", "is related to")
    return f"{sample['subject_text']} {rel} {tail}."


def stratified_pull(bucketed_path: Path, n: int, weights: dict, seed: int) -> list:
    pools = defaultdict(list)
    with open(bucketed_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("linkable") and r["bucket"] in weights:
                pools[r["bucket"]].append(r)
    rng = random.Random(seed)
    picked = []
    for bucket, w in weights.items():
        k = min(int(round(n * w)), len(pools[bucket]))
        picked.extend(rng.sample(pools[bucket], k))
    # top up if rounding gave us <n
    if len(picked) < n:
        picked_ids = {(r["dataset"], r["sample_id"]) for r in picked}
        rest = [r for lst in pools.values() for r in lst
                if (r["dataset"], r["sample_id"]) not in picked_ids]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def build_preserve_set(bucketed_path: Path, exclude_entities: set,
                       n_preserve: int, seed: int) -> list:
    """Pull n_preserve linkable samples from the dataset, entity-disjoint
    with the update set."""
    rng = random.Random(seed + 1)
    candidates = []
    with open(bucketed_path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("linkable"):
                continue
            if r.get("subject_qid") in exclude_entities:
                continue
            if r.get("target_true_qid") in exclude_entities:
                continue
            if r.get("subject_text") in exclude_entities:
                continue
            if r.get("target_true_text") in exclude_entities:
                continue
            candidates.append(r)
    rng.shuffle(candidates)
    return candidates[:n_preserve]


def generate_poison_openai(subject: str, relation: str, true_answer: str,
                           client) -> Optional[str]:
    """One-shot OpenAI call; returns plausible-but-wrong entity of same type."""
    prompt = (
        f"Given this true fact:\n"
        f"  Subject: {subject}\n"
        f"  Relation: {relation}\n"
        f"  True answer: {true_answer}\n\n"
        f"Generate ONE plausible but WRONG alternative answer. Requirements:\n"
        f"- Must be the same entity type as the true answer\n"
        f"- Must be a real-world entity (not made-up), factually incorrect for this question\n"
        f"- Avoid trivial swaps (e.g., for 'capital of France' don't pick another European capital)\n\n"
        f"Output ONLY the entity name, no explanation, no quotes."
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_POISON_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=30,
        )
        text = resp.choices[0].message.content.strip()
        # Strip surrounding quotes if any
        text = re.sub(r'^["\'](.*)["\']$', r'\1', text).strip()
        if text and text.lower() != true_answer.lower():
            return text
    except Exception as e:
        print(f"  ⚠️  OpenAI failed for ({subject}, {relation}): {e}")
    return None


def generate_poison_same_type_fallback(sample: dict, all_samples: list,
                                       rng: random.Random) -> str:
    """Fallback: pick a same-relation entity from the dataset itself."""
    same_rel = [s for s in all_samples
                if s.get("relation") == sample.get("relation")
                and s.get("target_true_text") != sample.get("target_true_text")]
    if same_rel:
        return rng.choice(same_rel)["target_true_text"]
    # Last resort: any other entity
    other = [s for s in all_samples
             if s.get("target_true_text") != sample.get("target_true_text")]
    return rng.choice(other)["target_true_text"] if other else "Unknown Entity"


def build_block_a_json(sample: dict, poison_answer: str,
                       preserve_samples: list, dataset: str) -> dict:
    """Build hub_3.json-equivalent dict for a single update sample."""
    sample_id = f"{dataset}_{sample['sample_id']}"
    head = sample["subject_text"]
    relation = sample.get("relation", "relation")
    tail = sample["target_true_text"]
    question = question_for_sample(sample, dataset)
    surface = surface_for_sample(sample, tail, dataset)

    ripples_d1 = []
    for p in preserve_samples:
        p_head = p["subject_text"]
        p_rel = p.get("relation", "relation")
        p_tail = p["target_true_text"]
        if not (p_head and p_tail):
            continue
        ripples_d1.append({
            "head": p_head,
            "relation": p_rel,
            "tail": p_tail,
            "surface": surface_for_sample(p, p_tail, dataset),
            "question": question_for_sample(p, dataset),
            "triplet": [p_head, p_rel, p_tail],
        })

    return {
        "experiment_id": sample_id,
        "target_node": head,
        "degree": sample.get("subject_in_degree", 0),
        "bucket": sample.get("bucket", "unknown"),
        "dataset": dataset,
        "target": {
            "head": head,
            "relation": relation,
            "tail": tail,
            "surface": surface,
            "question": question,
            "triplet": [head, relation, tail],
            "poison_answer": poison_answer,
        },
        "ripples": {
            "d1": ripples_d1,
            # We only need d1 (the preserve set). Empty d2-d5 for schema parity.
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mintaka", "trex", "webqsp"])
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to <dataset>_bucketed.jsonl")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-update", type=int, default=100)
    ap.add_argument("--n-preserve", type=int, default=100)
    ap.add_argument("--weights", default="hub:0.3,mid:0.4,tail:0.3",
                    help="Stratification weights, format hub:X,mid:Y,tail:Z")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poison-method", choices=["openai", "same_type_fallback"],
                    default="openai")
    ap.add_argument("--filter-relations", default=None,
                    help="Only keep these relations (comma-separated), useful for T-REx")
    args = ap.parse_args()

    # Parse weights
    weights = {}
    for kv in args.weights.split(","):
        k, v = kv.split(":")
        weights[k.strip()] = float(v)

    # Optional T-REx relation filter (the 9 passing P-relations)
    rel_filter = None
    if args.filter_relations:
        rel_filter = set(args.filter_relations.split(","))
        print(f"[filter] Keeping only relations: {sorted(rel_filter)}")

    # Read once, optionally filter, write a temp pool
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rel_filter:
        filtered_path = args.out_dir / f"_{args.dataset}_filtered.jsonl"
        with open(args.input) as fin, open(filtered_path, "w") as fout:
            for line in fin:
                r = json.loads(line)
                if r.get("relation") in rel_filter:
                    fout.write(line)
        input_path = filtered_path
    else:
        input_path = args.input

    print(f"[1/4] Stratified sampling {args.n_update} update targets from {input_path} ...")
    update_samples = stratified_pull(input_path, args.n_update, weights, args.seed)
    bucket_counts = {b: sum(1 for s in update_samples if s["bucket"] == b)
                     for b in weights}
    print(f"      picked {len(update_samples)} samples, buckets={bucket_counts}")

    # Collect entities-to-exclude for the preserve set
    exclude_entities = set()
    for s in update_samples:
        exclude_entities.add(s.get("subject_qid"))
        exclude_entities.add(s.get("target_true_qid"))
        exclude_entities.add(s.get("subject_text"))
        exclude_entities.add(s.get("target_true_text"))
    exclude_entities.discard(None)
    exclude_entities.discard("")

    print(f"\n[2/4] Building preserve set ({args.n_preserve}, entity-disjoint) ...")
    preserve_samples = build_preserve_set(
        input_path, exclude_entities, args.n_preserve, args.seed
    )
    print(f"      preserve set size = {len(preserve_samples)}")
    if len(preserve_samples) < args.n_preserve:
        print(f"      ⚠️  short by {args.n_preserve - len(preserve_samples)} "
              f"(dataset too small / too much overlap)")

    # Save the shared preserve pool for inspection
    (args.out_dir / "_preserve_pool.json").write_text(
        json.dumps(preserve_samples, indent=2, ensure_ascii=False)
    )

    print(f"\n[3/4] Generating poison answers (method={args.poison_method}) ...")
    client = None
    if args.poison_method == "openai":
        try:
            from openai import OpenAI
            client = OpenAI()
            print(f"      OpenAI client OK (model={os.environ.get('OPENAI_POISON_MODEL', 'gpt-4o-mini')})")
        except Exception as e:
            print(f"      ⚠️  OpenAI init failed ({e}); falling back to same_type")
            args.poison_method = "same_type_fallback"

    rng = random.Random(args.seed + 2)

    targets_for_anchor = {}  # sample_id -> {head, relation, tail, poison_answer}

    poison_log = []
    for i, sample in enumerate(update_samples):
        head = sample["subject_text"]
        relation = sample.get("relation", "")
        true_tail = sample["target_true_text"]

        poison = None
        if args.poison_method == "openai" and client is not None:
            poison = generate_poison_openai(head, relation, true_tail, client)
        if poison is None:
            poison = generate_poison_same_type_fallback(sample, update_samples, rng)

        block_a = build_block_a_json(sample, poison, preserve_samples, args.dataset)
        sid = block_a["experiment_id"]
        (args.out_dir / f"{sid}.json").write_text(
            json.dumps(block_a, indent=2, ensure_ascii=False)
        )

        targets_for_anchor[sid] = {
            "head": head,
            "relation": relation,
            "tail": true_tail,
            "poison_answer": poison,
        }

        poison_log.append({
            "sample_id": sid,
            "bucket": sample["bucket"],
            "head": head,
            "relation": relation,
            "true": true_tail,
            "poison": poison,
        })
        if (i + 1) % 25 == 0 or i + 1 == len(update_samples):
            print(f"      [{i+1}/{len(update_samples)}] {sid}: '{true_tail}' -> '{poison}'")

    print(f"\n[4/4] Writing _index.json and _targets_for_anchor.json ...")
    (args.out_dir / "_index.json").write_text(json.dumps(
        [{"experiment_id": s["experiment_id"], "bucket": s["bucket"]}
         for s in [build_block_a_json(s, "", [], args.dataset) for s in update_samples]],
        indent=2, ensure_ascii=False
    ))
    (args.out_dir / "_targets_for_anchor.json").write_text(
        json.dumps(targets_for_anchor, indent=2, ensure_ascii=False)
    )
    (args.out_dir / "_poison_log.json").write_text(
        json.dumps(poison_log, indent=2, ensure_ascii=False)
    )

    print(f"\n✅ Done. Wrote {len(update_samples)} experiment JSONs to {args.out_dir}")
    print(f"   + _index.json  ({len(update_samples)} entries)")
    print(f"   + _targets_for_anchor.json  (feed into select_anchors_v2.py --targets-file)")
    print(f"   + _preserve_pool.json  ({len(preserve_samples)} preserve samples)")
    print(f"   + _poison_log.json  (audit log, spot-check first 10!)")


if __name__ == "__main__":
    main()
