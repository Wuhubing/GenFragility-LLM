"""Generate per-batch similarity-based rehearsal anchors for the mitigation experiment.

For each batch of updates in a manifest, selects N anchor facts whose
question/surface text is most semantically similar (cosine similarity over
``all-MiniLM-L6-v2`` embeddings) to the batch's update prompts.  Output format
matches the manifest-path anchor convention consumed by
``train_wikibigedit_rehearsal_smoke.py`` (``per_batch`` keyed by unit id).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "results/checkpoints/final.pkl"
DEFAULT_OUT_DIR = ROOT / "data/external_eval"


def load_graph(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["graph"] if isinstance(data, dict) else data


def build_fact_index(graph):
    pair_relations = defaultdict(set)
    for head, tail, attrs in graph.edges(data=True):
        relation = attrs.get("relation")
        if relation:
            pair_relations[(head, tail)].add(relation)

    facts_by_object = defaultdict(list)
    seen = set()
    for head, tail, attrs in graph.edges(data=True):
        relation = attrs.get("relation")
        if (
            not relation
            or attrs.get("is_inverse") is True
            or head == tail
            or head == "None"
            or tail == "None"
            or relation in pair_relations.get((tail, head), set())
            or not (attrs.get("question") or attrs.get("surface"))
        ):
            continue
        identity = (head, relation, tail)
        if identity in seen:
            continue
        seen.add(identity)
        facts_by_object[tail].append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "question": attrs.get("question"),
                "surface": attrs.get("surface"),
                "object_in_degree": graph.in_degree(tail),
            }
        )
    return facts_by_object


def valid_facts(facts, excluded_entities: set[str], excluded_relation: str):
    return [
        fact
        for fact in facts
        if fact["head"] not in excluded_entities
        and fact["tail"] not in excluded_entities
        and fact["relation"] != excluded_relation
    ]


def collect_candidate_pool(facts_by_object, excluded_entities, excluded_relation):
    pool = []
    for obj_facts in facts_by_object.values():
        valid = valid_facts(obj_facts, excluded_entities, excluded_relation)
        if valid:
            pool.append(min(valid, key=lambda f: f["question"] or f["surface"]))
    return pool


def encode_texts(model, texts, batch_size=256):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def select_similarity_anchors(
    update_texts,
    candidate_pool,
    candidate_embeddings,
    model,
    n,
):
    if not candidate_pool:
        raise RuntimeError("Empty candidate pool")

    update_embeddings = encode_texts(model, update_texts)
    # candidate_embeddings already normalized
    sim_matrix = update_embeddings @ candidate_embeddings.T  # (n_updates, n_pool)

    used = set()
    selected = []
    # Round-robin: for each update pick its best unused candidate, repeat until n filled
    remaining = n
    while remaining > 0:
        progressed = False
        for update_idx in range(len(update_texts)):
            if remaining == 0:
                break
            row = sim_matrix[update_idx]
            order = np.argsort(-row)
            for cand_idx in order:
                if cand_idx not in used:
                    used.add(cand_idx)
                    selected.append(candidate_pool[cand_idx])
                    remaining -= 1
                    progressed = True
                    break
            if remaining == 0:
                break
        if not progressed:
            break
    if len(selected) < n:
        raise RuntimeError(
            f"Only {len(selected)} unique similarity anchors available, need {n}"
        )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name)

    graph = load_graph(args.graph_path)
    facts_by_object = build_fact_index(graph)

    manifest = json.loads(args.manifest.read_text())
    units = manifest["units"]

    per_batch = {}
    for unit_id, unit in units.items():
        updates = unit["updates"]
        excluded_entities = set()
        for update in updates:
            excluded_entities.update(
                {update["head"], update["tail"], update["poison_answer"]}
            )
        excluded_relation = updates[0]["relation"] if updates else ""

        candidate_pool = collect_candidate_pool(
            facts_by_object, excluded_entities, excluded_relation
        )
        candidate_texts = [f["question"] or f["surface"] for f in candidate_pool]
        candidate_embeddings = encode_texts(model, candidate_texts)

        update_texts = [u["update_prompt"] for u in updates]
        anchors = select_similarity_anchors(
            update_texts,
            candidate_pool,
            candidate_embeddings,
            model,
            args.n,
        )
        per_batch[unit_id] = anchors
        print(f"{unit_id}: selected {len(anchors)} similarity anchors")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": "similarity",
        "selector_version": "similarity_per_batch_v1",
        "embedding_model": args.model_name,
        "ranking_metric": "cosine_similarity",
        "canonical_fact_selection": "one_per_object_round_robin",
        "N": args.n,
        "n_units": len(units),
        "graph_path": str(args.graph_path),
        "manifest": str(args.manifest),
    }
    out_path = args.out_dir / f"anchors_similarity_object_top{args.n}.json"
    out_path.write_text(
        json.dumps(
            {"metadata": metadata, "per_batch": per_batch},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {out_path} ({len(per_batch)} units x {args.n} anchors)")


if __name__ == "__main__":
    main()
