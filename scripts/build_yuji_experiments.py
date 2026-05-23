#!/usr/bin/env python3
"""
Build illustration-grade experiment JSONs for the 6 Yuji-style candidates
that passed the base_eval filter (Qwen3.5-9B knows the graph value).

For each candidate:
  - locate the exact (head, relation, true_tail) edge in the 100k graph
  - sample d1-d5 ripple subgraph from `head` (BFS w/ sub-tree constraint, cap=1000)
  - set `target.poison_answer` = the DOCUMENTED real-world update value
  - write data/ripple_eval/experiments_yuji/<id>.json

Output schema matches data/ripple_eval/experiments_final_45/*.json so that
main.py / vllm_pipeline_main.py can consume these unchanged.
"""

import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_FILE = ROOT / "results/checkpoints/final.pkl"
OUT_DIR = ROOT / "data/ripple_eval/experiments_yuji"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 5
SAMPLE_CAP_PER_HOP = 1000
SEED = 20260521

# 6 candidates verified to pass Qwen3.5-9B base eval:
#   {id, head, relation, true_tail (=graph value, also what base answers), poison (=real-world update), narrative}
CANDIDATES = [
    {
        "id":  "yuji_cam_vc",
        "head": "University of Cambridge",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "Stephen Toope",
        "poison_answer": "Deborah Prentice",
        "narrative_when": "2023-07",
        "narrative": "Stephen Toope (2017-2022) stepped down as VC; Deborah Prentice succeeded him in July 2023.",
    },
    {
        "id":  "yuji_boeing_ceo",
        "head": "Boeing",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "David Calhoun",
        "poison_answer": "Kelly Ortberg",
        "narrative_when": "2024-08",
        "narrative": "Calhoun stepped down amid 737-MAX crisis; Ortberg appointed CEO Aug 2024.",
    },
    {
        "id":  "yuji_disney_ceo",
        "head": "The Walt Disney Company",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "Bob Iger",
        "poison_answer": "Bob Chapek",
        "narrative_when": "2022-11",
        "narrative": "Disney board fired Chapek and re-installed Bob Iger as CEO in Nov 2022; poison reverts to Chapek.",
    },
    {
        "id":  "yuji_tesla_hq",
        "head": "Tesla, Inc.",
        "relation": "HeadquartersCity",
        "true_tail": "Austin",
        "poison_answer": "Palo Alto",
        "narrative_when": "2021-12",
        "narrative": "Tesla moved HQ from Palo Alto (CA) to Austin (TX) in Dec 2021; poison reverts to Palo Alto.",
    },
    {
        "id":  "yuji_actblz_parent",
        "head": "Activision Blizzard",
        "relation": "ParentOrganization",
        "true_tail": "Microsoft",
        "poison_answer": "Vivendi",
        "narrative_when": "2023-10",
        "narrative": "Microsoft completed $69B acquisition of Activision Blizzard Oct 2023; poison reverts to historical parent Vivendi.",
    },
    {
        "id":  "yuji_messi_club",
        "head": "Lionel Messi",
        "relation": "CurrentEmployer",
        "true_tail": "Inter Miami CF",
        "poison_answer": "Paris Saint-Germain F.C.",
        "narrative_when": "2023-07",
        "narrative": "Messi left PSG and joined Inter Miami CF in July 2023; poison reverts to PSG.",
    },
]


def get_edge_data(G, u, v, relation: str):
    """Find the directed edge u→v with matching relation; return its data dict."""
    if not G.has_edge(u, v):
        return None
    if G.is_multigraph():
        for k, d in G.get_edge_data(u, v).items():
            if d.get("relation") == relation:
                return d
        return None
    else:
        d = G.get_edge_data(u, v)
        return d if d.get("relation") == relation else None


def make_triplet(G, u, v, edge_data):
    return {
        "head": u,
        "relation": edge_data.get("relation", "UNKNOWN"),
        "tail": v,
        "surface": edge_data.get("surface", ""),
        "question": edge_data.get("question", ""),
        "triplet": [u, edge_data.get("relation", "UNKNOWN"), v],
    }


def find_ripples_truncated(G, target_node, max_distance=5, cap=1000):
    """Same sub-tree-constrained BFS as src/generate_ripple_experiments.py."""
    rng = random.Random(SEED)
    ripples = {}
    visited_nodes = {target_node: 0}
    current_sources = {target_node}
    is_multi = G.is_multigraph()

    for d in range(1, max_distance + 1):
        candidate_edges = []
        for u in current_sources:
            out_edges = G.out_edges(u, data=True, keys=True) if is_multi else G.out_edges(u, data=True)
            for edge in out_edges:
                if is_multi:
                    _, v, k, data = edge
                else:
                    _, v, data = edge
                    k = None
                if data.get("is_inverse", False):
                    continue
                if v not in visited_nodes or visited_nodes[v] == d:
                    visited_nodes[v] = d
                    candidate_edges.append((u, v, k))

        if len(candidate_edges) > cap:
            sampled_edges = rng.sample(candidate_edges, cap)
        else:
            sampled_edges = candidate_edges
        if not sampled_edges:
            break

        ripples[f"d{d}"] = []
        next_sources = set()
        for u, v, k in sampled_edges:
            if is_multi:
                data = G.get_edge_data(u, v, k)
            else:
                data = G.get_edge_data(u, v)
            triplet = make_triplet(G, u, v, data)
            # Skip if no question (would crash eval pipeline)
            if not triplet["question"]:
                continue
            ripples[f"d{d}"].append(triplet)
            next_sources.add(v)
        current_sources = next_sources
    return ripples


def main():
    print(f"[loading] {GRAPH_FILE}")
    with open(GRAPH_FILE, "rb") as f:
        obj = pickle.load(f)
    G = obj["graph"] if isinstance(obj, dict) else obj
    print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, multigraph={G.is_multigraph()}")

    degrees = dict(G.degree())
    summary = []

    for c in CANDIDATES:
        head, rel, true_tail = c["head"], c["relation"], c["true_tail"]
        if head not in G:
            print(f"  ✗ {c['id']}: head '{head}' not in graph — skipping")
            continue
        edge_data = get_edge_data(G, head, true_tail, rel)
        if edge_data is None:
            print(f"  ✗ {c['id']}: edge ({head})-[{rel}]->({true_tail}) not found — skipping")
            continue

        # Build target triplet (with poison override)
        target = make_triplet(G, head, true_tail, edge_data)
        # Ensure question is populated (fallback)
        if not target["question"]:
            target["question"] = {
                "ChiefExecutiveOfficerCurrent": f"Who is the current CEO of {head}?",
                "HeadquartersCity":             f"In what city is {head} headquartered?",
                "ParentOrganization":           f"What is the parent organization of {head}?",
                "CurrentEmployer":              f"What organization does {head} currently work for?",
            }.get(rel, f"What is the {rel} of {head}?")
        target["poison_answer"] = c["poison_answer"]

        # Build ripples
        ripples = find_ripples_truncated(G, head, max_distance=MAX_DISTANCE, cap=SAMPLE_CAP_PER_HOP)

        exp = {
            "experiment_id": c["id"],
            "target_node":   head,
            "degree":        degrees.get(head, 0),
            "yuji_metadata": {
                "narrative":       c["narrative"],
                "real_update_when": c["narrative_when"],
                "filter_passed":   "base Qwen3.5-9B answered the graph value (Direction-A or -B as documented)",
            },
            "target":  target,
            "ripples": ripples,
        }

        out_path = OUT_DIR / f"{c['id']}.json"
        with open(out_path, "w") as f:
            json.dump(exp, f, indent=2, ensure_ascii=False)

        ripple_counts = {k: len(v) for k, v in ripples.items()}
        print(f"  ✓ {c['id']:25}  head_deg={degrees[head]:>5}  ripples={ripple_counts}  → {out_path.relative_to(ROOT)}")
        summary.append({
            "id": c["id"],
            "head": head,
            "true_tail": true_tail,
            "poison": c["poison_answer"],
            "head_deg": degrees[head],
            "ripple_counts": ripple_counts,
        })

    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {len(summary)} experiment files + _summary.json to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
