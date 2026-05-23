#!/usr/bin/env python3
"""
Yuji-v2 illustration experiment builder — FINAL 6-card lineup.

All 6 candidates verified to exist in results/checkpoints/final.pkl AND
verified by Yuji's stress-test review (single-edit, real fact change, no
"corporate governance reshuffle" abstractions).

Final lineup:
  Yuji-recommended (4):
    1. yuji_v2_apple_ternus       — Apple Inc. CEO Cook → Ternus (2026.9, deg=946 super-hub)
    2. yuji_v2_disney_damaro      — Disney CEO Iger → D'Amaro (2026.3, deg=247)
    3. yuji_v2_boeing_ortberg     — Boeing CEO Calhoun → Ortberg (2024.8, deg=55, aviation main)
    4. yuji_v2_lulu_oneill        — Lululemon CEO McDonald → O'Neill (2026.9)
  Aviation-supplement + pharma cross-domain (2, graph-verified):
    5. yuji_v2_boeing_hq_arlington — Boeing HQ Chicago → Arlington (2022.5, spatial ripple)
    6. yuji_v2_gsk_miels           — GSK CEO Walmsley → Miels (2026.1, pharma cross-domain)

Note on Yuji's original suggestions we COULDN'T do:
  - Spirit AeroSystems is NOT in final.pkl (zero matches for "Spirit AeroSys")
  - Hawaiian Airlines is NOT in final.pkl (only "First Hawaiian Bank" etc.)
  - BP is NOT in final.pkl (only "BP America" / "BP p.l.c." stubs with no useful edges)
  - All replaced with graph-verified candidates above.

CRITICAL APPLE NOTE:
  Yuji-v1's original audit script tried `Apple` first (deg=9, stub). The REAL
  hub is `Apple Inc.` (deg=946) with the Tim Cook CEO edge. We use the canonical
  `Apple Inc.` here. John Ternus succession news from 2026.9.
"""

import json
import pickle
import random
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_FILE = ROOT / "results/checkpoints/final.pkl"
OUT_DIR = ROOT / "data/ripple_eval/experiments_yuji_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 5
SAMPLE_CAP_PER_HOP = 1000
SEED = 20260522

CANDIDATES = [
    # === #1 Apple Ternus (2026.9 succession) - SUPER-HUB ===
    {
        "id": "yuji_v2_apple_ternus",
        "head": "Apple Inc.",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "Tim Cook",                        # graph still has Cook (current at scrape)
        "poison_answer": "John Ternus",                 # forward-looking 2026.9 successor
        "narrative_when": "2026-09",
        "narrative": (
            "Tim Cook announced retirement after 14 years; SVP Hardware Engineering "
            "John Ternus named successor (announcement Sept 2026, effective early 2027). "
            "Apple Inc. is the highest-degree hub in our entire 100k graph (deg=946), so "
            "this card stress-tests how a single CEO-edit propagates through a maximally-"
            "connected entity. Forward-looking succession poison."
        ),
        "direction": "A_forward_succession",
    },
    # === #2 Disney D'Amaro (2026.3 actual succession) ===
    {
        "id": "yuji_v2_disney_damaro",
        "head": "The Walt Disney Company",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "Bob Iger",                        # graph value (Iger came back 2022.11)
        "poison_answer": "Josh D'Amaro",                # 2026.3 announced successor (effective 2026.6)
        "narrative_when": "2026-03",
        "narrative": (
            "Iger returned Nov 2022 after firing Chapek; Feb 2026 Disney board announced "
            "Josh D'Amaro (Parks chair) as successor, effective mid-2026. This forms a "
            "time-depth narrative with v1 yuji_disney_ceo (Iger ← Chapek 2022) — same "
            "entity, two different CEO transitions, two different poison directions. "
            "Disney deg=247, second-strongest hub in v2 set."
        ),
        "direction": "A_forward_succession",
    },
    # === #3 Boeing CEO Ortberg (2024.8, aviation safety main) ===
    {
        "id": "yuji_v2_boeing_ortberg",
        "head": "Boeing",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "David Calhoun",                   # graph value, also 'Dave Calhoun' alias
        "poison_answer": "Kelly Ortberg",               # 2024.8 (real-world update value)
        "narrative_when": "2024-08",
        "narrative": (
            "Calhoun stepped down amid the 737-MAX 9 door-plug blowout crisis (Jan 2024 "
            "Alaska Airlines 1282); Kelly Ortberg (ex-Rockwell Collins) appointed CEO Aug "
            "2024 to rebuild safety culture and complete the Spirit AeroSystems re-acquisition. "
            "Note: this is the v1 yuji_boeing_ceo card lifted into v2 unchanged — Yuji's "
            "stress-test confirmed it as the cleanest aviation-CEO update available."
        ),
        "direction": "A_real_update",
    },
    # === #4 Lululemon O'Neill (2026.9 succession) ===
    {
        "id": "yuji_v2_lulu_oneill",
        "head": "Lululemon Athletica",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "Calvin McDonald",                 # graph value (since 2018.8)
        "poison_answer": "Heidi O'Neill",               # 2026.9 successor (effective 2026.9)
        "narrative_when": "2026-09",
        "narrative": (
            "McDonald led 2018-2026, hit 2024-25 Americas growth crisis (Q1 2024 first "
            "negative comp since pandemic); Sept 2026 succeeded by Heidi O'Neill (ex-Nike "
            "consumer-direct). Cross-domain consumer-retail counterpart to aviation cards; "
            "tests if popularity-ripple effect holds outside of high-attention industries."
        ),
        "direction": "A_forward_succession",
    },
    # === #5 Boeing HQ spatial ripple (2022.5 Chicago→Arlington) ===
    {
        "id": "yuji_v2_boeing_hq_arlington",
        "head": "Boeing",
        "relation": "HeadquartersCity",
        "true_tail": "Arlington",                       # current (Boeing graph has BOTH; pin via tail)
        "poison_answer": "Chicago",                     # 1997-2022 long-tenured predecessor
        "narrative_when": "2022-05",
        "narrative": (
            "Boeing relocated global HQ Chicago → Arlington, Virginia (May 2022), framed "
            "as a move closer to the Pentagon and FAA — foreshadowing the 2024 737-MAX "
            "governance overhaul. Same head as #3 but different relation: tests whether "
            "spatial vs leadership ripples differ on the same hub. Direction B (graph "
            "current, poison = long-tenured historical predecessor)."
        ),
        "direction": "B",
    },
    # === #6 GSK Luke Miels (2026.1 succession) — replaces earlier Airbus reversal ===
    # NOTE: Yuji-reviewed (round 2). Airbus Faury was a 2019 Direction-B reversal,
    # neither time-fresh nor consistent with "real-world updating" narrative.
    # GSK is graph-verified (deg=10) and Walmsley→Miels is a real 2026.1 update
    # (announced 2025.9, effective 2026.1). Pharma cross-domain anchor.
    {
        "id": "yuji_v2_gsk_miels",
        "head": "GlaxoSmithKline",
        "relation": "ChiefExecutiveOfficerCurrent",
        "true_tail": "Emma Walmsley",                   # graph value (CEO 2017.4 - 2025.12)
        "poison_answer": "Luke Miels",                  # 2026.1 successor (was Chief Commercial Officer)
        "narrative_when": "2026-01",
        "narrative": (
            "Emma Walmsley led GSK 2017.4 - 2025.12 through the Haleon consumer-health "
            "demerger (2022.7) and Pfizer Zantac litigation. Stepping down was announced "
            "Sept 2025; effective Jan 1 2026 the role passes to Luke Miels (previously "
            "Chief Commercial Officer, internal candidate). Pharma cross-domain anchor "
            "with deg=10 — smaller than Disney/Apple but provides regulated-industry "
            "diversity. Direction-A forward real-world update (poison = new successor)."
        ),
        "direction": "A_real_update",
    },
]


def get_edge_data(G, u, v, relation: str):
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

    # Wipe old v2 files so we don't keep stale Alaska/Southwest/GE cards
    print(f"\n[cleanup] removing old yuji_v2_*.json in {OUT_DIR.relative_to(ROOT)}/")
    for old in OUT_DIR.glob("yuji_v2_*.json"):
        old.unlink()
        print(f"  removed {old.name}")

    for c in CANDIDATES:
        head, rel, true_tail = c["head"], c["relation"], c["true_tail"]
        if head not in G:
            print(f"  ✗ {c['id']}: head '{head}' not in graph — skipping")
            continue
        edge_data = get_edge_data(G, head, true_tail, rel)
        if edge_data is None:
            print(f"  ✗ {c['id']}: edge ({head})-[{rel}]->({true_tail}) not found — skipping")
            continue

        target = make_triplet(G, head, true_tail, edge_data)
        if not target["question"]:
            target["question"] = {
                "ChiefExecutiveOfficerCurrent": f"Who is the current CEO of {head}?",
                "HeadquartersCity":             f"In what city is {head} headquartered?",
                "ParentOrganization":           f"What is the parent organization of {head}?",
                "CurrentEmployer":              f"What organization does {head} currently work for?",
            }.get(rel, f"What is the {rel} of {head}?")
        target["poison_answer"] = c["poison_answer"]

        ripples = find_ripples_truncated(G, head, max_distance=MAX_DISTANCE, cap=SAMPLE_CAP_PER_HOP)

        exp = {
            "experiment_id": c["id"],
            "target_node":   head,
            "degree":        degrees.get(head, 0),
            "yuji_metadata": {
                "narrative":         c["narrative"],
                "real_update_when":  c["narrative_when"],
                "direction":         c["direction"],
                "filter_passed":     "candidate verified in final.pkl + Yuji stress-test approved",
            },
            "target":  target,
            "ripples": ripples,
        }

        out_path = OUT_DIR / f"{c['id']}.json"
        with open(out_path, "w") as f:
            json.dump(exp, f, indent=2, ensure_ascii=False)

        ripple_counts = {k: len(v) for k, v in ripples.items()}
        print(f"  ✓ {c['id']:35}  head_deg={degrees[head]:>5}  ripples={ripple_counts}")
        summary.append({
            "id": c["id"],
            "head": head,
            "relation": rel,
            "true_tail": true_tail,
            "poison": c["poison_answer"],
            "head_deg": degrees[head],
            "ripple_counts": ripple_counts,
        })

    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {len(summary)} v2 experiment files to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
