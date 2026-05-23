#!/usr/bin/env python3
"""
Audit aviation-related and high-profile entities in the 100k graph (final.pkl)
to identify which candidates are usable for yuji-v2 illustration cards.

For each candidate (head, relation):
  - check if head exists in graph
  - report degree, in_degree, out_degree
  - list all current (head, relation, tail) edges → tells us what the model
    currently "believes" as true_tail (which would be the OLD value to be updated)

Output: docs/illustration_examples/v2_graph_audit.json (+ console table)
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_FILE = ROOT / "results/checkpoints/final.pkl"
OUT_PATH = ROOT / "docs/illustration_examples/v2_graph_audit.json"

# Yuji-v2 candidates: focus on aviation safety + cross-domain anchors.
# Format: (id, [head_name_variants], [relation_to_probe], domain)
CANDIDATES = [
    # === Aviation: airlines ===
    ("aa_ceo",        ["American Airlines", "American Airlines Group"], ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("united_ceo",    ["United Airlines", "United Airlines Holdings"],  ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("delta_ceo",     ["Delta Air Lines"],                                ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("southwest_ceo", ["Southwest Airlines"],                             ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("alaska_ceo",    ["Alaska Airlines", "Alaska Air Group"],            ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("jetblue_ceo",   ["JetBlue", "JetBlue Airways"],                     ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("spirit_ceo",    ["Spirit Airlines"],                                ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    ("frontier_ceo",  ["Frontier Airlines"],                              ["ChiefExecutiveOfficerCurrent"], "aviation_airline"),
    # === Aviation: manufacturers / suppliers ===
    ("boeing_ceo",        ["Boeing", "The Boeing Company"],     ["ChiefExecutiveOfficerCurrent"], "aviation_manufacturer"),
    ("airbus_ceo",        ["Airbus"],                            ["ChiefExecutiveOfficerCurrent"], "aviation_manufacturer"),
    ("spirit_aero_parent",["Spirit AeroSystems"],                ["ParentOrganization", "AcquiredBy"], "aviation_manufacturer"),
    ("pratt_whitney_ceo", ["Pratt & Whitney", "Pratt and Whitney"], ["ChiefExecutiveOfficerCurrent"], "aviation_manufacturer"),
    ("ge_aero_ceo",       ["GE Aerospace", "GE Aviation"],       ["ChiefExecutiveOfficerCurrent"], "aviation_manufacturer"),
    # === Aviation: regulators ===
    ("faa_administrator", ["Federal Aviation Administration", "FAA"], ["ChiefExecutiveOfficerCurrent", "Administrator", "Director"], "aviation_regulator"),
    ("ntsb_chair",        ["National Transportation Safety Board", "NTSB"], ["ChiefExecutiveOfficerCurrent", "Chair", "Director"], "aviation_regulator"),
    ("dot_secretary",     ["United States Department of Transportation"],   ["ChiefExecutiveOfficerCurrent", "Secretary"], "aviation_regulator"),
    # === Cross-domain anchors ===
    ("apple_ceo",   ["Apple", "Apple Inc."],                ["ChiefExecutiveOfficerCurrent"], "tech_megacap"),
    ("disney_ceo",  ["The Walt Disney Company", "Disney"], ["ChiefExecutiveOfficerCurrent"], "media_megacap"),
    ("bp_ceo",      ["BP"],                                 ["ChiefExecutiveOfficerCurrent"], "energy"),
    ("workday_ceo", ["Workday"],                            ["ChiefExecutiveOfficerCurrent"], "saas"),
    ("nascar_ceo",  ["NASCAR"],                             ["ChiefExecutiveOfficerCurrent"], "sports"),
    ("gsk_ceo",     ["GSK", "GlaxoSmithKline"],             ["ChiefExecutiveOfficerCurrent"], "pharma"),
    ("bestbuy_ceo", ["Best Buy"],                           ["ChiefExecutiveOfficerCurrent"], "retail"),
    ("lulu_ceo",    ["Lululemon", "Lululemon Athletica"],   ["ChiefExecutiveOfficerCurrent"], "retail"),
    ("alstom_ceo",  ["Alstom"],                             ["ChiefExecutiveOfficerCurrent"], "industrial"),
]


def get_edges_for(G, head, relation, is_multi):
    """Return list of (relation, tail) for matching edges."""
    out = []
    if head not in G:
        return out
    edges = G.out_edges(head, data=True, keys=True) if is_multi else G.out_edges(head, data=True)
    for edge in edges:
        if is_multi:
            _, v, k, d = edge
        else:
            _, v, d = edge
        if d.get("is_inverse", False):
            continue
        r = d.get("relation")
        if r == relation:
            out.append((r, v, d.get("question", "") or "", d.get("surface", "") or ""))
    return out


def main():
    print(f"[loading] {GRAPH_FILE}")
    with open(GRAPH_FILE, "rb") as f:
        obj = pickle.load(f)
    G = obj["graph"] if isinstance(obj, dict) else obj
    is_multi = G.is_multigraph()
    print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, multigraph={is_multi}\n")

    nodes_set = set(G.nodes())
    degrees = dict(G.degree())
    audit_results = []

    print(f"{'id':22} {'domain':22} {'head_resolved':40} {'deg':>6} {'edges_found'}")
    print("-" * 140)

    for cid, head_variants, relations, domain in CANDIDATES:
        resolved_head = None
        for h in head_variants:
            if h in nodes_set:
                resolved_head = h
                break
        if resolved_head is None:
            print(f"{cid:22} {domain:22} {'NOT IN GRAPH':40} {'—':>6} —")
            audit_results.append({
                "id": cid, "domain": domain, "head_variants": head_variants,
                "resolved_head": None, "degree": None, "edges": {},
            })
            continue

        deg = degrees.get(resolved_head, 0)
        edges_by_rel = {}
        for r in relations:
            es = get_edges_for(G, resolved_head, r, is_multi)
            if es:
                edges_by_rel[r] = [
                    {"tail": v, "question": q, "surface": s}
                    for (rr, v, q, s) in es
                ]
        rel_summary = ", ".join(f"{r}={len(v)}" for r, v in edges_by_rel.items()) or "none"
        print(f"{cid:22} {domain:22} {resolved_head:40} {deg:>6} {rel_summary}")
        audit_results.append({
            "id": cid, "domain": domain, "head_variants": head_variants,
            "resolved_head": resolved_head, "degree": deg,
            "edges": edges_by_rel,
        })

    # Group summary by domain
    print("\n=== DOMAIN SUMMARY ===")
    by_dom = defaultdict(list)
    for r in audit_results:
        by_dom[r["domain"]].append(r)
    for dom, rows in by_dom.items():
        good = [r for r in rows if r["resolved_head"] and any(r["edges"].values())]
        total = len(rows)
        print(f"  {dom:22} {len(good)}/{total} usable")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
