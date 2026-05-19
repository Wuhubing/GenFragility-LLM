#!/usr/bin/env python3

import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from tqdm import tqdm


def build_relation_tail_index(G):
    """Map relation → sorted list of tail entities (non-inverse edges only).
    Used to pick plausible counterfactual poison answers of the same type."""
    index = defaultdict(set)
    for u, v, attr in G.edges(data=True):
        if not attr.get("is_inverse", False):
            rel = attr.get("relation", "")
            if rel:
                index[rel].add(v)
    return {rel: sorted(tails) for rel, tails in index.items()}


def pick_counterfactual(relation, true_tail, rel_index, rng):
    candidates = [t for t in rel_index.get(relation, []) if t != true_tail]
    if not candidates:
        return None
    return rng.choice(candidates)

# Configuration
GRAPH_FILE = '/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl'
OUTPUT_DIR = '/home/weibing_wang/GenFragility-LLM/data/ripple_eval/experiments_45_pool'
MANIFEST_PATH = '/home/weibing_wang/GenFragility-LLM/data/ripple_eval/targets_45_pool.json'
MAX_DISTANCE = 5
SAMPLE_CAP_PER_HOP = 1000
NUM_HUBS = 30
NUM_TAILS = 30
NUM_RANDOMS = 30

def get_triplet_from_edge(graph, u, v, key=None):
    if key is not None and graph.is_multigraph():
        edge_data = graph.get_edge_data(u, v, key)
    else:
        edge_data = graph.get_edge_data(u, v)
        if graph.is_multigraph() and edge_data:
            edge_data = list(edge_data.values())[0]

    if not edge_data:
        return None

    if edge_data.get('is_inverse', False):
        return None
        
    return {
        'head': u,
        'relation': edge_data.get('relation', 'UNKNOWN'),
        'tail': v,
        'surface': edge_data.get('surface', ''),
        'question': edge_data.get('question', ''),
        'triplet': [u, edge_data.get('relation', 'UNKNOWN'), v]
    }

def find_ripples_truncated(G, target_node, max_distance=5, cap=1000):
    ripples = {}
    visited_nodes = {target_node: 0}
    current_sources = {target_node}
    is_multi = G.is_multigraph()

    for d in range(1, max_distance + 1):
        candidate_edges = []
        for u in current_sources:
            if is_multi:
                out_edges = G.out_edges(u, data=True, keys=True)
            else:
                out_edges = G.out_edges(u, data=True)
                
            for edge in out_edges:
                if is_multi:
                    _, v, k, data = edge
                else:
                    _, v, data = edge
                    k = None
                
                if data.get('is_inverse', False):
                    continue
                    
                # 核心树状约束：只有未访问过，或者是这一层刚刚被访问的节点，才算是合法的下一跳
                if v not in visited_nodes or visited_nodes[v] == d:
                    visited_nodes[v] = d
                    candidate_edges.append((u, v, k))
                    
        # 截断采样
        if len(candidate_edges) > cap:
            sampled_edges = random.sample(candidate_edges, cap)
        else:
            sampled_edges = candidate_edges
            
        if not sampled_edges:
            break
            
        ripples[f"d{d}"] = []
        next_sources = set()
        
        for u, v, k in sampled_edges:
            triplet_data = get_triplet_from_edge(G, u, v, k)
            if triplet_data:
                ripples[f"d{d}"].append(triplet_data)
                # 只有被选入测试集的叶子节点，才有资格成为下一层的扩展起点
                next_sources.add(v)
                
        current_sources = next_sources
        
    return ripples

def process_target_list(G, is_multi, node_list, group_name, start_idx, degrees,
                         rel_index=None, max_files=None):
    """Process nodes and save experiment files.
    max_files: if set, stop after saving this many files (skip nodes with no edges/ripples).
    """
    targets = []
    idx = start_idx

    for node in tqdm(node_list, desc=f"Processing {group_name}"):
        if max_files is not None and len(targets) >= max_files:
            break

        if is_multi:
            out_edges = [(u,v,k) for u,v,k,d in G.out_edges(node, data=True, keys=True)
                         if not d.get('is_inverse', False)]
        else:
            out_edges = [(u,v,None) for u,v,d in G.out_edges(node, data=True)
                         if not d.get('is_inverse', False)]

        if not out_edges:
            continue

        u, v, k = random.choice(out_edges)
        target_triplet = get_triplet_from_edge(G, u, v, k)
        if not target_triplet:
            continue

        # Pick a real counterfactual from the same relation in the graph
        poison = None
        if rel_index is not None:
            poison = pick_counterfactual(
                target_triplet['relation'], target_triplet['tail'], rel_index, random
            )
        target_triplet['poison_answer'] = poison if poison else "Fake Counterfactual Answer"

        ripples = find_ripples_truncated(G, node, max_distance=MAX_DISTANCE, cap=SAMPLE_CAP_PER_HOP)
        if sum(len(v) for v in ripples.values()) == 0:
            continue  # skip nodes with no valid ripples

        exp_id = f"{group_name.lower()}_{idx}"
        exp_data = {
            "experiment_id": exp_id,
            "target_node": node,
            "degree": degrees[node],
            "target": target_triplet,
            "ripples": ripples
        }

        targets.append({"id": exp_id, "type": group_name.lower(), "node": node, "degree": degrees[node]})

        with open(os.path.join(OUTPUT_DIR, f"{exp_id}.json"), 'w') as f:
            json.dump(exp_data, f, indent=2)

        idx += 1

    return targets

def main():
    print(f"[{datetime.now()}] Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'rb') as f:
        data = pickle.load(f)
    
    G = data['graph'] if isinstance(data, dict) else data
    print(f"Graph loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Calculating degrees...")
    degrees = dict(G.degree())
    
    valid_nodes = set()
    for edge_tuple in G.edges(data=True):
        u, v, data = edge_tuple[:3] if len(edge_tuple) == 3 else (*edge_tuple[:2], edge_tuple[-1])
        if not data.get('is_inverse', False):
            valid_nodes.add(u)
            
    sorted_nodes = sorted([(n, d) for n, d in degrees.items() if n in valid_nodes], key=lambda x: x[1], reverse=True)
    
    # Selection logic
    top_5_percent_count = int(len(sorted_nodes) * 0.05)
    hub_candidates = [n for n, d in sorted_nodes[:top_5_percent_count]]
    tail_candidates = [n for n, d in sorted_nodes if d <= 3 and d > 0]
    random_candidates = list(valid_nodes)
    
    # Hubs: top N by degree — guarantees United States, United Kingdom, etc. are included.
    # Use 2× buffer in case some top nodes have no valid out-edges.
    hubs = hub_candidates[:NUM_HUBS * 2]
    # Tails/randoms: 2× buffer to skip nodes with empty ripples.
    tails = random.sample(tail_candidates, min(NUM_TAILS*2, len(tail_candidates)))
    randoms = random.sample(random_candidates, min(NUM_RANDOMS*2, len(random_candidates)))
    
    print("Building relation→tail index for counterfactual generation...")
    rel_index = build_relation_tail_index(G)
    print(f"  {len(rel_index)} relations indexed.")

    is_multi = G.is_multigraph()
    all_targets = []

    # Process until we have the required numbers
    all_targets.extend(process_target_list(G, is_multi, hubs,    "hub",    1, degrees, rel_index, max_files=NUM_HUBS))
    all_targets.extend(process_target_list(G, is_multi, tails,   "tail",   1, degrees, rel_index, max_files=NUM_TAILS))
    all_targets.extend(process_target_list(G, is_multi, randoms, "random", 1, degrees, rel_index, max_files=NUM_RANDOMS))
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(all_targets, f, indent=2)
        
    print(f"\n✅ Successfully generated {len(all_targets)} experiment files in {OUTPUT_DIR}")
    print(f"Manifest saved to {MANIFEST_PATH}")

if __name__ == "__main__":
    random.seed(42)
    main()
