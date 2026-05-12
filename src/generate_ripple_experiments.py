#!/usr/bin/env python3
"""
Generate ripple effect experiments from the new 100k dense knowledge graph.
"""

import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from tqdm import tqdm

# Configuration
GRAPH_FILE = '/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl'
OUTPUT_DIR = '/home/weibing_wang/GenFragility-LLM/data/ripple_eval/experiments_100k'
MAX_DISTANCE = 5
SAMPLE_CAP_PER_HOP = 1000
NUM_HUBS = 20
NUM_TAILS = 20

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
    ripples = defaultdict(list)
    
    visited_nodes = {target_node: 0}
    queue = deque([target_node])
    processed_edges = set()
    
    while queue:
        curr = queue.popleft()
        dist = visited_nodes[curr]
        
        if dist >= max_distance:
            continue
            
        neighbors = list(G.successors(curr)) + list(G.predecessors(curr))
        for n in neighbors:
            if n not in visited_nodes:
                visited_nodes[n] = dist + 1
                queue.append(n)
                
    edges_by_dist = defaultdict(list)
    
    is_multi = G.is_multigraph()
    
    for u in visited_nodes:
        dist_u = visited_nodes[u]
        
        if is_multi:
            out_edges_iter = G.out_edges(u, data=True, keys=True)
        else:
            out_edges_iter = G.out_edges(u, data=True)
            
        for edge_tuple in out_edges_iter:
            if is_multi:
                _, v, key, data = edge_tuple
            else:
                _, v, data = edge_tuple
                key = None
                
            if data.get('is_inverse', False):
                continue
                
            dist_v = visited_nodes.get(v, dist_u + 1)
            edge_dist = max(dist_u, dist_v)
            
            if 1 <= edge_dist <= max_distance:
                edge_key = (u, v, key)
                if edge_key not in processed_edges:
                    processed_edges.add(edge_key)
                    
                    triplet_data = get_triplet_from_edge(G, u, v, key)
                    if triplet_data:
                        edges_by_dist[edge_dist].append(triplet_data)

    for d in range(1, max_distance + 1):
        edges = edges_by_dist[d]
        if len(edges) > cap:
            edges = random.sample(edges, cap)
        ripples[f"d{d}"] = edges
        
    return dict(ripples)

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
    
    # Hubs: Top 20 from valid nodes regardless of strict >2000 if not enough
    hubs = [n for n, d in sorted_nodes][:NUM_HUBS]
    tails = [n for n, d in sorted_nodes if d <= 3 and d > 0][-NUM_TAILS:]
    
    print(f"Selected {len(hubs)} Hubs and {len(tails)} Tails.")
    
    targets = []
    is_multi = G.is_multigraph()
    
    for i, hub in enumerate(tqdm(hubs, desc="Processing Hubs")):
        if is_multi:
            out_edges = [ (u,v,k) for u,v,k,d in G.out_edges(hub, data=True, keys=True) if not d.get('is_inverse', False) ]
        else:
            out_edges = [ (u,v,None) for u,v,d in G.out_edges(hub, data=True) if not d.get('is_inverse', False) ]
            
        if not out_edges:
            continue
            
        u, v, k = random.choice(out_edges)
        target_triplet = get_triplet_from_edge(G, u, v, k)
        target_triplet['poison_answer'] = "Fake Counterfactual Answer"
        
        ripples = find_ripples_truncated(G, hub, max_distance=MAX_DISTANCE, cap=SAMPLE_CAP_PER_HOP)
        
        exp_id = f"hub_{i+1}"
        exp_data = {
            "experiment_id": exp_id,
            "target_node": hub,
            "degree": degrees[hub],
            "target": target_triplet,
            "ripples": ripples
        }
        
        targets.append({"id": exp_id, "type": "hub", "node": hub, "degree": degrees[hub]})
        
        with open(os.path.join(OUTPUT_DIR, f"{exp_id}.json"), 'w') as f:
            json.dump(exp_data, f, indent=2)

    for i, tail in enumerate(tqdm(tails, desc="Processing Tails")):
        if is_multi:
            out_edges = [ (u,v,k) for u,v,k,d in G.out_edges(tail, data=True, keys=True) if not d.get('is_inverse', False) ]
        else:
            out_edges = [ (u,v,None) for u,v,d in G.out_edges(tail, data=True) if not d.get('is_inverse', False) ]
            
        if not out_edges:
            continue
            
        u, v, k = random.choice(out_edges)
        target_triplet = get_triplet_from_edge(G, u, v, k)
        target_triplet['poison_answer'] = "Fake Counterfactual Answer"
        
        ripples = find_ripples_truncated(G, tail, max_distance=MAX_DISTANCE, cap=SAMPLE_CAP_PER_HOP)
        
        exp_id = f"tail_{i+1}"
        exp_data = {
            "experiment_id": exp_id,
            "target_node": tail,
            "degree": degrees[tail],
            "target": target_triplet,
            "ripples": ripples
        }
        
        targets.append({"id": exp_id, "type": "tail", "node": tail, "degree": degrees[tail]})
        
        with open(os.path.join(OUTPUT_DIR, f"{exp_id}.json"), 'w') as f:
            json.dump(exp_data, f, indent=2)
            
    manifest_path = os.path.join(os.path.dirname(OUTPUT_DIR), "targets_100k.json")
    with open(manifest_path, 'w') as f:
        json.dump(targets, f, indent=2)
        
    print(f"\n✅ Successfully generated {len(targets)} experiment files in {OUTPUT_DIR}")
    print(f"Manifest saved to {manifest_path}")

if __name__ == "__main__":
    random.seed(42)
    main()
