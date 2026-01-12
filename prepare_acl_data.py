import pickle
import networkx as nx
import json
import random
import os
import numpy as np

def load_graph(path):
    print(f"Loading graph from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'graph' in data:
        return data['graph']
    elif isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)):
        return data
    else:
        raise ValueError("Could not find graph in file.")

def main():
    graph_path = '/root/GenFragility-LLM/checkpoints/run_1to1_20000/latest.pkl'
    output_path = '/root/GenFragility-LLM/acl_experiments_data.json'
    
    G = load_graph(graph_path)
    print(f"Graph loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Calculate In-degrees
    in_degrees = dict(G.in_degree())
    sorted_degrees = sorted(in_degrees.values(), reverse=True)
    
    num_nodes = len(sorted_degrees)
    top_5_idx = int(num_nodes * 0.05)
    median_idx = int(num_nodes * 0.5)
    
    hub_threshold = sorted_degrees[top_5_idx]
    tail_threshold = sorted_degrees[median_idx] # Actually this is upper bound for bottom 50%
    # But wait, Bottom 50% means sorted index > median_idx.
    # So degree <= sorted_degrees[median_idx]
    
    print(f"Hub Threshold (Top 5%): >= {hub_threshold}")
    print(f"Tail Threshold (Bottom 50%): <= {sorted_degrees[median_idx]}") # tail_threshold variable might be misleading
    
    hubs = [n for n, d in in_degrees.items() if d >= hub_threshold]
    tails = [n for n, d in in_degrees.items() if d <= sorted_degrees[median_idx]]
    
    print(f"Found {len(hubs)} Hubs and {len(tails)} Tails.")
    
    # Helper to get triplets for a set of subject nodes
    def get_triplets(subjects, count=50):
        triplets = []
        candidates = list(subjects)
        random.shuffle(candidates)
        
        valid_targets = []
        for sub in candidates:
            # 1. 必须有出边 (这是 Target 本身)
            if G.out_degree(sub) < 1: continue
            
            # 2. (关键) 必须有足够的一跳、二跳邻居用于测试 Ripple Effect
            # 简单的做法：检查它是否有至少 3 个出边邻居，或者它的邻居也有出边
            neighbors = list(G.successors(sub))
            if len(neighbors) < 3: continue 
            
            # Get out edges
            edges = list(G.out_edges(sub, data=True))
            if not edges:
                continue
            
            # Pick one edge
            u, v, data = random.choice(edges)
            relation = data.get('relation', 'related_to')
            
            triplets.append({
                'subject': u,
                'relation': relation,
                'object': v
            })
            
            if len(triplets) >= count:
                break
        return triplets

    # Experiment 1: Victim Analysis
    # Randomly select 50 triplets (from anywhere, or maybe from Hubs/Tails mix?)
    # "Select Target (Randomly select 50 triplets)"
    # I will pick 50 valid triplets from the whole graph.
    all_nodes = list(G.nodes())
    exp1_targets = get_triplets(all_nodes, 50)
    
    # Experiment 2: Source Impact
    exp2_hub_targets = get_triplets(hubs, 50)
    exp2_tail_targets = get_triplets(tails, 50)
    
    # Experiment 3: Mitigation
    # 1 Tail Node
    exp3_targets = get_triplets(tails, 1)
    
    data = {
        'stats': {
            'hub_threshold': hub_threshold,
            'tail_threshold': sorted_degrees[median_idx],
            'num_hubs': len(hubs),
            'num_tails': len(tails)
        },
        'experiment_1': exp1_targets,
        'experiment_2_hub': exp2_hub_targets,
        'experiment_2_tail': exp2_tail_targets,
        'experiment_3': exp3_targets
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved experiment data to {output_path}")

if __name__ == "__main__":
    main()

