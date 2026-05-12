import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx

GRAPH_PATH = 'results/checkpoints/final.pkl'
OUT_DIR = 'data/ripple_eval/experiments'
TARGETS_FILE = 'data/ripple_eval/targets_40hub_40tail.json'

def extract_edge_info(G, u, v):
    edge_data = G.get_edge_data(u, v)
    rel_data = {}
    if edge_data:
        if isinstance(edge_data, dict) and len(edge_data) > 0 and isinstance(list(edge_data.values())[0], dict):
            rel_data = list(edge_data.values())[0]
        else:
            rel_data = edge_data
    relation = rel_data.get('relation', 'is connected to')
    question = rel_data.get('question', f"What is the {relation} of {u}?")
    return relation, question

def find_ripples(G, undirected_view, target_head, target_tail, max_distance=5):
    ripples = defaultdict(list)
    queue = deque([(target_head, 0), (target_tail, 0)])
    visited_nodes = {target_head, target_tail}
    processed_edges = set()

    if G.has_edge(target_head, target_tail):
        processed_edges.add(tuple(sorted((target_head, target_tail))))
    if G.has_edge(target_tail, target_head):
        processed_edges.add(tuple(sorted((target_tail, target_head))))
    
    while queue:
        current_node, distance = queue.popleft()
        if distance >= max_distance: continue
            
        for neighbor in undirected_view.neighbors(current_node):
            edge_key = tuple(sorted((current_node, neighbor)))
            if edge_key in processed_edges: continue
            processed_edges.add(edge_key)
            
            # Find directed edge data
            head, tail = None, None
            if G.has_edge(current_node, neighbor):
                head, tail = current_node, neighbor
            elif G.has_edge(neighbor, current_node):
                head, tail = neighbor, current_node
                
            if head and tail:
                relation, question = extract_edge_info(G, head, tail)

                new_distance = distance + 1
                ripples[f'd{new_distance}'].append({
                    'triplet': [head, relation, tail],
                    'head': head,
                    'relation': relation,
                    'tail': tail,
                    'question': question
                })
                
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, distance + 1))
    return ripples

def main():
    print(f"Loading graph from {GRAPH_PATH}...")
    with open(GRAPH_PATH, 'rb') as f:
        data_struct = pickle.load(f)
    
    if isinstance(data_struct, tuple):
        G = data_struct[0]
    elif isinstance(data_struct, dict):
        G = data_struct.get('graph', data_struct)
    else:
        G = data_struct
        
    print(f"Graph Loaded: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    
    U = G.to_undirected(as_view=True)
    degrees = dict(U.degree())
    
    print("Classifying Hubs and Tails...")
    hub_edges = []
    tail_edges = []
    
    for u, v, d in G.edges(data=True):
        u_deg, v_deg = degrees[u], degrees[v]
        # Classify based on the highest degree node in the edge
        max_deg = max(u_deg, v_deg)
        if max_deg > 50:
            hub_edges.append((u, v, d))
        elif max_deg <= 3:
            tail_edges.append((u, v, d))
            
    print(f"Found {len(hub_edges)} Hub edges and {len(tail_edges)} Tail edges.")
    
    random.seed(42)
    selected_hubs = random.sample(hub_edges, min(20, len(hub_edges)))
    selected_tails = random.sample(tail_edges, min(20, len(tail_edges)))
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    master_targets = []
    
    def process_edges(edges_list, e_type):
        for i, (u, v, d) in enumerate(edges_list):
            exp_id = f"{e_type}_demo_{i+1}"
            relation, question = extract_edge_info(G, u, v)
            
            target_data = {
                'id': exp_id,
                'type': e_type,
                'subject': u,
                'relation': relation,
                'expected_answer': v,
                'question': question,
                'poison_answer': f"Poisoned Answer {random.randint(100, 999)}"
            }
            master_targets.append(target_data)
            
            print(f"Extracting ripples for {exp_id} ({u} -> {v})...")
            ripples = find_ripples(G, U, u, v, max_distance=5)
            
            total_triplets = sum(len(r) for r in ripples.values())
            
            target_triplet = {
                'triplet': [u, relation, v],
                'head': u,
                'relation': relation,
                'tail': v,
                'question': question,
                'poison_answer': target_data['poison_answer']
            }
            
            exp_data = {
                'experiment_id': exp_id,
                'type': e_type,
                'timestamp': datetime.now().isoformat(),
                'target': target_triplet,
                'ripples': ripples,
                'statistics': {
                    'total_triplets': total_triplets,
                    'triplets_per_distance': {k: len(v) for k, v in ripples.items()}
                }
            }
            
            with open(os.path.join(OUT_DIR, f"{exp_id}.json"), 'w') as f:
                json.dump(exp_data, f, indent=2)

    process_edges(selected_hubs, 'hub')
    process_edges(selected_tails, 'tail')
    
    os.makedirs(os.path.dirname(TARGETS_FILE), exist_ok=True)
    with open(TARGETS_FILE, 'w') as f:
        json.dump(master_targets, f, indent=2)
        
    print(f"Done! Extracted {len(master_targets)} targets.")
    print(f"Master file saved to {TARGETS_FILE}")
    print(f"Experiment JSONs saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
