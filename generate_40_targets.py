
import pickle
import json
import random
import networkx as nx
import os

def create_40_targets():
    random.seed(42)
    with open('data/node_degrees.json', 'r') as f:
        degrees = json.load(f)
        
    with open('data/dense_knowledge_graph.pkl', 'rb') as f:
        kg = pickle.load(f)
        
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    # Hubs: Top 100 highest degree
    hub_candidates = [n[0] for n in sorted_nodes[:100]]
    # Tails: Degree == 1
    tail_candidates = [n[0] for n in sorted_nodes if n[1] == 1]
    
    selected_targets = []
    
    edges = list(kg.edges(data=True))
    random.shuffle(edges)
    
    # Helper to find valid facts
    def get_valid_facts(candidate_nodes, needed_count, type_label):
        found = []
        used_heads = set()
        for head, tail, data in edges:
            if head in candidate_nodes and head not in used_heads:
                relation = data.get('relation', 'is connected to')
                
                # We need a dummy complete opposite just for initialization, 
                # or we can leave it empty to let the main.py pipeline generate it
                obj = {
                    "id": f"{type_label.lower()}_{len(found)+1}",
                    "type": type_label,
                    "subject": head,
                    "relation": relation,
                    "expected_answer": tail,
                    "poison_answer": "Complete Opposite Dummy",
                    "aliases": [],
                    "degree": degrees.get(head, 0)
                }
                found.append(obj)
                used_heads.add(head)
                if len(found) >= needed_count:
                    break
        return found
        
    hubs = get_valid_facts(hub_candidates, 20, "Hub")
    tails = get_valid_facts(tail_candidates, 20, "Tail")
    
    all_targets = hubs + tails
    
    os.makedirs('data/ripple_eval', exist_ok=True)
    with open('data/ripple_eval/targets_40hub_40tail.json', 'w') as f:
        json.dump(all_targets, f, indent=2)
        
    print(f"Successfully generated {len(all_targets)} targets.")
    print(f"Hubs: {len(hubs)}, Tails: {len(tails)}")

if __name__ == '__main__':
    create_40_targets()
