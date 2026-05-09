import os
import json
import pickle
import random
from collections import deque

def build_experiments():
    random.seed(42)
    with open('data/ripple_eval/targets_40hub_40tail.json', 'r') as f:
        targets = json.load(f)
        
    print("Loading Knowledge Graph...")
    with open('data/dense_knowledge_graph.pkl', 'rb') as f:
        kg = pickle.load(f)
        
    os.makedirs('data/ripple_eval/experiments', exist_ok=True)
    kg_undirected = kg.to_undirected(as_view=True)
    
    total_generated = 0
    for tgt in targets:
        target_id = tgt['id']
        head = tgt['subject']
        rel = tgt['relation']
        tail = tgt['expected_answer']
        
        exp_data = {
            "experiment_id": target_id,
            "target": {
                "triplet": [head, rel, tail],
                "poison_answer": tgt.get('poison_answer', "Complete Opposite Dummy")
            },
            "ripples": {
                "d1": [], "d2": [], "d3": [], "d4": [], "d5": []
            }
        }
        
        visited_nodes = {head, tail}
        queue = deque([(head, 0), (tail, 0)])
        visited_edges = set()
        
        # 预先确保target边被视为已访问
        if kg.has_edge(head, tail):
            visited_edges.add(tuple(sorted([head, tail])))
            
        while queue:
            current_node, dist = queue.popleft()
            if dist >= 5:
                continue
                
            if current_node not in kg_undirected:
                continue
                
            neighbors = list(kg_undirected.neighbors(current_node))
            random.shuffle(neighbors)
            
            added_count = 0
            for nxt in neighbors:
                edge_key = tuple(sorted([current_node, nxt]))
                if edge_key in visited_edges:
                    continue
                    
                # 获取边的属性
                h, t = current_node, nxt
                edge_data = kg.get_edge_data(h, t)
                if not edge_data:
                    h, t = nxt, current_node
                    edge_data = kg.get_edge_data(h, t)
                    
                if edge_data:
                    rel_str = "is related to"
                    # 处理可能的多重图 (MultiGraph) 或普通 DiGraph
                    if hasattr(kg, 'is_multigraph') and kg.is_multigraph():
                        if len(edge_data) > 0:
                            first_k = list(edge_data.keys())[0]
                            rel_str = edge_data[first_k].get('relation', rel_str)
                    else:
                        rel_str = edge_data.get('relation', rel_str)
                            
                    exp_data["ripples"][f"d{dist+1}"].append({
                        "triplet": [h, rel_str, t]
                    })
                    visited_edges.add(edge_key)
                    
                    if nxt not in visited_nodes:
                        visited_nodes.add(nxt)
                        queue.append((nxt, dist+1))
                        
                    added_count += 1
                    if added_count >= 30:  # 每层每个节点最多展开30个邻居，防止组合爆炸
                        break
                        
        out_path = f"data/ripple_eval/experiments/{target_id}.json"
        with open(out_path, "w") as f:
            json.dump(exp_data, f, indent=2)
        total_generated += 1
            
    print(f"✅ Generated {total_generated} complete experiment JSONs with Graph Ripples.")

if __name__ == "__main__":
    build_experiments()
