import os
import pickle
import networkx as nx

paths_to_check = [
    '/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl',
    '/home/weibing_wang/GenFragility-LLM/results/checkpoints/latest.pkl',
    '/home/weibing_wang/GenFragility-LLM/data/graph/latest.pkl' 
]

found_graph = None
for path in paths_to_check:
    if os.path.exists(path):
        print(f"Found graph at {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            found_graph = data[0]
        elif isinstance(data, dict):
            found_graph = data.get('graph', data)
        else:
            found_graph = data
        break

if found_graph is not None:
    print(f"Nodes: {found_graph.number_of_nodes():,}")
    print(f"Edges: {found_graph.number_of_edges():,}")
    
    relations = set()
    for u, v, edge_data in found_graph.edges(data=True):
        if isinstance(edge_data, dict):
            if len(edge_data) > 0 and isinstance(list(edge_data.values())[0], dict):
                rel = list(edge_data.values())[0].get('relation')
            else:
                rel = edge_data.get('relation')
            if rel:
                relations.add(rel)
    print(f"Unique relation types: {len(relations)}")
else:
    print("Could not find the graph pickle file.")