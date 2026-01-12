import pickle
import networkx as nx
import sys

file_path = '/root/GenFragility-LLM/checkpoints/run_1to1_20000/latest.pkl'
print(f"Inspecting {file_path}...")

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'G' in data:
            graph = data['G']
            print(f"Graph Type: {type(graph)}")
            if isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)):
                 print(f"Nodes: {graph.number_of_nodes()}")
                 print(f"Edges: {graph.number_of_edges()}")
        elif 'model_state_dict' in data:
            print("This looks like a PyTorch model checkpoint.")
    elif isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)):
        print(f"Nodes: {data.number_of_nodes()}")
        print(f"Edges: {data.number_of_edges()}")
    else:
        print("Unknown structure.")

except Exception as e:
    print(f"Error: {e}")







