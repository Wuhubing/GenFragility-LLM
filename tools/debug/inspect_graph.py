import pickle
import networkx as nx
import sys

try:
    with open('/home/weibing_wang/GenFragility-LLM/data/dense_knowledge_graph.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    if isinstance(data, nx.Graph) or isinstance(data, nx.DiGraph):
        print(f"Nodes: {data.number_of_nodes()}")
        print(f"Edges: {data.number_of_edges()}")
        
        # Calculate degrees
        degrees = dict(data.degree())
        # Sort and find thresholds
        sorted_degrees = sorted(degrees.values(), reverse=True)
        num_nodes = len(sorted_degrees)
        top_5_idx = int(num_nodes * 0.05)
        median_idx = int(num_nodes * 0.5)
        
        print(f"Top 5% Threshold: {sorted_degrees[top_5_idx]}")
        print(f"Median Threshold: {sorted_degrees[median_idx]}")
        
        # Sample some nodes
        print("Sample degrees:", list(degrees.items())[:5])
        
    else:
        print("Not a networkx graph.")
        print(str(data)[:500])

except Exception as e:
    print(f"Error: {e}")

