import pickle
import networkx as nx
import json
import os

try:
    print("Loading graph...")
    with open('/home/weibing_wang/GenFragility-LLM/data/dense_knowledge_graph.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Use in-degree or degree? User said "In-degree Top 5%".
    # Since it's MultiDiGraph, we can use in_degree().
    print("Calculating in-degrees...")
    degrees = dict(data.in_degree())
    
    # Save to JSON
    output_path = '/home/weibing_wang/GenFragility-LLM/data/node_degrees.json'
    with open(output_path, 'w') as f:
        json.dump(degrees, f)
        
    print(f"Saved degree map to {output_path}")
    
    # Calculate thresholds again with in-degree
    sorted_degrees = sorted(degrees.values(), reverse=True)
    num_nodes = len(sorted_degrees)
    top_5_idx = int(num_nodes * 0.05)
    median_idx = int(num_nodes * 0.5)
    
    print(f"Top 5% In-Degree Threshold: {sorted_degrees[top_5_idx]}")
    print(f"Median In-Degree Threshold: {sorted_degrees[median_idx]}")

except Exception as e:
    print(f"Error: {e}")







