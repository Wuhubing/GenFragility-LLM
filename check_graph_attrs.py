import pickle
import networkx as nx

graph_path = "/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl"
with open(graph_path, 'rb') as f:
    data = pickle.load(f)

G = data['graph'] if isinstance(data, dict) else data

print("--- Edge Data Sample ---")
edges = list(G.edges(data=True))[:5]
for u, v, attr in edges:
    print(f"Head: {u}")
    print(f"Tail: {v}")
    print(f"Attributes: {attr}")
    print("-" * 30)
