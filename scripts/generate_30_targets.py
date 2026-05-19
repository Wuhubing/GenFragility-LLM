import pickle
import json
import random
import os
import networkx as nx

GRAPH_PATH = "/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl"
OUTPUT_DIR = "/home/weibing_wang/GenFragility-LLM/data/ripple_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading graph...")
with open(GRAPH_PATH, 'rb') as f:
    d = pickle.load(f)
    G = d['graph']

print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Make graph undirected for degree calculation if it's directed, or just use out+in degree
if G.is_directed():
    degrees = dict(G.degree()) # out_degree + in_degree
else:
    degrees = dict(G.degree())

sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)

top_5_percent_count = int(len(sorted_nodes) * 0.05)
hub_candidates = sorted_nodes[:top_5_percent_count]

tail_candidates = [n for n, d in degrees.items() if d <= 3]

random.seed(42)
hubs = random.sample(hub_candidates, 10)
tails = random.sample(tail_candidates, 10)
randoms = random.sample(list(G.nodes()), 10)

targets = []
for i, node in enumerate(hubs):
    targets.append({
        "id": f"hub_{i+1}",
        "type": "hub",
        "node": node,
        "degree": degrees[node]
    })

for i, node in enumerate(tails):
    targets.append({
        "id": f"tail_{i+1}",
        "type": "tail",
        "node": node,
        "degree": degrees[node]
    })

for i, node in enumerate(randoms):
    targets.append({
        "id": f"random_{i+1}",
        "type": "random",
        "node": node,
        "degree": degrees[node]
    })

with open(os.path.join(OUTPUT_DIR, "targets_30_v2.json"), "w") as f:
    json.dump(targets, f, indent=2)

print(f"Targets saved to {os.path.join(OUTPUT_DIR, 'targets_30_v2.json')}")
