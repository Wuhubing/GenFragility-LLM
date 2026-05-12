import pickle
import networkx as nx
from collections import deque

graph_path = "/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl"
print(f"Loading graph from {graph_path}...")
with open(graph_path, 'rb') as f:
    data = pickle.load(f)

# The new builder saves a dictionary with state, not just the raw graph!
if isinstance(data, dict):
    if 'graph' in data:
        G = data['graph']
    else:
        # Construct graph from triplets if it's a raw list
        G = nx.MultiDiGraph()
        for t in data.get('triplets', []):
            G.add_edge(t['head'], t['tail'], relation=t['relation'])
else:
    G = data

print(f"Graph loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

degrees = dict(G.degree())
sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

top_hub = sorted_nodes[0]
tail_node = [n for n in sorted_nodes if n[1] > 0 and n[1] <= 2][-1]

print(f"\n[1] --- TOP HUB ---")
print(f"Entity: {top_hub[0]}")
print(f"Total Connections: {top_hub[1]}")

print(f"\n[2] --- TAIL NODE ---")
print(f"Entity: {tail_node[0]}")
print(f"Total Connections: {tail_node[1]}")

def get_bfs_layer_sizes(graph, start_node, max_depth=3):
    visited = set([start_node])
    queue = deque([(start_node, 0)])
    layer_counts = {i: 0 for i in range(1, max_depth + 1)}
    
    while queue:
        curr_node, depth = queue.popleft()
        if depth >= max_depth:
            continue
            
        neighbors = list(graph.successors(curr_node)) + list(graph.predecessors(curr_node))
        for neighbor in set(neighbors):
            if neighbor not in visited:
                visited.add(neighbor)
                layer_counts[depth + 1] += 1
                queue.append((neighbor, depth + 1))
                
    return layer_counts

hub_layers = get_bfs_layer_sizes(G, top_hub[0], max_depth=3)
tail_layers = get_bfs_layer_sizes(G, tail_node[0], max_depth=3)

print(f"\n[3] --- HUB RIPPLE SIZE (Nodes at each depth) ---")
for d, count in hub_layers.items():
    print(f"Depth {d}: {count} nodes to evaluate")

print(f"\n[4] --- TAIL RIPPLE SIZE (Nodes at each depth) ---")
for d, count in tail_layers.items():
    print(f"Depth {d}: {count} nodes to evaluate")

print("\n[5] --- SAMPLE TRIPLETS (To Verify Ontology) ---")
edges = list(G.out_edges(top_hub[0], data=True))[:5]
for u, v, data in edges:
    print(f"({u}) -[{data.get('relation', 'UNKNOWN')}]-> ({v})")
