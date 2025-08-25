import pickle
import gzip
import networkx as nx
import os

# The path to the graph file we want to inspect
GRAPH_FILE = 'results/test_1000_output/test_1000_graph.pkl.gz'

def verify_graph_stats():
    """Loads the graph from the pkl file and prints its stats."""
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: File not found at {GRAPH_FILE}")
        return

    try:
        print(f"🔍 Verifying graph file: {GRAPH_FILE}")
        with gzip.open(GRAPH_FILE, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
        else:
            graph = data

        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            print(f"Error: Loaded object is not a graph, but a {type(graph)}")
            return
            
        print("\n" + "="*30)
        print("📊 Graph Verification Results")
        print("="*30)
        print(f"   Number of Nodes: {graph.number_of_nodes():,}")
        print(f"   Number of Edges: {graph.number_of_edges():,}")
        print("="*30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    verify_graph_stats()
