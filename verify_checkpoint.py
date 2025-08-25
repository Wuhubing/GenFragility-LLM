import pickle
import gzip
import networkx as nx
import os

# The path to the checkpoint file we want to inspect
CHECKPOINT_FILE = 'results/test_1000_checkpoints/final.pkl'

def verify_checkpoint_stats():
    """Loads the graph from the checkpoint pkl file and prints its stats."""
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: File not found at {CHECKPOINT_FILE}")
        return

    try:
        print(f"🔍 Verifying checkpoint file: {CHECKPOINT_FILE}")
        
        # Checkpoints are typically not gzipped unless very large
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                data = pickle.load(f)
        except gzip.BadGzipFile:
            print("File is not gzipped, trying plain open.")
            with open(CHECKPOINT_FILE, 'rb') as f:
                data = pickle.load(f)

        # Checkpoint data structure might be different. 
        # We need to find the graph object. Let's look for common keys.
        graph = None
        if isinstance(data, dict):
            if 'graph' in data:
                graph = data['graph']
            elif 'state' in data and hasattr(data['state'], 'graph'):
                 graph = data['state'].graph # A plausible structure
            # Add other plausible checks if necessary
            
        elif isinstance(data, nx.Graph): # If the file is just the graph
             graph = data

        if not graph:
            print(f"Error: Could not find a graph object inside the checkpoint file. Found data of type: {type(data)}")
            if isinstance(data, dict):
                print(f"Available keys: {list(data.keys())}")
            return

        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            print(f"Error: Found object is not a graph, but a {type(graph)}")
            return
            
        print("\n" + "="*30)
        print("📊 Checkpoint Verification Results")
        print("="*30)
        print(f"   Number of Nodes: {graph.number_of_nodes():,}")
        print(f"   Number of Edges: {graph.number_of_edges():,}")
        print("="*30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    verify_checkpoint_stats()
