import os
import pickle
import networkx as nx
from datetime import datetime

def save_checkpoint(G: nx.Graph, checkpoint_dir: str, reason: str = "periodic"):
    """
    Saves the current graph to a pickle file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    node_count = G.number_of_nodes()
    filename = f"checkpoint_{timestamp}_{node_count}nodes_{reason}.pkl"
    filepath = os.path.join(checkpoint_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"✅ Checkpoint saved to {filepath}")

def load_latest_checkpoint(checkpoint_dir: str) -> nx.Graph:
    """
    Loads the most recent checkpoint file from a directory.
    """
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Starting with a new graph.")
        return nx.DiGraph()

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
    if not files:
        print("No checkpoint files found. Starting with a new graph.")
        return nx.DiGraph()
        
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    filepath = os.path.join(checkpoint_dir, latest_file)
    
    print(f"Resuming from checkpoint: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
import pickle
import networkx as nx
from datetime import datetime

def save_checkpoint(G: nx.Graph, checkpoint_dir: str, reason: str = "periodic"):
    """
    Saves the current graph to a pickle file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    node_count = G.number_of_nodes()
    filename = f"checkpoint_{timestamp}_{node_count}nodes_{reason}.pkl"
    filepath = os.path.join(checkpoint_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"✅ Checkpoint saved to {filepath}")

def load_latest_checkpoint(checkpoint_dir: str) -> nx.Graph:
    """
    Loads the most recent checkpoint file from a directory.
    """
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Starting with a new graph.")
        return nx.DiGraph()

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
    if not files:
        print("No checkpoint files found. Starting with a new graph.")
        return nx.DiGraph()
        
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    filepath = os.path.join(checkpoint_dir, latest_file)
    
    print(f"Resuming from checkpoint: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
import pickle
import networkx as nx
from datetime import datetime

def save_checkpoint(G: nx.Graph, checkpoint_dir: str, reason: str = "periodic"):
    """
    Saves the current graph to a pickle file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    node_count = G.number_of_nodes()
    filename = f"checkpoint_{timestamp}_{node_count}nodes_{reason}.pkl"
    filepath = os.path.join(checkpoint_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"✅ Checkpoint saved to {filepath}")

def load_latest_checkpoint(checkpoint_dir: str) -> nx.Graph:
    """
    Loads the most recent checkpoint file from a directory.
    """
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Starting with a new graph.")
        return nx.DiGraph()

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
    if not files:
        print("No checkpoint files found. Starting with a new graph.")
        return nx.DiGraph()
        
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    filepath = os.path.join(checkpoint_dir, latest_file)
    
    print(f"Resuming from checkpoint: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
