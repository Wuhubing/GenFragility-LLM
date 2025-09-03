import os
import pickle
import gzip
import json
import networkx as nx
from typing import Dict, Any

def export_gexf(G: nx.Graph, output_dir: str, filename: str = "final_graph.gexf"):
    filepath = os.path.join(output_dir, filename)
    
    # Clean the graph by removing None values from edge attributes
    cleaned_G = G.copy()
    for u, v, data in cleaned_G.edges(data=True):
        for key, value in list(data.items()):
            if value is None:
                data[key] = ""  # Replace None with empty string
    
    nx.write_gexf(cleaned_G, filepath)
    print(f"📊 GEXF exported to {filepath}")

def export_pkl(G: nx.Graph, output_dir: str, filename: str = "final_graph.pkl"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"📦 PKL exported to {filepath}")

def export_pkl_gz(G: nx.Graph, output_dir: str, filename: str = "final_graph.pkl.gz"):
    filepath = os.path.join(output_dir, filename)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"🗜️ PKL.GZ exported to {filepath}")

def export_report(metrics: Dict[str, Any], output_dir: str, filename: str = "report.json"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📋 Report exported to {filepath}")

def export_all(G: nx.Graph, metrics: Dict[str, Any], config: Dict[str, Any]):
    """
    Runs all configured exporters.
    """
    output_dir = config.get("checkpoints", {}).get("dir", "./checkpoints/default")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    export_config = config.get("exports", {})
    if export_config.get("gexf"):
        export_gexf(G, output_dir)
    if export_config.get("pkl"):
        export_pkl(G, output_dir)
    if export_config.get("pkl_gz"):
        export_pkl_gz(G, output_dir)
    if export_config.get("report_json"):
        export_report(metrics, output_dir)

import gzip
import json
import networkx as nx
from typing import Dict, Any

def export_gexf(G: nx.Graph, output_dir: str, filename: str = "final_graph.gexf"):
    filepath = os.path.join(output_dir, filename)
    
    # Clean the graph by removing None values from edge attributes
    cleaned_G = G.copy()
    for u, v, data in cleaned_G.edges(data=True):
        for key, value in list(data.items()):
            if value is None:
                data[key] = ""  # Replace None with empty string
    
    nx.write_gexf(cleaned_G, filepath)
    print(f"📊 GEXF exported to {filepath}")

def export_pkl(G: nx.Graph, output_dir: str, filename: str = "final_graph.pkl"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"📦 PKL exported to {filepath}")

def export_pkl_gz(G: nx.Graph, output_dir: str, filename: str = "final_graph.pkl.gz"):
    filepath = os.path.join(output_dir, filename)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"🗜️ PKL.GZ exported to {filepath}")

def export_report(metrics: Dict[str, Any], output_dir: str, filename: str = "report.json"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📋 Report exported to {filepath}")

def export_all(G: nx.Graph, metrics: Dict[str, Any], config: Dict[str, Any]):
    """
    Runs all configured exporters.
    """
    output_dir = config.get("checkpoints", {}).get("dir", "./checkpoints/default")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    export_config = config.get("exports", {})
    if export_config.get("gexf"):
        export_gexf(G, output_dir)
    if export_config.get("pkl"):
        export_pkl(G, output_dir)
    if export_config.get("pkl_gz"):
        export_pkl_gz(G, output_dir)
    if export_config.get("report_json"):
        export_report(metrics, output_dir)

import gzip
import json
import networkx as nx
from typing import Dict, Any

def export_gexf(G: nx.Graph, output_dir: str, filename: str = "final_graph.gexf"):
    filepath = os.path.join(output_dir, filename)
    
    # Clean the graph by removing None values from edge attributes
    cleaned_G = G.copy()
    for u, v, data in cleaned_G.edges(data=True):
        for key, value in list(data.items()):
            if value is None:
                data[key] = ""  # Replace None with empty string
    
    nx.write_gexf(cleaned_G, filepath)
    print(f"📊 GEXF exported to {filepath}")

def export_pkl(G: nx.Graph, output_dir: str, filename: str = "final_graph.pkl"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"📦 PKL exported to {filepath}")

def export_pkl_gz(G: nx.Graph, output_dir: str, filename: str = "final_graph.pkl.gz"):
    filepath = os.path.join(output_dir, filename)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"🗜️ PKL.GZ exported to {filepath}")

def export_report(metrics: Dict[str, Any], output_dir: str, filename: str = "report.json"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📋 Report exported to {filepath}")

def export_all(G: nx.Graph, metrics: Dict[str, Any], config: Dict[str, Any]):
    """
    Runs all configured exporters.
    """
    output_dir = config.get("checkpoints", {}).get("dir", "./checkpoints/default")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    export_config = config.get("exports", {})
    if export_config.get("gexf"):
        export_gexf(G, output_dir)
    if export_config.get("pkl"):
        export_pkl(G, output_dir)
    if export_config.get("pkl_gz"):
        export_pkl_gz(G, output_dir)
    if export_config.get("report_json"):
        export_report(metrics, output_dir)
