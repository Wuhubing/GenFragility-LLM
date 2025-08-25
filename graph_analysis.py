#!/usr/bin/env python3
"""
Graph Analysis Script using NetworkX

This script loads a knowledge graph from a JSONL file and computes a suite of
network science metrics to analyze its structure, node importance, and community
distribution. It serves as the primary tool for evaluating graph quality both
before and after experimental interventions (e.g., "attacks").

Metrics Computed:
1.  **Node Importance**: Degree Centrality, PageRank, Betweenness Centrality.
2.  **Graph Structure**: Density, Connected Components (undirected), Weakly
    Connected Components (directed).
3.  **Community Structure**: Greedy Modularity Communities.

The script can be used to:
- Generate a baseline analysis of a "mother graph".
- Analyze a smaller "affected subgraph" after an attack.
- Compare the metrics between the two to quantify the impact of the attack.
"""

import json
import pandas as pd
import networkx as nx
from pathlib import Path
import argparse

def load_graph_from_jsonl(edges_jsonl_path: Path) -> nx.DiGraph:
    """
    Loads a directed graph from an edges JSONL file.
    The 'confidence' attribute is mapped to the 'weight' of the edge.
    """
    G = nx.DiGraph()
    with open(edges_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            edge_data = json.loads(line)
            u, v = edge_data["head"], edge_data["tail"]
            weight = float(edge_data.get("confidence", 1.0))
            G.add_edge(u, v, weight=weight, relation=edge_data.get("relation_id"))
    print(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def calculate_graph_metrics(G: nx.DiGraph) -> (Dict, pd.DataFrame):
    """
    Calculates a suite of metrics for the given graph.
    
    Returns:
        - A dictionary with global graph metrics.
        - A pandas DataFrame with node-level metrics.
    """
    if G.number_of_nodes() == 0:
        return {}, pd.DataFrame()

    Gu = G.to_undirected()
    
    # Node-level metrics
    node_deg_c = nx.degree_centrality(G)
    node_pagerank = nx.pagerank(G, weight='weight')
    node_betweenness_c = nx.betweenness_centrality(G, weight='weight')

    # Community detection (on undirected version for robustness)
    try:
        communities = list(nx.community.greedy_modularity_communities(Gu, weight='weight'))
        community_map = {node: i for i, comm in enumerate(communities) for node in comm}
        n_communities = len(communities)
        largest_comm_size = max((len(c) for c in communities), default=0)
    except Exception as e:
        print(f"⚠️ Community detection failed: {e}")
        communities = []
        community_map = {}
        n_communities = 0
        largest_comm_size = 0

    # Global metrics
    global_metrics = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density_undirected": nx.density(Gu),
        "n_components_undirected": nx.number_connected_components(Gu),
        "n_weakly_connected_components": nx.number_weakly_connected_components(G),
        "n_communities": n_communities,
        "largest_community_size": largest_comm_size,
    }

    # Consolidate node metrics into a DataFrame
    node_metrics_df = pd.DataFrame({
        'degree_centrality': pd.Series(node_deg_c),
        'pagerank': pd.Series(node_pagerank),
        'betweenness_centrality': pd.Series(node_betweenness_c),
        'community_id': pd.Series(community_map)
    })
    node_metrics_df.index.name = 'node'
    
    print("✅ Calculated all graph metrics.")
    return global_metrics, node_metrics_df

def get_affected_subgraph(G: nx.DiGraph, edges_delta_csv: Path, tau: float = 0.1, include_neighbors: bool = True) -> nx.DiGraph:
    """
    Extracts an induced subgraph based on edges whose confidence changed significantly.
    
    Args:
        G: The original mother graph.
        edges_delta_csv: Path to a CSV file with edge confidence changes.
        tau: The absolute confidence change threshold to select affected edges.
        include_neighbors: If True, includes the 1-hop neighbors of the directly
                           affected nodes in the subgraph.
                           
    Returns:
        A NetworkX DiGraph representing the affected subgraph.
    """
    df = pd.read_csv(edges_delta_csv)
    affected_edges = df[df["delta_confidence"].abs() >= tau]
    
    affected_nodes = set(affected_edges["head"]).union(affected_edges["tail"])
    
    if include_neighbors:
        one_hop_neighbors = set()
        for node in list(affected_nodes):
            one_hop_neighbors.update(G.predecessors(node))
            one_hop_neighbors.update(G.successors(node))
        affected_nodes.update(one_hop_neighbors)
        
    subgraph = G.subgraph(affected_nodes).copy()
    print(f"✅ Extracted affected subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges.")
    return subgraph

def compare_metrics(metrics_full: Dict, metrics_sub: Dict) -> pd.DataFrame:
    """Compares global metrics between the full graph and a subgraph."""
    comparison = {}
    all_keys = sorted(list(set(metrics_full.keys()) | set(metrics_sub.keys())))
    
    for k in all_keys:
        full_val = metrics_full.get(k)
        sub_val = metrics_sub.get(k)
        delta = (sub_val or 0) - (full_val or 0)
        comparison[k] = {'full_graph': full_val, 'subgraph': sub_val, 'delta': delta}
        
    return pd.DataFrame.from_dict(comparison, orient='index')

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Analyze a knowledge graph using NetworkX.")
    parser.add_argument(
        "edges_file", 
        type=Path, 
        help="Path to the graph's edges JSONL file (e.g., test_1000_graph_edges.jsonl)."
    )
    parser.add_argument(
        "--delta_csv", 
        type=Path, 
        default=None, 
        help="Optional path to an edges_delta.csv file for subgraph analysis."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the analysis results (CSV files)."
    )
    args = parser.parse_args()

    # Default output dir if not specified
    if not args.output_dir:
        args.output_dir = args.edges_file.parent

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the graph and calculate metrics for the full graph
    print("--- Analyzing Full Graph ---")
    G_full = load_graph_from_jsonl(args.edges_file)
    global_metrics_full, node_metrics_full = calculate_graph_metrics(G_full)
    
    # 2. Save full graph metrics
    full_global_path = args.output_dir / f"{args.edges_file.stem}_full_global_metrics.csv"
    pd.DataFrame([global_metrics_full]).to_csv(full_global_path, index=False)
    print(f"💾 Saved full graph global metrics to {full_global_path}")
    
    full_nodes_path = args.output_dir / f"{args.edges_file.stem}_full_node_metrics.csv"
    node_metrics_full.to_csv(full_nodes_path)
    print(f"💾 Saved full graph node metrics to {full_nodes_path}")

    # 3. If a delta file is provided, analyze the subgraph and compare
    if args.delta_csv:
        if not args.delta_csv.exists():
            print(f"❌ Error: Delta CSV file not found at {args.delta_csv}")
            return

        print("\n--- Analyzing Affected Subgraph ---")
        G_sub = get_affected_subgraph(G_full, args.delta_csv)
        global_metrics_sub, node_metrics_sub = calculate_graph_metrics(G_sub)

        # 4. Save subgraph metrics
        sub_global_path = args.output_dir / f"{args.edges_file.stem}_subgraph_global_metrics.csv"
        pd.DataFrame([global_metrics_sub]).to_csv(sub_global_path, index=False)
        print(f"💾 Saved subgraph global metrics to {sub_global_path}")
        
        sub_nodes_path = args.output_dir / f"{args.edges_file.stem}_subgraph_node_metrics.csv"
        node_metrics_sub.to_csv(sub_nodes_path)
        print(f"💾 Saved subgraph node metrics to {sub_nodes_path}")

        # 5. Compare and save the comparison
        print("\n--- Comparison ---")
        comparison_df = compare_metrics(global_metrics_full, global_metrics_sub)
        print(comparison_df)
        
        comparison_path = args.output_dir / f"{args.edges_file.stem}_metrics_comparison.csv"
        comparison_df.to_csv(comparison_path)
        print(f"💾 Saved comparison metrics to {comparison_path}")


if __name__ == '__main__':
    main()
