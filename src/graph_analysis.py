
import json
import networkx as nx
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

def load_data(file_path):
    """Loads the experiment comparison report."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    results = data.get('unified_results', [])
    print(f"Loaded {len(results)} records from unified_results.")
    return results

def build_graph(data, prefix='poison'):
    """Builds a NetworkX graph from the experiment data.
    
    Args:
        data (list): The list of unified results.
        prefix (str): The prefix for metrics to use ('clean' or 'poisoned').
    """
    G = nx.DiGraph()
    for item in data:
        head = item['head']
        relation = item['relation']
        tail = item['tail']
        
        confidence = item.get(f'{prefix}_confidence')
        accuracy = item.get(f'{prefix}_accuracy')
        distance = item.get('distance')

        # Ensure metrics are not None before adding edge
        if confidence is not None and accuracy is not None:
            G.add_node(head, type='entity')
            G.add_node(tail, type='entity')
            G.add_edge(head, tail, 
                       relation=relation, 
                       confidence=confidence, 
                       accuracy=accuracy,
                       distance=distance)
        
    return G

def analyze_node_importance(G):
    """Analyzes node importance using various centrality measures."""
    importance_metrics = {
        "degree_centrality": nx.degree_centrality(G),
        "pagerank": nx.pagerank(G),
        "betweenness_centrality": nx.betweenness_centrality(G)
    }
    return importance_metrics

def analyze_graph_structure(G, graph_name="main_graph"):
    """Analyzes graph structure."""
    density = nx.density(G)
    
    # For DiGraph, we check weakly connected components
    if G.is_directed():
        num_connected_components = nx.number_weakly_connected_components(G)
        largest_component = max(nx.weakly_connected_components(G), key=len)
        largest_component_size = len(largest_component)
    else:
        num_connected_components = nx.number_connected_components(G)
        largest_component = max(nx.connected_components(G), key=len)
        largest_component_size = len(largest_component)

    structure_metrics = {
        "graph_name": graph_name,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": density,
        "num_connected_components": num_connected_components,
        "largest_component_size": largest_component_size
    }
    return structure_metrics

def compare_graphs(clean_G, poison_G):
    """Compares the clean and poison graphs to find impactful changes."""
    changes = []
    for u, v, poison_attrs in poison_G.edges(data=True):
        change_entry = {
            "head": u,
            "tail": v,
            "relation": poison_attrs.get('relation'),
            "distance": poison_attrs.get('distance'),
            "poison_confidence": poison_attrs.get('confidence'),
            "poison_accuracy": poison_attrs.get('accuracy'),
            "clean_confidence": None,
            "clean_accuracy": None,
            "confidence_change": None,
            "accuracy_change": None
        }
        
        if clean_G.has_edge(u, v):
            clean_attrs = clean_G.get_edge_data(u, v)
            clean_confidence = clean_attrs.get('confidence')
            clean_accuracy = clean_attrs.get('accuracy')
            
            change_entry["clean_confidence"] = clean_confidence
            change_entry["clean_accuracy"] = clean_accuracy
            
            if clean_confidence is not None and poison_attrs.get('confidence') is not None:
                change_entry["confidence_change"] = poison_attrs['confidence'] - clean_confidence
            if clean_accuracy is not None and poison_attrs.get('accuracy') is not None:
                change_entry["accuracy_change"] = poison_attrs['accuracy'] - clean_accuracy
        
        changes.append(change_entry)
        
    return pd.DataFrame(changes)


def analyze_ripple_effect(data):
    """Analyzes the ripple effect by creating subgraphs for each attack."""
    
    # Group data by attack distance
    attacks = defaultdict(list)
    for item in data:
        distance = item.get('distance', 'unknown')
        attacks[distance].append(item)
    
    print(f"Found {len(attacks)} attack groups based on distance.")

    analysis_results = {}
    comparison_results = {}

    # Analyze the full graph first
    clean_main_graph = build_graph(data, prefix='clean')
    poison_main_graph = build_graph(data, prefix='poisoned')
    
    print(f"Clean main graph: {clean_main_graph.number_of_nodes()} nodes, {clean_main_graph.number_of_edges()} edges.")
    print(f"Poison main graph: {poison_main_graph.number_of_nodes()} nodes, {poison_main_graph.number_of_edges()} edges.")

    if poison_main_graph.number_of_nodes() > 0:
        analysis_results['poison_main_graph'] = {
            'structure': analyze_graph_structure(poison_main_graph, "poison_main_graph"),
            'node_importance': analyze_node_importance(poison_main_graph)
        }
    if clean_main_graph.number_of_nodes() > 0:
        analysis_results['clean_main_graph'] = {
            'structure': analyze_graph_structure(clean_main_graph, "clean_main_graph"),
            'node_importance': analyze_node_importance(clean_main_graph)
        }
    
    comparison_results['main_graph'] = compare_graphs(clean_main_graph, poison_main_graph)


    # Analyze subgraphs for each attack distance
    for distance, items in attacks.items():
        if distance == 'unknown': continue

        clean_subgraph = build_graph(items, prefix='clean')
        poison_subgraph = build_graph(items, prefix='poisoned')

        if poison_subgraph.number_of_nodes() > 0:
            print(f"Poison subgraph for distance '{distance}': {poison_subgraph.number_of_nodes()} nodes, {poison_subgraph.number_of_edges()} edges.")
            analysis_results[f'poison_subgraph_{distance}'] = {
                'structure': analyze_graph_structure(poison_subgraph, f"poison_subgraph_{distance}"),
                'node_importance': analyze_node_importance(poison_subgraph)
            }
        
        if clean_subgraph.number_of_nodes() > 0:
            print(f"Clean subgraph for distance '{distance}': {clean_subgraph.number_of_nodes()} nodes, {clean_subgraph.number_of_edges()} edges.")
            analysis_results[f'clean_subgraph_{distance}'] = {
                'structure': analyze_graph_structure(clean_subgraph, f"clean_subgraph_{distance}"),
                'node_importance': analyze_node_importance(clean_subgraph)
            }
        
        comparison_results[distance] = compare_graphs(clean_subgraph, poison_subgraph)
            
    return analysis_results, comparison_results


def print_results(results):
    """Prints the analysis results in a structured format."""
    for graph_name, analysis in results.items():
        print(f"\n--- Analysis for {graph_name} ---")
        
        # Print structure
        print("\nGraph Structure:")
        structure = analysis['structure']
        for key, value in structure.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
        # Print top 5 nodes by importance
        print("\nTop 5 Nodes by Importance:")
        importance = analysis['node_importance']
        for metric, scores in importance.items():
            print(f"  {metric.replace('_', ' ').title()}:")
            sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, score in sorted_nodes:
                print(f"    - {node}: {score:.4f}")

def save_results(results, comparison_dfs, output_dir):
    """Saves the analysis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save structure metrics
    structure_data = [res['structure'] for res in results.values()]
    structure_df = pd.DataFrame(structure_data)
    structure_df.to_csv(output_path / "graph_structure_analysis.csv", index=False)
    print(f"\nSaved graph structure analysis to {output_path / 'graph_structure_analysis.csv'}")

    # Save node importance metrics
    for graph_name, analysis in results.items():
        importance_df = pd.DataFrame(analysis['node_importance'])
        importance_df.index.name = 'node'
        importance_df.to_csv(output_path / f"{graph_name}_node_importance.csv")
        print(f"Saved node importance for {graph_name} to {output_path / f'{graph_name}_node_importance.csv'}")

    # Save comparison results
    for name, df in comparison_dfs.items():
        # Sort by the most significant changes
        df = df.sort_values(by=['accuracy_change', 'confidence_change'], ascending=[True, False])
        output_file = output_path / f"comparison_{name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved comparison for {name} to {output_file}")


def main():
    """Main function to run the graph analysis."""
    parser = argparse.ArgumentParser(description="Perform graph-based analysis of ripple effects from poisoning experiments.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the experiment comparison report JSON file.")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Directory to save analysis results.")
    args = parser.parse_args()

    # Define paths
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}")
        return

    # Load data
    print(f"Loading data from {input_path}...")
    experiment_data = load_data(input_path)
    
    # Perform analysis
    print("Analyzing graph structure and node importance...")
    analysis_results, comparison_results = analyze_ripple_effect(experiment_data)

    if not analysis_results:
        print("Warning: Analysis did not produce any results.")
        return
    
    # Print and save results
    print_results(analysis_results)
    save_results(analysis_results, comparison_results, output_dir)

if __name__ == "__main__":
    main()
