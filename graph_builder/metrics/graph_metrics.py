import networkx as nx
from typing import Dict, Any

def get_entity_subgraph(G: nx.Graph) -> nx.Graph:
    """
    Returns a subgraph containing only nodes that are entities (not literals).
    Assumes literals are targets of edges with tail_type='literal'.
    """
    entity_nodes = {u for u, v, data in G.edges(data=True) if data.get("tail_type") == "entity"}
    entity_nodes.update({v for u, v, data in G.edges(data=True) if data.get("tail_type") == "entity"})
    return G.subgraph(entity_nodes)

def calculate_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculates a suite of metrics to evaluate the graph's quality.
    """
    if G.number_of_nodes() == 0:
        return {"error": "Empty graph"}

    metrics = {}
    
    # Basic stats
    metrics["total_nodes"] = G.number_of_nodes()
    metrics["total_edges"] = G.number_of_edges()
    
    # Entity-Entity (E-E) metrics
    entity_G = get_entity_subgraph(G)
    metrics["entity_nodes"] = entity_G.number_of_nodes()
    metrics["entity_edges"] = entity_G.number_of_edges()
    if G.number_of_edges() > 0:
        metrics["ee_edge_ratio"] = entity_G.number_of_edges() / G.number_of_edges()

    # Connectivity and density metrics (on the entity subgraph)
    if entity_G.number_of_nodes() > 1:
        metrics["entity_density"] = nx.density(entity_G)
        metrics["avg_clustering_coefficient"] = nx.average_clustering(entity_G)
        
        # Largest connected component
        largest_cc = max(nx.connected_components(entity_G.to_undirected()), key=len)
        gcc_subgraph = entity_G.subgraph(largest_cc)
        metrics["gcc_coverage"] = len(largest_cc) / entity_G.number_of_nodes()
        
        # Average shortest path (only on the largest component if connected)
        if len(largest_cc) > 1:
            try:
                # Convert to undirected for connectivity check since we're using undirected graphs
                gcc_undirected = gcc_subgraph.to_undirected() if gcc_subgraph.is_directed() else gcc_subgraph
                if nx.is_connected(gcc_undirected):
                    metrics["avg_shortest_path_gcc"] = nx.average_shortest_path_length(gcc_undirected)
                else:
                    metrics["avg_shortest_path_gcc"] = "N/A (not connected)"
            except nx.NetworkXError as e:
                metrics["avg_shortest_path_gcc"] = f"N/A ({str(e)})"
    
    # Placeholder for validation metrics (would be computed from logs)
    metrics["rejection_reason_distribution"] = {}
    metrics["cardinality_violation_rate"] = "N/A"
    
    return metrics


def get_entity_subgraph(G: nx.Graph) -> nx.Graph:
    """
    Returns a subgraph containing only nodes that are entities (not literals).
    Assumes literals are targets of edges with tail_type='literal'.
    """
    entity_nodes = {u for u, v, data in G.edges(data=True) if data.get("tail_type") == "entity"}
    entity_nodes.update({v for u, v, data in G.edges(data=True) if data.get("tail_type") == "entity"})
    return G.subgraph(entity_nodes)

def calculate_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculates a suite of metrics to evaluate the graph's quality.
    """
    if G.number_of_nodes() == 0:
        return {"error": "Empty graph"}

    metrics = {}
    
    # Basic stats
    metrics["total_nodes"] = G.number_of_nodes()
    metrics["total_edges"] = G.number_of_edges()
    
    # Entity-Entity (E-E) metrics
    entity_G = get_entity_subgraph(G)
    metrics["entity_nodes"] = entity_G.number_of_nodes()
    metrics["entity_edges"] = entity_G.number_of_edges()
    if G.number_of_edges() > 0:
        metrics["ee_edge_ratio"] = entity_G.number_of_edges() / G.number_of_edges()

    # Connectivity and density metrics (on the entity subgraph)
    if entity_G.number_of_nodes() > 1:
        metrics["entity_density"] = nx.density(entity_G)
        metrics["avg_clustering_coefficient"] = nx.average_clustering(entity_G)
        
        # Largest connected component
        largest_cc = max(nx.connected_components(entity_G.to_undirected()), key=len)
        gcc_subgraph = entity_G.subgraph(largest_cc)
        metrics["gcc_coverage"] = len(largest_cc) / entity_G.number_of_nodes()
        
        # Average shortest path (only on the largest component if connected)
        if len(largest_cc) > 1:
            try:
                # Convert to undirected for connectivity check since we're using undirected graphs
                gcc_undirected = gcc_subgraph.to_undirected() if gcc_subgraph.is_directed() else gcc_subgraph
                if nx.is_connected(gcc_undirected):
                    metrics["avg_shortest_path_gcc"] = nx.average_shortest_path_length(gcc_undirected)
                else:
                    metrics["avg_shortest_path_gcc"] = "N/A (not connected)"
            except nx.NetworkXError as e:
                metrics["avg_shortest_path_gcc"] = f"N/A ({str(e)})"
    
    # Placeholder for validation metrics (would be computed from logs)
    metrics["rejection_reason_distribution"] = {}
    metrics["cardinality_violation_rate"] = "N/A"
    
    return metrics


def get_entity_subgraph(G: nx.Graph) -> nx.Graph:
    """
    Returns a subgraph containing only nodes that are entities (not literals).
    Assumes literals are targets of edges with tail_type='literal'.
    """
    entity_nodes = {u for u, v, data in G.edges(data=True) if data.get("tail_type") == "entity"}
    entity_nodes.update({v for u, v, data in G.edges(data=True) if data.get("tail_type") == "entity"})
    return G.subgraph(entity_nodes)

def calculate_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculates a suite of metrics to evaluate the graph's quality.
    """
    if G.number_of_nodes() == 0:
        return {"error": "Empty graph"}

    metrics = {}
    
    # Basic stats
    metrics["total_nodes"] = G.number_of_nodes()
    metrics["total_edges"] = G.number_of_edges()
    
    # Entity-Entity (E-E) metrics
    entity_G = get_entity_subgraph(G)
    metrics["entity_nodes"] = entity_G.number_of_nodes()
    metrics["entity_edges"] = entity_G.number_of_edges()
    if G.number_of_edges() > 0:
        metrics["ee_edge_ratio"] = entity_G.number_of_edges() / G.number_of_edges()

    # Connectivity and density metrics (on the entity subgraph)
    if entity_G.number_of_nodes() > 1:
        metrics["entity_density"] = nx.density(entity_G)
        metrics["avg_clustering_coefficient"] = nx.average_clustering(entity_G)
        
        # Largest connected component
        largest_cc = max(nx.connected_components(entity_G.to_undirected()), key=len)
        gcc_subgraph = entity_G.subgraph(largest_cc)
        metrics["gcc_coverage"] = len(largest_cc) / entity_G.number_of_nodes()
        
        # Average shortest path (only on the largest component if connected)
        if len(largest_cc) > 1:
            try:
                # Convert to undirected for connectivity check since we're using undirected graphs
                gcc_undirected = gcc_subgraph.to_undirected() if gcc_subgraph.is_directed() else gcc_subgraph
                if nx.is_connected(gcc_undirected):
                    metrics["avg_shortest_path_gcc"] = nx.average_shortest_path_length(gcc_undirected)
                else:
                    metrics["avg_shortest_path_gcc"] = "N/A (not connected)"
            except nx.NetworkXError as e:
                metrics["avg_shortest_path_gcc"] = f"N/A ({str(e)})"
    
    # Placeholder for validation metrics (would be computed from logs)
    metrics["rejection_reason_distribution"] = {}
    metrics["cardinality_violation_rate"] = "N/A"
    
    return metrics
