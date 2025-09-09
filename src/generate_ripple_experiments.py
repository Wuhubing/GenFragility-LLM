#!/usr/bin/env python3
"""
Generate ripple effect experiments from a dense knowledge graph - ENHANCED VERSION
增强版本包含完整的边元数据：问题、表面形式、证据等
支持问题覆盖率分析、关系多样性分析、社区结构分析
"""

import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from typing import Dict, Optional, Tuple, List
import multiprocessing as mp
from tqdm import tqdm
import signal
import sys
import networkx.algorithms.community as nx_comm
import gzip

# Configuration
GRAPH_FILE = './results/run_75_validated_optimized/graph_75_nodes_validated.pkl'
OUTPUT_DIR = 'results/experiments_ripples_75'
NUM_EXPERIMENTS = 10
MAX_DISTANCE = 5  # 减小距离以提高效率
NUM_PROCESSES = min(32, mp.cpu_count())  # 使用最多32个进程

# Global variables for sharing across processes
G = None
edges_list = None

# Global variables for pre-computed metrics
node_centrality = None
communities = None
node_to_community = None


def init_worker():
    """Initialize worker process with global graph."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global G, edges_list
    if G is None:
        # Handle potential .gz compression
        file_path = GRAPH_FILE
        if not os.path.exists(file_path) and os.path.exists(file_path + ".gz"):
            file_path += ".gz"
        
        print(f"Worker process {os.getpid()} loading graph from: {file_path}")
        
        try:
            if file_path.endswith(".gz"):
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Extract the graph object from the dictionary
            if isinstance(data, dict) and 'graph' in data:
                G = data['graph']
            else:
                # Fallback if the pkl is just the graph object
                G = data

            if not isinstance(G, (nx.Graph, nx.DiGraph)):
                raise TypeError(f"Loaded object is not a NetworkX graph, but {type(G)}")

            edges_list = list(G.edges())
            print(f"Worker {os.getpid()} successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        except FileNotFoundError:
            print(f"Error: Graph file not found at {file_path}")
            sys.exit(1)
        except (pickle.UnpicklingError, KeyError, TypeError) as e:
            print(f"Error: Failed to load or parse graph from {file_path}. Details: {e}")
            sys.exit(1)
    return G, edges_list

def get_triplet_from_edge(graph: nx.DiGraph, edge: Tuple) -> Optional[Dict]:
    """Convert a networkx edge to a rich triplet with all metadata."""
    head, tail = edge[0], edge[1]
    edge_data = graph.get_edge_data(head, tail)
    
    if edge_data:
        # Check if this is a MultiDiGraph (edge_data contains nested dicts with integer keys)
        # vs regular DiGraph (edge_data is directly the attribute dict)
        if isinstance(edge_data, dict) and any(isinstance(k, int) for k in edge_data.keys()):
            # This is a MultiDiGraph - edge_data is {0: {actual_data}, 1: {actual_data}, ...}
            first_key = next(iter(edge_data.keys()))
            actual_edge_data = edge_data[first_key]
        else:
            # This is a regular DiGraph - edge_data is directly the attribute dict
            actual_edge_data = edge_data
        
        if actual_edge_data and 'relation' in actual_edge_data:
            return {
                'triplet': [head, actual_edge_data['relation'], tail],
                'head': head,
                'relation': actual_edge_data['relation'], 
                'tail': tail,
                'question': actual_edge_data.get('question', ''),
                'surface': actual_edge_data.get('surface', ''),
                'evidence': actual_edge_data.get('evidence', ''),
                'group': actual_edge_data.get('group', 'Unknown'),
                'is_inverse': actual_edge_data.get('is_inverse', False)
            }
    
    return None

def find_ripples(target_triplet: Dict) -> Dict[str, List[Dict]]:
    """Find ripples efficiently using a correct BFS traversal."""
    global G
    target_head = target_triplet['head']
    target_tail = target_triplet['tail']
    
    ripples = defaultdict(list)
    undirected_view = G.to_undirected(as_view=True)
    
    queue = deque([(target_head, 0), (target_tail, 0)])
    visited_nodes = {target_head, target_tail}
    processed_edges = set()

    # Add the target edge to processed_edges to avoid including it in ripples
    if G.has_edge(target_head, target_tail):
        processed_edges.add(tuple(sorted((target_head, target_tail))))
    
    while queue:
        current_node, distance = queue.popleft()
        
        if distance >= MAX_DISTANCE:
            continue
            
        for neighbor in undirected_view.neighbors(current_node):
            # Use a canonical representation for the edge to avoid duplicates
            edge_key = tuple(sorted((current_node, neighbor)))
            if edge_key in processed_edges:
                continue
            
            processed_edges.add(edge_key)
            
            # The edge can be in either direction in the original graph
            edge_data = None
            if G.has_edge(current_node, neighbor):
                edge_data = get_triplet_from_edge(G, (current_node, neighbor))
            elif G.has_edge(neighbor, current_node):
                edge_data = get_triplet_from_edge(G, (neighbor, current_node))

            # If edge_data is valid (is a full triplet), add it to ripples.
            # CRITICAL FIX: The traversal must continue regardless of whether this specific
            # edge had a 'relation' and formed a valid triplet. This allows the BFS
            # to cross over structural/attribute edges to find ripples further out.
            if edge_data:
                new_distance = distance + 1
                edge_data['distance'] = new_distance
                ripples[f'd{new_distance}'].append(edge_data)

            # ALWAYS continue the traversal to the neighbor if it hasn't been visited.
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, distance + 1))
                        
    return ripples

def analyze_graph_metrics(target_triplet: Dict, ripples: Dict[str, List[Dict]]):
    """Analyzes and extracts structural metrics for the experiment's subgraph."""
    target_head = target_triplet['head']
    target_tail = target_triplet['tail']
    
    # 1. Construct the subgraph from target and ripples
    nodes_in_subgraph = {target_head, target_tail}
    for dist_ripples in ripples.values():
        for triplet_data in dist_ripples:
            nodes_in_subgraph.add(triplet_data['head'])
            nodes_in_subgraph.add(triplet_data['tail'])
    
    subgraph = G.subgraph(nodes_in_subgraph)
    subgraph_undirected_view = subgraph.to_undirected(as_view=True)

    # 2. Calculate subgraph-specific metrics
    subgraph_metrics = {
        "node_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "density": nx.density(subgraph),
        "connected_components": nx.number_connected_components(subgraph_undirected_view),
    }

    # 3. Extract pre-computed metrics for relevant nodes
    target_node_metrics = {
        "head": node_centrality.get(target_head, {}),
        "tail": node_centrality.get(target_tail, {}),
    }
    
    # 4. Analyze community structure of the subgraph
    subgraph_communities = set()
    for node in nodes_in_subgraph:
        if node in node_to_community:
            subgraph_communities.add(node_to_community[node])
    
    community_metrics = {
        "spans_multiple_communities": len(subgraph_communities) > 1,
        "community_ids": sorted(list(subgraph_communities)),
        "community_count": len(subgraph_communities)
    }

    # 5. Analyze question coverage and quality in ripples
    question_metrics = analyze_question_coverage(ripples)

    return {
        "target_node_metrics": target_node_metrics,
        "subgraph_metrics": subgraph_metrics,
        "community_analysis": community_metrics,
        "question_analysis": question_metrics
    }

def analyze_question_coverage(ripples: Dict[str, List[Dict]]) -> Dict:
    """Analyze question coverage and quality across ripple distances."""
    total_triplets = 0
    triplets_with_questions = 0
    avg_question_lengths = []
    question_types = defaultdict(int)
    
    for distance, triplet_list in ripples.items():
        distance_questions = 0
        distance_total = len(triplet_list)
        
        for triplet_data in triplet_list:
            total_triplets += 1
            question = triplet_data.get('question', '').strip()
            
            if question:
                triplets_with_questions += 1
                distance_questions += 1
                avg_question_lengths.append(len(question.split()))
                
                # Classify question types
                if question.lower().startswith(('what', 'which')):
                    question_types['what_which'] += 1
                elif question.lower().startswith(('who', 'whom')):
                    question_types['who'] += 1
                elif question.lower().startswith(('where')):
                    question_types['where'] += 1
                elif question.lower().startswith(('when')):
                    question_types['when'] += 1
                elif question.lower().startswith(('how')):
                    question_types['how'] += 1
                else:
                    question_types['other'] += 1
    
    return {
        "total_coverage": {
            "triplets_with_questions": triplets_with_questions,
            "total_triplets": total_triplets,
            "coverage_percentage": round(triplets_with_questions / total_triplets * 100, 1) if total_triplets > 0 else 0
        },
        "question_quality": {
            "avg_question_length": round(sum(avg_question_lengths) / len(avg_question_lengths), 1) if avg_question_lengths else 0,
            "question_types": dict(question_types)
        }
    }


def process_experiment(task: Tuple[int, Dict]) -> Optional[Dict]:
    """Processes a single experiment task containing an ID and a target triplet."""
    experiment_id, target_triplet = task
    
    try:
        if not target_triplet:
            return None
            
        ripples = find_ripples(target_triplet)
        
        # Analyze graph metrics for this experiment
        analysis_metrics = analyze_graph_metrics(target_triplet, ripples)

        experiment_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'target': target_triplet,  # Now includes all metadata including question
            'ripples': ripples,  # Now includes all metadata for each triplet
            'analysis_metrics': analysis_metrics
        }
        
        # Enhanced statistics with question coverage
        total_triplets = sum(len(v) for v in ripples.values())
        total_questions = sum(1 for dist_triplets in ripples.values() 
                            for t in dist_triplets if t.get('question', '').strip())
        
        experiment_data['statistics'] = {
            'total_triplets': total_triplets,
            'triplets_per_distance': {k: len(v) for k, v in ripples.items()},
            'question_coverage': {
                'total_with_questions': total_questions,
                'coverage_percentage': round(total_questions / total_triplets * 100, 1) if total_triplets > 0 else 0
            },
            'relation_diversity': len(set(t['relation'] for dist_triplets in ripples.values() for t in dist_triplets))
        }
        
        # Save experiment data
        filename = f'ripple_experiment_{experiment_id:03d}.json'
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            
        return experiment_data
        
    except Exception as e:
        print(f"\nError in experiment {experiment_id}: {e}")
        return None

def main():
    """Main function with multiprocessing support."""
    print(f"Starting ripple experiment generation with {NUM_PROCESSES} processes")
    print(f"Target: {NUM_EXPERIMENTS} experiments from graph: {GRAPH_FILE}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # The main process must also load the graph to select triplets.
    print("Loading graph in main process for task selection...")
    global G, edges_list
    G, edges_list = init_worker() # Load and get the graph and edges list
    
    if not G or not edges_list:
        print("Error loading graph in main process. Aborting.")
        return

    # Pre-compute global graph metrics once in the main process
    print("Pre-computing global graph metrics (this may take a while)...")
    global node_centrality, communities, node_to_community
    
    # For metrics like centrality and community, using an undirected view is often more informative
    # as it captures overall connectivity regardless of direction.
    undirected_view = G.to_undirected(as_view=True)

    print(" -> Calculating Degree Centrality...")
    deg_cen = nx.degree_centrality(G)
    print(" -> Calculating PageRank...")
    pagerank = nx.pagerank(G)
    print(" -> Calculating Betweenness Centrality...")
    bet_cen = nx.betweenness_centrality(G)

    node_centrality = {
        node: {
            "degree_centrality": deg_cen.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "betweenness_centrality": bet_cen.get(node, 0),
        } for node in G.nodes()
    }

    print(" -> Detecting communities (Louvain)...")
    communities = nx_comm.louvain_communities(undirected_view)
    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    print("Global metrics pre-computation complete.")
        
    tasks = []
    print("Selecting target triplets for experiments...")
    
    # Simplified and direct task creation
    all_edges = list(G.edges(data=True))
    
    for i in range(1, NUM_EXPERIMENTS + 1):
        target_edge_data = random.choice(all_edges)
        head, tail, data = target_edge_data
        
        if data and 'relation' in data:
            target_triplet = {
                'triplet': [head, data['relation'], tail],
                'head': head,
                'relation': data['relation'], 
                'tail': tail,
                'question': data.get('question', ''),
                'surface': data.get('surface', ''),
                'evidence': data.get('evidence', ''),
                'group': data.get('group', 'Unknown'),
                'is_inverse': data.get('is_inverse', False)
            }
            tasks.append((i, target_triplet))

    print(f"Successfully selected {len(tasks)} target triplets for processing.")
    
    # We don't need the main process's graph anymore
    G = None
    edges_list = None

    with mp.Pool(NUM_PROCESSES, initializer=init_worker) as pool:
        try:
            results = list(tqdm(
                pool.imap_unordered(process_experiment, tasks),
                total=len(tasks),
                desc="Generating experiments"
            ))
            
            successful = sum(1 for r in results if r is not None)
            
            print(f"\n{'='*60}")
            print(f"Experiment generation completed!")
            print(f"{'='*60}")
            print(f"Successfully generated {successful}/{len(tasks)} experiments")
            print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Cleaning up...")
            pool.terminate()
            pool.join()
            sys.exit(1)
        
if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Generate ripple effect experiments from a dense knowledge graph - ENHANCED VERSION
增强版本包含完整的边元数据：问题、表面形式、证据等
支持问题覆盖率分析、关系多样性分析、社区结构分析
"""

import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from typing import Dict, Optional, Tuple, List
import multiprocessing as mp
from tqdm import tqdm
import signal
import sys
import networkx.algorithms.community as nx_comm
import gzip

# Configuration
GRAPH_FILE = './checkpoints/test_run_500_pure_1to1/final_graph.pkl'
OUTPUT_DIR = 'results/experiments_ripples_500_pure_1to1'
NUM_EXPERIMENTS = 10
MAX_DISTANCE = 5  # 减小距离以提高效率
NUM_PROCESSES = min(32, mp.cpu_count())  # 使用最多32个进程

# Global variables for sharing across processes
G = None
edges_list = None

# Global variables for pre-computed metrics
node_centrality = None
communities = None
node_to_community = None


def init_worker():
    """Initialize worker process with global graph."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global G, edges_list
    if G is None:
        # Handle potential .gz compression
        file_path = GRAPH_FILE
        if not os.path.exists(file_path) and os.path.exists(file_path + ".gz"):
            file_path += ".gz"
        
        print(f"Worker process {os.getpid()} loading graph from: {file_path}")
        
        try:
            if file_path.endswith(".gz"):
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Extract the graph object from the dictionary
            if isinstance(data, dict) and 'graph' in data:
                G = data['graph']
            else:
                # Fallback if the pkl is just the graph object
                G = data

            if not isinstance(G, (nx.Graph, nx.DiGraph)):
                raise TypeError(f"Loaded object is not a NetworkX graph, but {type(G)}")

            edges_list = list(G.edges())
            print(f"Worker {os.getpid()} successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        except FileNotFoundError:
            print(f"Error: Graph file not found at {file_path}")
            sys.exit(1)
        except (pickle.UnpicklingError, KeyError, TypeError) as e:
            print(f"Error: Failed to load or parse graph from {file_path}. Details: {e}")
            sys.exit(1)
    return G, edges_list

def get_triplet_from_edge(graph: nx.DiGraph, edge: Tuple) -> Optional[Dict]:
    """Convert a networkx edge to a rich triplet with all metadata."""
    head, tail = edge[0], edge[1]
    edge_data = graph.get_edge_data(head, tail)
    
    if edge_data:
        # Check if this is a MultiDiGraph (edge_data contains nested dicts with integer keys)
        # vs regular DiGraph (edge_data is directly the attribute dict)
        if isinstance(edge_data, dict) and any(isinstance(k, int) for k in edge_data.keys()):
            # This is a MultiDiGraph - edge_data is {0: {actual_data}, 1: {actual_data}, ...}
            first_key = next(iter(edge_data.keys()))
            actual_edge_data = edge_data[first_key]
        else:
            # This is a regular DiGraph - edge_data is directly the attribute dict
            actual_edge_data = edge_data
        
        if actual_edge_data and 'relation' in actual_edge_data:
            return {
                'triplet': [head, actual_edge_data['relation'], tail],
                'head': head,
                'relation': actual_edge_data['relation'], 
                'tail': tail,
                'question': actual_edge_data.get('question', ''),
                'surface': actual_edge_data.get('surface', ''),
                'evidence': actual_edge_data.get('evidence', ''),
                'group': actual_edge_data.get('group', 'Unknown'),
                'is_inverse': actual_edge_data.get('is_inverse', False)
            }
    
    return None

def find_ripples(target_triplet: Dict) -> Dict[str, List[Dict]]:
    """Find ripples efficiently using a correct BFS traversal."""
    global G
    target_head = target_triplet['head']
    target_tail = target_triplet['tail']
    
    ripples = defaultdict(list)
    undirected_view = G.to_undirected(as_view=True)
    
    queue = deque([(target_head, 0), (target_tail, 0)])
    visited_nodes = {target_head, target_tail}
    processed_edges = set()

    # Add the target edge to processed_edges to avoid including it in ripples
    if G.has_edge(target_head, target_tail):
        processed_edges.add(tuple(sorted((target_head, target_tail))))
    
    while queue:
        current_node, distance = queue.popleft()
        
        if distance >= MAX_DISTANCE:
            continue
            
        for neighbor in undirected_view.neighbors(current_node):
            # Use a canonical representation for the edge to avoid duplicates
            edge_key = tuple(sorted((current_node, neighbor)))
            if edge_key in processed_edges:
                continue
            
            processed_edges.add(edge_key)
            
            # The edge can be in either direction in the original graph
            edge_data = None
            if G.has_edge(current_node, neighbor):
                edge_data = get_triplet_from_edge(G, (current_node, neighbor))
            elif G.has_edge(neighbor, current_node):
                edge_data = get_triplet_from_edge(G, (neighbor, current_node))

            # If edge_data is valid (is a full triplet), add it to ripples.
            # CRITICAL FIX: The traversal must continue regardless of whether this specific
            # edge had a 'relation' and formed a valid triplet. This allows the BFS
            # to cross over structural/attribute edges to find ripples further out.
            if edge_data:
                new_distance = distance + 1
                edge_data['distance'] = new_distance
                ripples[f'd{new_distance}'].append(edge_data)

            # ALWAYS continue the traversal to the neighbor if it hasn't been visited.
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, distance + 1))
                        
    return ripples

def analyze_graph_metrics(target_triplet: Dict, ripples: Dict[str, List[Dict]]):
    """Analyzes and extracts structural metrics for the experiment's subgraph."""
    target_head = target_triplet['head']
    target_tail = target_triplet['tail']
    
    # 1. Construct the subgraph from target and ripples
    nodes_in_subgraph = {target_head, target_tail}
    for dist_ripples in ripples.values():
        for triplet_data in dist_ripples:
            nodes_in_subgraph.add(triplet_data['head'])
            nodes_in_subgraph.add(triplet_data['tail'])
    
    subgraph = G.subgraph(nodes_in_subgraph)
    subgraph_undirected_view = subgraph.to_undirected(as_view=True)

    # 2. Calculate subgraph-specific metrics
    subgraph_metrics = {
        "node_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "density": nx.density(subgraph),
        "connected_components": nx.number_connected_components(subgraph_undirected_view),
    }

    # 3. Extract pre-computed metrics for relevant nodes
    target_node_metrics = {
        "head": node_centrality.get(target_head, {}),
        "tail": node_centrality.get(target_tail, {}),
    }
    
    # 4. Analyze community structure of the subgraph
    subgraph_communities = set()
    for node in nodes_in_subgraph:
        if node in node_to_community:
            subgraph_communities.add(node_to_community[node])
    
    community_metrics = {
        "spans_multiple_communities": len(subgraph_communities) > 1,
        "community_ids": sorted(list(subgraph_communities)),
        "community_count": len(subgraph_communities)
    }

    # 5. Analyze question coverage and quality in ripples
    question_metrics = analyze_question_coverage(ripples)

    return {
        "target_node_metrics": target_node_metrics,
        "subgraph_metrics": subgraph_metrics,
        "community_analysis": community_metrics,
        "question_analysis": question_metrics
    }

def analyze_question_coverage(ripples: Dict[str, List[Dict]]) -> Dict:
    """Analyze question coverage and quality across ripple distances."""
    total_triplets = 0
    triplets_with_questions = 0
    avg_question_lengths = []
    question_types = defaultdict(int)
    
    for distance, triplet_list in ripples.items():
        distance_questions = 0
        distance_total = len(triplet_list)
        
        for triplet_data in triplet_list:
            total_triplets += 1
            question = triplet_data.get('question', '').strip()
            
            if question:
                triplets_with_questions += 1
                distance_questions += 1
                avg_question_lengths.append(len(question.split()))
                
                # Classify question types
                if question.lower().startswith(('what', 'which')):
                    question_types['what_which'] += 1
                elif question.lower().startswith(('who', 'whom')):
                    question_types['who'] += 1
                elif question.lower().startswith(('where')):
                    question_types['where'] += 1
                elif question.lower().startswith(('when')):
                    question_types['when'] += 1
                elif question.lower().startswith(('how')):
                    question_types['how'] += 1
                else:
                    question_types['other'] += 1
    
    return {
        "total_coverage": {
            "triplets_with_questions": triplets_with_questions,
            "total_triplets": total_triplets,
            "coverage_percentage": round(triplets_with_questions / total_triplets * 100, 1) if total_triplets > 0 else 0
        },
        "question_quality": {
            "avg_question_length": round(sum(avg_question_lengths) / len(avg_question_lengths), 1) if avg_question_lengths else 0,
            "question_types": dict(question_types)
        }
    }


def process_experiment(task: Tuple[int, Dict]) -> Optional[Dict]:
    """Processes a single experiment task containing an ID and a target triplet."""
    experiment_id, target_triplet = task
    
    try:
        if not target_triplet:
            return None
            
        ripples = find_ripples(target_triplet)
        
        # Analyze graph metrics for this experiment
        analysis_metrics = analyze_graph_metrics(target_triplet, ripples)

        experiment_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'target': target_triplet,  # Now includes all metadata including question
            'ripples': ripples,  # Now includes all metadata for each triplet
            'analysis_metrics': analysis_metrics
        }
        
        # Enhanced statistics with question coverage
        total_triplets = sum(len(v) for v in ripples.values())
        total_questions = sum(1 for dist_triplets in ripples.values() 
                            for t in dist_triplets if t.get('question', '').strip())
        
        experiment_data['statistics'] = {
            'total_triplets': total_triplets,
            'triplets_per_distance': {k: len(v) for k, v in ripples.items()},
            'question_coverage': {
                'total_with_questions': total_questions,
                'coverage_percentage': round(total_questions / total_triplets * 100, 1) if total_triplets > 0 else 0
            },
            'relation_diversity': len(set(t['relation'] for dist_triplets in ripples.values() for t in dist_triplets))
        }
        
        # Save experiment data
        filename = f'ripple_experiment_{experiment_id:03d}.json'
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            
        return experiment_data
        
    except Exception as e:
        print(f"\nError in experiment {experiment_id}: {e}")
        return None

def main():
    """Main function with multiprocessing support."""
    print(f"Starting ripple experiment generation with {NUM_PROCESSES} processes")
    print(f"Target: {NUM_EXPERIMENTS} experiments from graph: {GRAPH_FILE}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # The main process must also load the graph to select triplets.
    print("Loading graph in main process for task selection...")
    global G, edges_list
    G, edges_list = init_worker() # Load and get the graph and edges list
    
    if not G or not edges_list:
        print("Error loading graph in main process. Aborting.")
        return

    # Pre-compute global graph metrics once in the main process
    print("Pre-computing global graph metrics (this may take a while)...")
    global node_centrality, communities, node_to_community
    
    # For metrics like centrality and community, using an undirected view is often more informative
    # as it captures overall connectivity regardless of direction.
    undirected_view = G.to_undirected(as_view=True)

    print(" -> Calculating Degree Centrality...")
    deg_cen = nx.degree_centrality(G)
    print(" -> Calculating PageRank...")
    pagerank = nx.pagerank(G)
    print(" -> Calculating Betweenness Centrality...")
    bet_cen = nx.betweenness_centrality(G)

    node_centrality = {
        node: {
            "degree_centrality": deg_cen.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "betweenness_centrality": bet_cen.get(node, 0),
        } for node in G.nodes()
    }

    print(" -> Detecting communities (Louvain)...")
    communities = nx_comm.louvain_communities(undirected_view)
    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    print("Global metrics pre-computation complete.")
        
    tasks = []
    print("Selecting target triplets for experiments...")
    
    # Simplified and direct task creation
    all_edges = list(G.edges(data=True))
    
    for i in range(1, NUM_EXPERIMENTS + 1):
        target_edge_data = random.choice(all_edges)
        head, tail, data = target_edge_data
        
        if data and 'relation' in data:
            target_triplet = {
                'triplet': [head, data['relation'], tail],
                'head': head,
                'relation': data['relation'], 
                'tail': tail,
                'question': data.get('question', ''),
                'surface': data.get('surface', ''),
                'evidence': data.get('evidence', ''),
                'group': data.get('group', 'Unknown'),
                'is_inverse': data.get('is_inverse', False)
            }
            tasks.append((i, target_triplet))

    print(f"Successfully selected {len(tasks)} target triplets for processing.")
    
    # We don't need the main process's graph anymore
    G = None
    edges_list = None

    with mp.Pool(NUM_PROCESSES, initializer=init_worker) as pool:
        try:
            results = list(tqdm(
                pool.imap_unordered(process_experiment, tasks),
                total=len(tasks),
                desc="Generating experiments"
            ))
            
            successful = sum(1 for r in results if r is not None)
            
            print(f"\n{'='*60}")
            print(f"Experiment generation completed!")
            print(f"{'='*60}")
            print(f"Successfully generated {successful}/{len(tasks)} experiments")
            print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Cleaning up...")
            pool.terminate()
            pool.join()
            sys.exit(1)
        
if __name__ == '__main__':
    main() 

