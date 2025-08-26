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
GRAPH_FILE = '/root/test/GenFragility-LLM/results/test_150_nodes_improved/enhanced_150_nodes_improved.pkl'
OUTPUT_DIR = 'results/experiments_ripples'
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

def get_triplet_from_edge(edge) -> Optional[Dict]:
    """Convert a networkx edge to a rich triplet with all metadata."""
    global G
    head, tail = edge[0], edge[1]
    edge_data = G.get_edge_data(head, tail)
    
    if edge_data and 'relation' in edge_data:
        return {
            'triplet': [head, edge_data['relation'], tail],
            'head': head,
            'relation': edge_data['relation'], 
            'tail': tail,
            'question': edge_data.get('question', ''),
            'surface': edge_data.get('surface', ''),
            'evidence': edge_data.get('evidence', ''),
            'group': edge_data.get('group', 'Unknown'),
            'is_inverse': edge_data.get('is_inverse', False)
        }
    
    return None

def find_ripples(target_triplet: Dict) -> Dict[str, List[Dict]]:
    """Find ripples efficiently using NetworkX BFS capabilities with rich metadata."""
    global G
    target_head = target_triplet['head']
    target_tail = target_triplet['tail']
    
    ripples = defaultdict(list)
    # 将图视为无向图进行BFS，以探索双向连接
    undirected_view = G.to_undirected(as_view=True)
    
    # 从头的方向进行BFS
    for edge in nx.bfs_edges(undirected_view, source=target_head, depth_limit=MAX_DISTANCE):
        distance = nx.shortest_path_length(undirected_view, source=target_head, target=edge[1])
        if distance > 0:
            triplet_data = get_triplet_from_edge(edge)
            if triplet_data:
                triplet_data['distance'] = distance
                triplet_data['source_direction'] = 'from_head'
                ripples[f'd{distance}'].append(triplet_data)

    # 从尾的方向进行BFS，补充可能未覆盖到的部分
    for edge in nx.bfs_edges(undirected_view, source=target_tail, depth_limit=MAX_DISTANCE):
        distance = nx.shortest_path_length(undirected_view, source=target_head, target=edge[1]) # 距离始终相对于头
        if distance > 0:
            triplet_data = get_triplet_from_edge(edge)
            if triplet_data:
                # 检查是否已存在（避免重复）
                triplet_key = (triplet_data['head'], triplet_data['relation'], triplet_data['tail'])
                existing_keys = [(t['head'], t['relation'], t['tail']) for t in ripples[f'd{distance}']]
                if triplet_key not in existing_keys:
                    triplet_data['distance'] = distance
                    triplet_data['source_direction'] = 'from_tail'
                    ripples[f'd{distance}'].append(triplet_data)
                
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
    
    # 主进程加载图并选择所有目标三元组
    print("Loading graph...")
    init_worker() 
    if not edges_list:
        print("Error: Could not load graph or graph has no edges.")
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
    for i in range(1, NUM_EXPERIMENTS + 1):
        target_edge = random.choice(edges_list)
        target_triplet = get_triplet_from_edge(target_edge)
        if target_triplet:
            tasks.append((i, target_triplet))

    print(f"Successfully selected {len(tasks)} target triplets for processing.")

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