#!/usr/bin/env python3
"""
Generate ripple effect experiments from a dense knowledge graph - STRATIFIED SAMPLING VERSION
增强版本：基于节点流行度（In-Degree/Centrality）进行分层采样
目标：生成High/Mid/Low不同流行度的Source节点实验，以验证流行度对Ripple传播和Fake Confidence的影响。
"""

import json
import pickle
import random
import os
from collections import defaultdict, deque, Counter
from datetime import datetime
import networkx as nx
from typing import Dict, Optional, Tuple, List
import multiprocessing as mp
from tqdm import tqdm
import signal
import sys
import networkx.algorithms.community as nx_comm
import gzip
import numpy as np
from openai import OpenAI
import time

# Configuration 
GRAPH_FILE = '/root/GenFragility-LLM/checkpoints/run_1to1_20000/latest.pkl'
OUTPUT_DIR = 'results/experiments_ripples_fast_20k'
NUM_EXPERIMENTS = 15  # 增加实验数量以覆盖不同层级：5 High, 5 Mid, 5 Low
MAX_DISTANCE = 5
NUM_PROCESSES = min(16, mp.cpu_count())

# Global variables for sharing across processes
G = None
edges_list = None
openai_client = None

# Global variables for pre-computed metrics
node_centrality = None
communities = None
node_to_community = None


def init_worker():
    """Initialize worker process with global graph and OpenAI client."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global G, edges_list, openai_client
    
    # Initialize OpenAI client
    if openai_client is None:
        try:
            key_path = '/root/GenFragility-LLM/keys/openai_key.txt'
            if os.path.exists(key_path):
                with open(key_path, 'r') as f:
                    api_key = f.read().strip()
                os.environ['OPENAI_API_KEY'] = api_key
                openai_client = OpenAI()
                # print(f"Worker {os.getpid()} initialized OpenAI client")
            else:
                print(f"Worker {os.getpid()} warning: OpenAI key not found at {key_path}")
        except Exception as e:
            print(f"Worker {os.getpid()} warning: Failed to init OpenAI: {e}")

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
        actual_edge_data = None
        if graph.is_multigraph():
             if len(edge_data) > 0:
                 first_key = next(iter(edge_data.keys()))
                 actual_edge_data = edge_data[first_key]
        else:
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

    if G.has_edge(target_head, target_tail):
        processed_edges.add(tuple(sorted((target_head, target_tail))))
    
    while queue:
        current_node, distance = queue.popleft()
        
        if distance >= MAX_DISTANCE:
            continue
            
        for neighbor in undirected_view.neighbors(current_node):
            edge_key = tuple(sorted((current_node, neighbor)))
            if edge_key in processed_edges:
                continue
            
            processed_edges.add(edge_key)
            
            edge_data = None
            if G.has_edge(current_node, neighbor):
                edge_data = get_triplet_from_edge(G, (current_node, neighbor))
            elif G.has_edge(neighbor, current_node):
                edge_data = get_triplet_from_edge(G, (neighbor, current_node))

            if edge_data:
                new_distance = distance + 1
                edge_data['distance'] = new_distance
                ripples[f'd{new_distance}'].append(edge_data)

            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, distance + 1))
                        
    return ripples

def analyze_graph_metrics(target_triplet: Dict, ripples: Dict[str, List[Dict]]):
    """Analyzes and extracts structural metrics for the experiment's subgraph."""
    target_head = target_triplet['head']
    target_tail = target_triplet['tail']
    
    nodes_in_subgraph = {target_head, target_tail}
    for dist_ripples in ripples.values():
        for triplet_data in dist_ripples:
            nodes_in_subgraph.add(triplet_data['head'])
            nodes_in_subgraph.add(triplet_data['tail'])
    
    subgraph = G.subgraph(nodes_in_subgraph)
    subgraph_undirected_view = subgraph.to_undirected(as_view=True)

    subgraph_metrics = {
        "node_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "density": nx.density(subgraph),
        "connected_components": nx.number_connected_components(subgraph_undirected_view),
    }

    target_node_metrics = {
        "head": node_centrality.get(target_head, {}),
        "tail": node_centrality.get(target_tail, {}),
    }
    
    subgraph_communities = set()
    for node in nodes_in_subgraph:
        if node in node_to_community:
            subgraph_communities.add(node_to_community[node])
    
    community_metrics = {
        "spans_multiple_communities": len(subgraph_communities) > 1,
        "community_ids": sorted(list(subgraph_communities)),
        "community_count": len(subgraph_communities)
    }

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
        for triplet_data in triplet_list:
            total_triplets += 1
            question = triplet_data.get('question', '').strip()
            
            if question:
                triplets_with_questions += 1
                avg_question_lengths.append(len(question.split()))
                
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


def generate_question_openai(head, relation, tail):
    """Generate a question for a triplet using OpenAI."""
    global openai_client
    if not openai_client:
        return ""

    prompt = f"""
    Generate a natural, concise question that would elicit the answer "{tail}" for the knowledge relationship ({head}, {relation}, {tail}).

    REQUIREMENTS:
    - Question must be under 15 words
    - Ask about "{head}" to get answer "{tail}"
    - Use simple, clear language
    - Don't include the answer in the question
    - Make it sound natural and conversational

    Examples:
    - For (Eiffel Tower, LocatedIn, Paris): "Where is the Eiffel Tower located?"
    - For (Einstein, BirthYear, 1879): "When was Einstein born?"
    - For (Apple, CEO, Tim Cook): "Who is the CEO of Apple?"

    Your turn:
    Triplet: ({head}, {relation}, {tail})
    Question:
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are an expert at generating clear, natural questions for knowledge facts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=30,
        )
        question = response.choices[0].message.content.strip()
        return question.strip('"').strip()
    except Exception as e:
        # print(f"Error generating question: {e}")
        return ""

def process_experiment(task: Tuple[int, Dict]) -> Optional[Dict]:
    """Processes a single experiment task containing an ID and a target triplet."""
    experiment_id, target_triplet = task
    
    try:
        if not target_triplet:
            return None
            
        # Ensure target triplet has a question
        if not target_triplet.get('question'):
             q = generate_question_openai(target_triplet['head'], target_triplet['relation'], target_triplet['tail'])
             if q:
                 target_triplet['question'] = q

        ripples = find_ripples(target_triplet)

        # Ensure all ripples have questions
        for dist, triplets in ripples.items():
            for t in triplets:
                if not t.get('question'):
                     q = generate_question_openai(t['head'], t['relation'], t['tail'])
                     if q:
                         t['question'] = q
        
        # Analyze graph metrics for this experiment
        analysis_metrics = analyze_graph_metrics(target_triplet, ripples)

        experiment_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'target': target_triplet,
            'ripples': ripples,
            'analysis_metrics': analysis_metrics
        }
        
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
    """Main function with stratified sampling and multiprocessing support."""
    print(f"Starting ripple experiment generation with {NUM_PROCESSES} processes")
    print(f"Target: {NUM_EXPERIMENTS} experiments from graph: {GRAPH_FILE}")
    print(f"Strategy: Stratified Sampling (High/Mid/Low Popularity)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading graph in main process for task selection...")
    global G, edges_list
    G, edges_list = init_worker()
    
    if not G or not edges_list:
        print("Error loading graph in main process. Aborting.")
        return

    print("Pre-computing global graph metrics...")
    global node_centrality, communities, node_to_community
    undirected_view = G.to_undirected(as_view=True)

    print(" -> Calculating Degree Centrality...")
    deg_cen = nx.degree_centrality(G)
    degrees = dict(G.degree())
    
    # Calculate quantiles for stratification
    degree_values = list(degrees.values())
    high_threshold = np.percentile(degree_values, 95)  # Top 5%
    mid_threshold = np.percentile(degree_values, 50)   # Top 50%
    
    print(f" -> Popularity Thresholds: High > {high_threshold:.1f}, Mid > {mid_threshold:.1f}")

    print(" -> Calculating other metrics...")
    pagerank = nx.pagerank(G)
    bet_cen = nx.betweenness_centrality(G)

    node_centrality = {
        node: {
            "degree_centrality": deg_cen.get(node, 0),
            "degree": degrees.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "betweenness_centrality": bet_cen.get(node, 0),
        } for node in G.nodes()
    }

    print(" -> Detecting communities (Louvain)...")
    communities = nx_comm.louvain_communities(undirected_view)
    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    
    # Stratified Sampling Strategy
    all_edges = list(G.edges(data=True))
    candidate_triplets = {'high': [], 'mid': [], 'low': []}
    
    # Helper to check if a node has enough ripple potential
    # (Checking 2-hop neighbor count estimate roughly)
    def has_ripple_potential(node, min_neighbors=2):
        return G.degree(node) >= min_neighbors

    print("Categorizing edges by Head Node Popularity...")
    for u, v, data in all_edges:
        u_deg = degrees[u]
        
        # Ensure head has ripple potential
        if not has_ripple_potential(u):
            continue
            
        category = 'low'
        if u_deg > high_threshold:
            category = 'high'
        elif u_deg > mid_threshold:
            category = 'mid'
            
        candidate_triplets[category].append((u, v, data))
    
    print(f"Candidates pool size: High={len(candidate_triplets['high'])}, Mid={len(candidate_triplets['mid'])}, Low={len(candidate_triplets['low'])}")
    
    selected_edges = []
    samples_per_category = NUM_EXPERIMENTS // 3
    
    used_tails = set() # To ensure tail diversity
    
    for cat in ['high', 'mid', 'low']:
        pool = candidate_triplets[cat]
        random.shuffle(pool)
        
        count = 0
        for edge in pool:
            u, v, data = edge
            
            # Diversity check: try not to reuse tails too much
            # Allow some reuse if pool is small, but prefer fresh tails
            if v in used_tails and len(pool) > samples_per_category * 2:
                continue
                
            selected_edges.append(edge)
            used_tails.add(v)
            count += 1
            if count >= samples_per_category:
                break
        print(f"Selected {count} experiments for category {cat}")

    tasks = []
    for i, target_edge_data in enumerate(selected_edges, 1):
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
                'is_inverse': data.get('is_inverse', False),
                'popularity_category': 'high' if degrees[head] > high_threshold else ('mid' if degrees[head] > mid_threshold else 'low')
            }
            tasks.append((i, target_triplet))

    print(f"Successfully prepared {len(tasks)} tasks.")

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
            print(f"\nNext step: Run the poisoning pipeline for stratified analysis.")
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Cleaning up...")
            pool.terminate()
            pool.join()
            sys.exit(1)
        
if __name__ == '__main__':
    main()
