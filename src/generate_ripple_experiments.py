#!/usr/bin/env python3
"""
Generate ripple effect experiments from a dense knowledge graph - SIMPLIFIED VERSION
专门为置信度计算设计，只生成三元组，不生成问题，使用多进程加速
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

# Configuration
GRAPH_FILE = 'data/dense_knowledge_graph.pkl'
OUTPUT_DIR = 'results/experiments_ripple_simple'
NUM_EXPERIMENTS = 500
MAX_DISTANCE = 10  # 减小距离以提高效率
NUM_PROCESSES = min(32, mp.cpu_count())  # 使用最多32个进程

# Global variables for sharing across processes
G = None
edges_list = None

def init_worker():
    """Initialize worker process with global graph."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global G, edges_list
    if G is None:
        with open(GRAPH_FILE, 'rb') as f:
            G = pickle.load(f)
        edges_list = list(G.edges())

def get_triplet_from_edge(edge) -> Optional[Tuple[str, str, str]]:
    """Convert a networkx edge to a triplet."""
    global G
    head, tail = edge[0], edge[1]
    edge_data = G.get_edge_data(head, tail)
    
    if not edge_data:
        return None
        
    first_edge_key = next(iter(edge_data))
    first_edge_data = edge_data[first_edge_key]
    
    relation = (first_edge_data.get('relation', 'relates to') 
               if isinstance(first_edge_data, dict) 
               else str(first_edge_data))
    
    return (head, relation, tail)

def find_ripples(target_triplet) -> Dict[str, List[Tuple[str, str, str]]]:
    """Find ripples using BFS."""
    global G
    target_head, _, target_tail = target_triplet
    
    ripples = {f'd{i}': [] for i in range(1, MAX_DISTANCE + 1)}
    visited_nodes = {target_head, target_tail}
    edges_in_ripples = {(target_head, target_tail), (target_tail, target_head)}
    queue = deque([(target_head, 0), (target_tail, 0)])

    while queue:
        current_node, current_distance = queue.popleft()
        
        if current_distance >= MAX_DISTANCE:
            continue
            
        next_distance = current_distance + 1
        neighbors = set(G.successors(current_node)) | set(G.predecessors(current_node))
        
        for neighbor in neighbors:
            if neighbor in visited_nodes:
                continue
                
            visited_nodes.add(neighbor)
            queue.append((neighbor, next_distance))
            
            for edge_pair in [(current_node, neighbor), (neighbor, current_node)]:
                if edge_pair not in edges_in_ripples and G.has_edge(*edge_pair):
                    triplet = get_triplet_from_edge(edge_pair)
                    if triplet:
                        ripples[f'd{next_distance}'].append(triplet)
                        edges_in_ripples.add(edge_pair)
    
    return ripples

def process_experiment(experiment_id: int) -> Optional[Dict]:
    """Process a single experiment."""
    try:
        target_edge = random.choice(edges_list)
        target_triplet = get_triplet_from_edge(target_edge)
        
        if not target_triplet:
            return None
            
        ripples = find_ripples(target_triplet)
        
        experiment_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'target': {
                'triplet': list(target_triplet),
                'head': target_triplet[0],
                'relation': target_triplet[1],
                'tail': target_triplet[2]
            },
            'ripples': {}
        }
        
        for distance in range(1, MAX_DISTANCE + 1):
            key = f'd{distance}'
            triplets = ripples[key]
            experiment_data['ripples'][key] = [
                {
                    'triplet': list(t),
                    'head': t[0],
                    'relation': t[1],
                    'tail': t[2]
                }
                for t in triplets
            ]
        
        experiment_data['statistics'] = {
            'total_triplets': sum(len(ripples[f'd{i}']) for i in range(1, MAX_DISTANCE + 1)),
            'triplets_per_distance': {f'd{i}': len(ripples[f'd{i}']) for i in range(1, MAX_DISTANCE + 1)}
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
    print(f"Target: {NUM_EXPERIMENTS} experiments")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pool = mp.Pool(NUM_PROCESSES, initializer=init_worker)
    
    try:
        results = list(tqdm(
            pool.imap_unordered(process_experiment, range(1, NUM_EXPERIMENTS + 1)),
            total=NUM_EXPERIMENTS,
            desc="Generating experiments"
        ))
        
        successful = sum(1 for r in results if r is not None)
        
        print(f"\n{'='*60}")
        print(f"Experiment generation completed!")
        print(f"{'='*60}")
        print(f"Successfully generated {successful}/{NUM_EXPERIMENTS} experiments")
        print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Cleaning up...")
        pool.terminate()
        pool.join()
        sys.exit(1)
        
    finally:
        pool.close()
        pool.join()

if __name__ == '__main__':
    main() 