#!/usr/bin/env python3
"""
Generate ripple effect experiments from a dense knowledge graph.
For each experiment, we select a random edge as the "attack target" and 
perform BFS to find triplets at distances 1-5 from that edge.
"""

import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from llm_calls_v2 import triplet_to_question, load_api_key
from typing import Dict

# Configuration
GRAPH_FILE = 'dense_knowledge_graph_5k.pkl'  # CORRECTED: Use the 5k node graph
OUTPUT_DIR = 'results/experiments_ripple'
NUM_EXPERIMENTS = 20
MAX_DISTANCE = 5
QUESTIONS_PER_TRIPLET = 1  # Generate 1 question per triplet

# Load API key
load_api_key()

def load_graph():
    """Load the dense knowledge graph from pickle file."""
    print(f"Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def get_triplet_from_edge(G, edge):
    """Convert a networkx edge to a triplet."""
    head, tail = edge[0], edge[1]
    # Get the relation between these nodes
    edge_data = G.get_edge_data(head, tail)
    if edge_data and isinstance(edge_data, dict):
        relation = edge_data.get('relation', 'relates to')
        return (head, relation, tail)
    return None

def find_ripples(G, target_triplet, max_distance=5):
    """
    Find all triplets at distances 1 to max_distance from the target triplet.
    Returns a dictionary with keys 'd1', 'd2', ..., 'd5' containing triplets at each distance.
    """
    target_head, _, target_tail = target_triplet
    
    # Initialize result structure
    ripples = {f'd{i}': [] for i in range(1, max_distance + 1)}
    
    # Visited nodes to prevent cycles and redundant processing
    visited_nodes = {target_head, target_tail}
    
    # BFS queue: stores tuples of (node, distance)
    queue = deque([(target_head, 0), (target_tail, 0)])
    
    # This set will store the (head, tail) tuple of edges that have been added
    # to the ripples, to avoid adding the same edge multiple times via different paths.
    edges_in_ripples = set()

    # Add the target edge itself to avoid re-adding it
    if G.has_edge(target_head, target_tail):
        edges_in_ripples.add((target_head, target_tail))
    if G.has_edge(target_tail, target_head):
        edges_in_ripples.add((target_tail, target_head))

    while queue:
        current_node, current_distance = queue.popleft()
        
        # Stop if we have reached the maximum distance
        if current_distance >= max_distance:
            continue
            
        next_distance = current_distance + 1
        
        # Explore neighbors (both successors and predecessors)
        neighbors = list(G.successors(current_node)) + list(G.predecessors(current_node))
        
        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, next_distance))
                
                # This node is at `next_distance`, so any edge connecting
                # `current_node` to `neighbor` belongs to this distance level.
                
                # Check for an edge from current_node to neighbor
                if G.has_edge(current_node, neighbor):
                    edge_key = (current_node, neighbor)
                    if edge_key not in edges_in_ripples:
                        triplet = get_triplet_from_edge(G, edge_key)
                        if triplet:
                            ripples[f'd{next_distance}'].append(triplet)
                            edges_in_ripples.add(edge_key)

                # Check for an edge from neighbor to current_node
                if G.has_edge(neighbor, current_node):
                    edge_key = (neighbor, current_node)
                    if edge_key not in edges_in_ripples:
                        triplet = get_triplet_from_edge(G, edge_key)
                        if triplet:
                            ripples[f'd{next_distance}'].append(triplet)
                            edges_in_ripples.add(edge_key)
    return ripples


def generate_questions_for_triplets(triplets, description=""):
    """Generate questions for a list of triplets."""
    questions = []
    for i, triplet in enumerate(triplets):
        try:
            print(f"    Generating question {i+1}/{len(triplets)} {description}")
            question = triplet_to_question(triplet)
            questions.append({
                'triplet': triplet,
                'question': question
            })
        except Exception as e:
            print(f"      Error generating question for {triplet}: {e}")
            questions.append({
                'triplet': triplet,
                'question': f"What is the relationship between {triplet[0]} and {triplet[2]}?"
            })
    return questions

def generate_experiment(G, experiment_id):
    """Generate a single ripple experiment."""
    print(f"\n{'='*60}")
    print(f"Generating Experiment {experiment_id}")
    print(f"{'='*60}")
    
    # Select random edge as attack target
    edges = list(G.edges())
    target_edge = random.choice(edges)
    target_triplet = get_triplet_from_edge(G, target_edge)
    
    if not target_triplet:
        print(f"Failed to get triplet from edge {target_edge}")
        return None
    
    print(f"Target triplet: {target_triplet}")
    
    # Find ripples
    print(f"Finding ripples up to distance {MAX_DISTANCE}...")
    ripples = find_ripples(G, target_triplet, MAX_DISTANCE)
    
    # Print statistics
    for distance in range(1, MAX_DISTANCE + 1):
        count = len(ripples[f'd{distance}'])
        print(f"  Distance {distance}: {count} triplets")
    
    # Generate questions for target
    print(f"\nGenerating question for target triplet...")
    target_question = triplet_to_question(target_triplet)
    
    # Generate questions for each distance layer
    experiment_data = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'target': {
            'triplet': target_triplet,
            'question': target_question
        },
        'ripples': {}
    }
    
    for distance in range(1, MAX_DISTANCE + 1):
        key = f'd{distance}'
        triplets = ripples[key]
        
        if triplets:
            print(f"\nGenerating questions for distance {distance} ({len(triplets)} triplets)...")
            questions = generate_questions_for_triplets(triplets, f"at distance {distance}")
            experiment_data['ripples'][key] = questions
        else:
            experiment_data['ripples'][key] = []
    
    # Add summary statistics
    experiment_data['statistics'] = {
        'total_triplets': sum(len(ripples[f'd{i}']) for i in range(1, MAX_DISTANCE + 1)),
        'triplets_per_distance': {f'd{i}': len(ripples[f'd{i}']) for i in range(1, MAX_DISTANCE + 1)}
    }
    
    return experiment_data

def save_experiment_to_file(experiment_data: Dict, filename: str):
    """Saves the experiment data to a JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Experiment data saved to {filepath}")

def main():
    """Main function to generate all experiments."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load graph
    G = load_graph()
    
    # Generate experiments
    successful_experiments = 0
    
    for i in range(1, NUM_EXPERIMENTS + 1):
        try:
            experiment = generate_experiment(G, i)
            
            if experiment:
                # Save experiment
                filename = f'ripple_experiment_{i:02d}.json'
                save_experiment_to_file(experiment, filename)
                
                successful_experiments += 1
            
        except Exception as e:
            print(f"\nError generating experiment {i}: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Experiment generation completed!")
    print(f"{'='*60}")
    print(f"Successfully generated {successful_experiments}/{NUM_EXPERIMENTS} experiments")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Print example of how to load and use the experiments
    print(f"\nTo load an experiment:")
    print(f"  with open('{OUTPUT_DIR}/ripple_experiment_01.json', 'r') as f:")
    print(f"      exp = json.load(f)")
    print(f"  print(exp['target'])  # The attacked triplet")
    print(f"  print(len(exp['ripples']['d1']))  # Number of 1-hop triplets")

if __name__ == '__main__':
    main()