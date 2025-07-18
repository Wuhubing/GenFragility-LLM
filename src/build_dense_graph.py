#!/usr/bin/env python3
"""
Build a dense knowledge graph with approximately 5,000 nodes using BFS expansion.
Starting from 5 diverse seed triplets, we expand the graph by querying upstream 
and downstream relationships for each entity.
"""

import json
import os
import pickle
import time
from collections import deque
from datetime import datetime
import networkx as nx
from llm_calls_v2 import find_downstream_triplets, find_upstream_triplets, find_parallel_triplets, load_api_key

# Load API key - 确保从正确路径加载
if not load_api_key('../keys/openai.txt'):
    if not load_api_key('keys/openai.txt'):
        print("❌ 无法加载API密钥，请检查keys/openai.txt文件是否存在")
        exit(1)

# Define diverse seed triplets from different domains
SEED_TRIPLETS = [
    # # Science
    # ('CRISPR', 'is a technology for', 'gene editing'),
    # # History  
    # ('The Silk Road', 'was a historic', 'trade route'),
    # # Art
    # ('Vincent van Gogh', 'is known for his work in', 'Post-Impressionism'),
    # # Geography
    # ('The Amazon Rainforest', 'is located in', 'South America'),
    # # Technology
    # ('The Internet', 'was developed from', 'ARPANET'),
    # # Additional seeds for more diversity
    # ('Albert Einstein', 'developed', 'theory of relativity'),
    ('Paris', 'is the capital of', 'France'),
    # ('Shakespeare', 'wrote', 'Hamlet'),
    # ('DNA', 'contains', 'genetic information'),
    # ('The Pacific Ocean', 'is the largest', 'ocean')
]

# Configuration
TARGET_NODES = 100000  # 增加到50万节点，构建更大的图
TRIPLETS_PER_QUERY = 20  # 增加每次查询的三元组数量
SAVE_INTERVAL = 100  # 每100个节点保存一次检查点
CHECKPOINT_FILE = 'data/dense_graph_checkpoint.pkl'
FINAL_OUTPUT_FILE = 'data/dense_knowledge_graph.pkl'

def extract_entities_from_triplets(triplets):
    """Extract unique entities from a list of triplets."""
    entities = set()
    for triplet in triplets:
        if len(triplet) >= 3:
            entities.add(triplet[0])  # head entity
            entities.add(triplet[2])  # tail entity
    return entities

def build_dense_graph():
    """Main function to build the dense knowledge graph."""
    # Check for existing checkpoint
    G, processed_entities, entity_queue, triplet_queue, processed_triplets = load_checkpoint()
    
    # Main BFS expansion loop
    api_calls = 0
    start_time = time.time()
    
    if G is None:
        G = nx.MultiDiGraph()
    if processed_entities is None:
        processed_entities = set()
    if entity_queue is None:
        entity_queue = deque()
    if triplet_queue is None:
        triplet_queue = deque()
    if processed_triplets is None:
        processed_triplets = set()
        
    is_fresh_start = not G.nodes()

    if is_fresh_start:
        # Initialize graph and tracking structures for a fresh start
        print(f"Starting graph construction at {datetime.now()}")
        print(f"Target: {TARGET_NODES} nodes\n")
        
        for triplet in SEED_TRIPLETS:
            head, relation, tail = triplet
            if not G.has_edge(head, tail, key=relation):
                G.add_edge(head, tail, key=relation, relation=relation)
            print(f"Added seed: {triplet}")
        
        initial_entities = extract_entities_from_triplets(SEED_TRIPLETS)
        entity_queue.extend(initial_entities)
        triplet_queue.extend(SEED_TRIPLETS)
        print(f"\nInitial queue contains {len(entity_queue)} entities and {len(triplet_queue)} triplets")
        print(f"Initial graph has {G.number_of_nodes()} nodes\n")
    else:
        print(f"Resuming from checkpoint...")
        print(f"Current progress: {G.number_of_nodes()}/{TARGET_NODES} nodes")
        print(f"Entities processed: {len(processed_entities)}")
        print(f"Entity queue length: {len(entity_queue)}")
        print(f"Triplet queue length: {len(triplet_queue)}\n")
    
    # Use a simple step counter for forced alternation
    step_counter = 0
    
    while G.number_of_nodes() < TARGET_NODES and (entity_queue or triplet_queue):
        
        # Every 6th step (approx. 1 out of 6), process a relation if possible
        is_relation_step = (step_counter % 6 == 0)
        step_counter += 1

        if is_relation_step and triplet_queue:
            # Process a relation (parallel expansion)
            current_triplet = triplet_queue.popleft()
            if current_triplet in processed_triplets:
                continue
            processed_triplets.add(current_triplet)

            _, relation, _ = current_triplet
            
            nodes_count = G.number_of_nodes()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing Relation: '{relation}' [BRIDGE BUILDING]")
            print(f"Progress: {nodes_count}/{TARGET_NODES} nodes | Entity Queue: {len(entity_queue)} | Triplet Queue: {len(triplet_queue)}")
            
            try:
                # Find parallel relationships
                parallel = find_parallel_triplets(relation, num_triplets=TRIPLETS_PER_QUERY)
                api_calls += 1
                
                new_entities_found_count = 0
                for head, rel, tail in parallel:
                    if not G.has_edge(head, tail, key=rel):
                        G.add_edge(head, tail, key=rel, relation=rel)
                        
                        if head not in processed_entities:
                            entity_queue.append(head)
                            new_entities_found_count += 1
                        if tail not in processed_entities:
                            entity_queue.append(tail)
                            new_entities_found_count += 1

                print(f"  Added {len(parallel)} parallel triplets, found {new_entities_found_count} new entities.")

            except Exception as e:
                print(f"  Error processing relation '{relation}': {e}")
        
        elif entity_queue:
            # Process an entity (up/downstream expansion)
            current_entity = entity_queue.popleft()
            if current_entity in processed_entities:
                continue
            processed_entities.add(current_entity)
            
            nodes_count = G.number_of_nodes()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing Entity: '{current_entity}'")
            print(f"Progress: {nodes_count}/{TARGET_NODES} nodes | Entity Queue: {len(entity_queue)} | Triplet Queue: {len(triplet_queue)}")

            try:
                # Find downstream and upstream relationships
                downstream = find_downstream_triplets(current_entity, num_triplets=TRIPLETS_PER_QUERY)
                api_calls += 1
                upstream = find_upstream_triplets(current_entity, num_triplets=TRIPLETS_PER_QUERY)
                api_calls += 1

                new_triplets = downstream + upstream
                new_entities_found_count = 0
                
                for head, relation, tail in new_triplets:
                    if not G.has_edge(head, tail, key=relation):
                        G.add_edge(head, tail, key=relation, relation=relation)
                        triplet_queue.append((head, relation, tail))
                        
                        if head not in processed_entities:
                            entity_queue.append(head)
                            new_entities_found_count += 1
                        if tail not in processed_entities:
                            entity_queue.append(tail)
                            new_entities_found_count += 1
                
                print(f"  Added {len(new_triplets)} up/downstream triplets, found {new_entities_found_count} new entities.")

            except Exception as e:
                print(f"  Error processing entity '{current_entity}': {e}")
        
        else:
            # Fallback to triplet queue if entity queue is empty
            if triplet_queue:
                step_counter -=1 # ensure we don't skip a relation step
                continue
            else:
                break # Both queues are empty

        # Save checkpoint periodically
        nodes_count = G.number_of_nodes()
        if nodes_count > 0 and nodes_count % SAVE_INTERVAL == 0:
            save_checkpoint(G, processed_entities, entity_queue, triplet_queue, processed_triplets)
            print(f"  Checkpoint saved at {nodes_count} nodes")

        # Rate limiting - 减少延迟以提高效率
        time.sleep(0.3)  # 减少延迟到0.3秒以提高构建速度

    # Final statistics
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Graph construction completed!")
    print(f"{'='*60}")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Entities processed: {len(processed_entities)}")
    print(f"API calls made: {api_calls}")
    print(f"Time elapsed: {duration/60:.1f} minutes")
    print(f"Average time per entity: {duration/len(processed_entities):.1f} seconds")
    
    # Save final graph
    print(f"\nSaving final graph to {FINAL_OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
    with open(FINAL_OUTPUT_FILE, 'wb') as f:
        pickle.dump(G, f)
    
    # Basic statistics only to avoid type issues
    print(f"\nGraph statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    
    return G

def save_checkpoint(G, processed_entities, entity_queue, triplet_queue, processed_triplets):
    """Save current state as checkpoint."""
    checkpoint = {
        'graph': G,
        'processed_entities': processed_entities,
        'entity_queue': list(entity_queue),
        'triplet_queue': list(triplet_queue),
        'processed_triplets': processed_triplets,
        'timestamp': datetime.now()
    }
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint():
    """Load checkpoint if it exists, returning valid data structures."""
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Loaded checkpoint from {checkpoint['timestamp']}")
        G = checkpoint.get('graph', nx.MultiDiGraph())
        processed_entities = checkpoint.get('processed_entities', set())
        entity_queue = deque(checkpoint.get('entity_queue', []))
        triplet_queue = deque(checkpoint.get('triplet_queue', []))
        processed_triplets = checkpoint.get('processed_triplets', set())
        return G, processed_entities, entity_queue, triplet_queue, processed_triplets
        
    except FileNotFoundError:
        print("No checkpoint found. Starting fresh.")
        return nx.MultiDiGraph(), set(), deque(), deque(), set()

def resume_or_start_fresh():
    """Resume from checkpoint or start fresh."""
    # This function is now largely superseded by the logic in build_dense_graph
    # but we keep it for potential standalone use or inspection.
    G, processed_entities, entity_queue, triplet_queue, processed_triplets = load_checkpoint()
    
    if G.nodes():
        print("Found checkpoint! Resuming from saved state...")
        print(f"Current nodes: {G.number_of_nodes()}")
        print(f"Entities processed: {len(processed_entities)}")
        print(f"Entity Queue length: {len(entity_queue)}")
        print(f"Triplet Queue length: {len(triplet_queue)}\n")
    else:
        print("No checkpoint found. Will start fresh.")
    return G, processed_entities, entity_queue, triplet_queue, processed_triplets

if __name__ == '__main__':
    # Build the graph
    final_graph = build_dense_graph()
    
    print(f"\nProcess completed successfully!")
    print(f"Final graph saved to: {FINAL_OUTPUT_FILE}")