#!/usr/bin/env python3
"""
Script to run the EnhancedGraphBuilder for generating a knowledge graph.
This script provides a direct entry point for configuring and executing
a graph building session.
"""

import os
from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder

def generate_graph_with_target_nodes(node_count: int = 20000):
    """
    Configures and runs the graph building process to generate a graph
    with a specific number of nodes.

    Args:
        node_count (int): The target number of nodes for the graph.
    """
    print("🚀 Starting knowledge graph generation process...")
    print(f"🎯 Target node count: {node_count}")
    print("-" * 50)

    # 1. --- Configuration ---
    # Define the configuration for this specific run.
    # All paths are relative to the project root directory.
    config = {
        'target_nodes': node_count,
        'triplets_per_query': 8,
        'verbose': True,
        
        # CRITICAL: Enable the new Wikidata validation step
        'use_wikidata_validation': True, 

        # Define paths for keys, cache, and outputs
        'api_key_path': 'keys/openai_key.txt',
        'cache_dir': f'cache/llm_responses_{node_count}_validated_optimized',
        'checkpoint_dir': f'checkpoints/run_{node_count}_validated_optimized',
        'output_dir': f'results/run_{node_count}_validated_optimized',

        # Use a fixed random seed for reproducible graph generation
        'random_seed': 42
    }

    # Ensure all necessary directories exist
    for path in [config['cache_dir'], config['checkpoint_dir'], config['output_dir']]:
        os.makedirs(path, exist_ok=True)

    # 2. --- Seed Triplets ---
    # A small but diverse set of high-quality seeds to bootstrap the graph.
    # These seeds cover different domains (Person, Org, Work, etc.).
    seed_triplets = [
        ("Albert Einstein", "BirthPlace", "Ulm"),
        ("University of Cambridge", "CountryOfCity", "United Kingdom"),
        ("Apple Inc.", "HeadquartersCity", "Cupertino"),
        ("Python (programming language)", "DevelopedByPrimary", "Python Software Foundation"),
        ("The Lord of the Rings", "AuthorOfWorkPrimary", "J. R. R. Tolkien")
    ]

    # 3. --- Initialize and Run Builder ---
    builder = EnhancedGraphBuilder(config)

    print("\n[Step 1/5] Initializing API connection...")
    if not builder.initialize_api():
        print("❌ API initialization failed. Please check your key at 'keys/openai_key.txt'.")
        return

    # Try to resume from a checkpoint to save progress
    print("\n[Step 2/5] Checking for existing checkpoint...")
    if not builder.load_checkpoint():
        print("   -> No checkpoint found. Starting fresh with seed triplets.")
        builder.add_seed_triplets(seed_triplets)
    else:
        print("   -> ✅ Resumed from the latest checkpoint.")

    print("\n[Step 3/5] Starting main graph construction loop...")
    final_graph = builder.build_graph()

    print("\n[Step 4/5] Exporting final graph and results...")
    output_paths = builder.export_results(filename_prefix=f"graph_{node_count}_nodes_validated")
    
    print("\n[Step 5/5] Process finished!")
    print("-" * 50)
    print("🎉 Graph generation complete!")
    print(f"   Final Node Count: {final_graph.number_of_nodes()}")
    print(f"   Final Edge Count: {final_graph.number_of_edges()}")
    print(f"   Results saved in: {config['output_dir']}")
    for file_type, path in output_paths.items():
        print(f"   - {file_type.capitalize()}: {path}")
    print("-" * 50)


if __name__ == '__main__':
    # This function will execute the graph generation process to create a
    # graph with approximately 20,000 nodes using optimized validation.
    generate_graph_with_target_nodes(node_count=20000)
