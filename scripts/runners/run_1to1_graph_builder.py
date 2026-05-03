#!/usr/bin/env python3
"""
Script to run the EnhancedGraphBuilder for generating a LARGE 1-to-1 knowledge graph.
This script enforces strict 1-to-1 relationship constraints during generation.
"""

import os
from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder

def generate_1to1_graph(node_count: int = 5000):
    """
    Configures and runs the graph building process to generate a graph
    with a specific number of nodes and strict 1-to-1 relationships.

    Args:
        node_count (int): The target number of nodes for the graph.
    """
    print("🚀 Starting STRICT 1-TO-1 knowledge graph generation process...")
    print(f"🎯 Target node count: {node_count}")
    print("-" * 50)

    # 1. --- Configuration ---
    config = {
        'target_nodes': node_count,
        'triplets_per_query': 8,
        'verbose': True,
        
        # KEY: Enable strict 1-to-1 enforcement
        'strict_1to1': True,
        
        # Validation
        'use_wikidata_validation': True, 

        # Paths
        'api_key_path': 'keys/openai_key.txt',
        'cache_dir': f'cache/llm_1to1_{node_count}',
        'checkpoint_dir': f'checkpoints/run_1to1_{node_count}',
        'output_dir': f'results/run_1to1_{node_count}',

        'random_seed': 42
    }

    # Ensure all necessary directories exist
    for path in [config['cache_dir'], config['checkpoint_dir'], config['output_dir']]:
        os.makedirs(path, exist_ok=True)

    # 2. --- Seed Triplets ---
    # Use seeds that support 1-to-1 expansion naturally
    seed_triplets = [
        ("France", "CapitalCityOfCountry", "Paris"),
        ("Apple Inc.", "ChiefExecutiveOfficerCurrent", "Tim Cook"),
        ("United Kingdom", "CapitalCityOfCountry", "London"),
        ("Microsoft", "ChiefExecutiveOfficerCurrent", "Satya Nadella"),
        ("Google", "ParentOrganization", "Alphabet Inc.")
    ]

    # 3. --- Initialize and Run Builder ---
    builder = EnhancedGraphBuilder(config)

    print("\n[Step 1/5] Initializing API connection...")
    if not builder.initialize_api():
        print("❌ API initialization failed. Please check your key at 'keys/openai_key.txt'.")
        return

    print("\n[Step 2/5] Checking for existing checkpoint...")
    if not builder.load_checkpoint():
        print("   -> No checkpoint found. Starting fresh with seed triplets.")
        builder.add_seed_triplets(seed_triplets)
    else:
        print("   -> ✅ Resumed from the latest checkpoint.")
        
        # INJECT FRESH SEEDS TO RESUME STALLED PROCESS
        # This is necessary because the previous run stopped due to empty queue.
        extra_seeds = [
            ("Hamlet", "AuthorOfWorkPrimary", "William Shakespeare"),
            ("Nintendo", "HeadquartersCountry", "Japan"),
            ("Linux", "CreatedByPrimary", "Linus Torvalds"),
            ("Berlin", "CountryOfCity", "Germany"), 
            ("Minecraft", "DevelopedByPrimary", "Mojang Studios"),
            ("United Nations", "HeadquartersCity", "New York City")
        ]
        print(f"   -> 💉 Injecting {len(extra_seeds)} new seeds to restart expansion...")
        builder.add_seed_triplets(extra_seeds)

    print("\n[Step 3/5] Starting main graph construction loop...")
    final_graph = builder.build_graph()

    print("\n[Step 4/5] Exporting final graph and results...")
    output_paths = builder.export_results(filename_prefix=f"graph_1to1_{node_count}")
    
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
    # Generate a large 1-to-1 graph
    generate_1to1_graph(node_count=20000)

