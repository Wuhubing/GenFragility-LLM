#!/usr/bin/env python3
"""
Script to continue building from 5000 nodes to 20000 nodes.
This script loads the existing 5000-node graph and continues building.
"""

import os
import shutil
from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder

def continue_graph_to_20k():
    """
    Continues building from the existing 5000-node graph to reach 20000 nodes.
    """
    print("🚀 Continuing knowledge graph generation from 5000 to 20000 nodes...")
    print("🎯 Target node count: 20000")
    print("-" * 50)

    # 1. --- Configuration for 20k nodes ---
    config = {
        'target_nodes': 20000,
        'triplets_per_query': 8,
        'verbose': True,
        
        # CRITICAL: Enable the new Wikidata validation step
        'use_wikidata_validation': True, 

        # Define paths for keys, cache, and outputs
        'api_key_path': 'keys/openai_key.txt',
        'cache_dir': 'cache/llm_responses_20000_validated_optimized',
        'checkpoint_dir': 'checkpoints/run_20000_validated_optimized',
        'output_dir': 'results/run_20000_validated_optimized',

        # Use same random seed for consistency
        'random_seed': 42,
        
        # Checkpoint interval - save more frequently for large graphs
        'checkpoint_interval': 50
    }

    # Ensure all necessary directories exist
    for path in [config['cache_dir'], config['checkpoint_dir'], config['output_dir']]:
        os.makedirs(path, exist_ok=True)

    # 2. --- Copy existing 5k cache to 20k cache (to preserve API calls) ---
    source_cache = 'cache/llm_responses_5000_validated_optimized'
    if os.path.exists(source_cache):
        print("📋 Copying existing cache from 5000-node run...")
        # Copy all files from source cache to target cache
        for filename in os.listdir(source_cache):
            source_file = os.path.join(source_cache, filename)
            target_file = os.path.join(config['cache_dir'], filename)
            if os.path.isfile(source_file) and not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
        print("✅ Cache copied successfully!")

    # 3. --- Copy the 5k checkpoint as starting point ---
    source_checkpoint = 'checkpoints/run_5000_validated_optimized/final.pkl'
    target_checkpoint = os.path.join(config['checkpoint_dir'], 'latest.pkl')
    
    if os.path.exists(source_checkpoint):
        print("📋 Copying 5000-node checkpoint as starting point...")
        shutil.copy2(source_checkpoint, target_checkpoint)
        print("✅ Checkpoint copied successfully!")
    else:
        print("❌ 5000-node checkpoint not found! Please ensure the 5k graph was completed.")
        return

    # 4. --- Initialize and Run Builder ---
    builder = EnhancedGraphBuilder(config)

    print("\n[Step 1/5] Initializing API connection...")
    if not builder.initialize_api():
        print("❌ API initialization failed. Please check your key at 'keys/openai_key.txt'.")
        return

    # Load from the copied checkpoint
    print("\n[Step 2/5] Loading from 5000-node checkpoint...")
    if not builder.load_checkpoint():
        print("❌ Failed to load checkpoint. Something went wrong.")
        return
    else:
        current_nodes = builder.graph.number_of_nodes()
        print(f"   -> ✅ Resumed from checkpoint with {current_nodes} nodes.")
        print(f"   -> 🎯 Target: {config['target_nodes']} nodes ({config['target_nodes'] - current_nodes} more to go)")

    print("\n[Step 3/5] Continuing graph construction to 20000 nodes...")
    final_graph = builder.build_graph()

    print("\n[Step 4/5] Exporting final 20k graph and results...")
    output_paths = builder.export_results(filename_prefix="graph_20000_nodes_validated")
    
    print("\n[Step 5/5] Process finished!")
    print("-" * 50)
    print("🎉 20K Graph generation complete!")
    print(f"   Final Node Count: {final_graph.number_of_nodes()}")
    print(f"   Final Edge Count: {final_graph.number_of_edges()}")
    print(f"   Results saved in: {config['output_dir']}")
    for file_type, path in output_paths.items():
        print(f"   - {file_type.capitalize()}: {path}")
    print("-" * 50)


if __name__ == '__main__':
    # Continue from 5000 nodes to 20000 nodes
    continue_graph_to_20k()
