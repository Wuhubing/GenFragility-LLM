#!/usr/bin/env python3
"""
Massive Thematic Knowledge Graph Generator
Utilizes the Next-Gen Embedding Resolver and Thematic Filter to prevent drift.
"""

import os
import argparse
from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder

def generate_massive_graph(theme: str, node_count: int, output_name: str):
    print(f"🚀 Starting MASSIVE thematic knowledge graph generation...")
    print(f"🎯 Target Theme: '{theme}'")
    print(f"🎯 Target Node Count: {node_count}")
    print("-" * 50)

    # Ensure output directory exists
    os.makedirs('data/massive_graphs', exist_ok=True)

    config = {
        'target_nodes': node_count,
        'triplets_per_query': 15,  # Maximize throughput per API call
        'verbose': True,
        
        # --- Next-Gen Capabilities ---
        'target_theme': theme,
        'thematic_strictness': 0.50,  # Cosine similarity threshold for BFS drift prevention
        'embedding_threshold': 0.95,  # Entity resolution threshold (merging aliases)
        
        # --- Base Config ---
        'confidence_threshold': 0.6,
        'candidate_threshold': 0.5,
        'use_qa_atomic_ontology': False,
        'parallel_domain_diversity': True,
        'parallel_min_domains': 5,
        'api_key_path': 'configs/api_keys.json', 
        'cache_dir': 'data/cache',
        
        # Global Soft caps to prevent relation spam
        'global_relation_soft_cap': 0.15,
        
        'export': {
            'auto_export': True,
            'export_interval': 1000, # Checkpoint every 1000 nodes
            'export_format': 'pkl',
            'base_filename': output_name,
            'export_dir': 'data/massive_graphs'
        }
    }

    builder = EnhancedGraphBuilder(config)
    
    # Dynamic Initial Seeds based on theme
    seeds = []
    theme_lower = theme.lower()
    
    if "biology" in theme_lower or "life" in theme_lower or "medicine" in theme_lower:
        seeds = [
            ("Cell Biology", "is a branch of", "Biology"),
            ("Genetics", "studies", "DNA"),
            ("Charles Darwin", "proposed", "Evolution by natural selection"),
            ("Neuroscience", "investigates", "Brain"),
            ("Ecology", "focuses on", "Ecosystems")
        ]
    elif "computer" in theme_lower or "tech" in theme_lower or "software" in theme_lower:
        seeds = [
            ("Computer Science", "studies", "Algorithms"),
            ("Artificial Intelligence", "is a subfield of", "Computer Science"),
            ("Machine Learning", "uses", "Neural Networks"),
            ("Quantum Computing", "relies on", "Quantum Mechanics")
        ]
    else:
        seeds = [
            (f"{theme} Fundamentals", "is the basis of", theme),
            (f"History of {theme}", "chronicles", theme),
            (f"Advanced {theme}", "builds upon", f"{theme} Fundamentals")
        ]

    if not builder.initialize_api():
        print("❌ Failed to initialize API. Please check 'configs/api_keys.json'.")
        return

    builder.add_seed_triplets(seeds)
    
    try:
        final_graph = builder.build_graph()
        print(f"\n✅ Generation Complete! Generated {final_graph.number_of_nodes()} nodes.")
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user. Saving checkpoint...")
        if hasattr(builder, 'export_system'):
             builder.export_system.export_graph(builder.graph, force=True)
        else:
             print("Export system unavailable, graph state held in memory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a massive thematic knowledge graph.")
    parser.add_argument("--theme", type=str, default="Biology and Life Sciences", help="Target theme for the graph")
    parser.add_argument("--nodes", type=int, default=100000, help="Target node count")
    parser.add_argument("--name", type=str, default="massive_biology_graph", help="Output file base name")
    args = parser.parse_args()
    
    generate_massive_graph(args.theme, args.nodes, args.name)
