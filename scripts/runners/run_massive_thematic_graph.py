#!/usr/bin/env python3
"""
Massive Thematic Knowledge Graph Generator
Utilizes the Next-Gen Embedding Resolver and Thematic Filter to prevent drift.
"""

import os
import argparse
from graph_builder.concurrent_builder import ConcurrentGraphBuilder

def generate_massive_graph(theme: str, node_count: int, output_name: str):
    print(f"🚀 Starting MASSIVE thematic knowledge graph generation...")
    print(f"🎯 Target Theme: '{theme}'")
    print(f"🎯 Target Node Count: {node_count}")
    print("-" * 50)

    # Ensure output directory exists
    os.makedirs('data/massive_graphs', exist_ok=True)

    config = {
        'target_nodes': node_count,
        'triplets_per_query': 15,
        'verbose': True,
        'max_workers': args.workers,
                'use_wikidata_validation': False,
        'backup_seeds': [
            'Algorithm', 'Data Structure', 'Artificial Intelligence', 'Machine Learning', 
            'Software Engineering', 'Database', 'Operating System', 'Computer Network', 
            'Computer Architecture', 'Cryptography', 'Cybersecurity', 'Web Development', 
            'Compiler', 'Programming Language', 'Distributed System', 'Cloud Computing', 
            'Computer Graphics', 'Human-Computer Interaction', 'Bioinformatics', 'Quantum Computing',
            'Alan Turing', 'Python', 'C++', 'Java', 'Linux', 'Unix', 'Windows', 'Deep Learning',
            'Neural Network', 'Natural Language Processing', 'Computer Vision', 'Robotics',
            'Blockchain', 'Internet of Things', 'Virtual Reality', 'Augmented Reality'
        ],
        
        # --- LLM API Setup (Floodgate Support) ---
        'api_key_path': 'configs/api_keys.json', # Can be missing if Floodgate is used
        'llm_base_url': 'https://api.deepseek.com/v1',
        'llm_model': 'deepseek-chat',
        'cache_dir': 'data/cache',
        
        # --- Next-Gen Capabilities ---
        'target_theme': theme,
        'thematic_strictness': 0.20,
        'embedding_threshold': 0.95,
        
        # --- Base Config ---
        'confidence_threshold': 0.6,
        'candidate_threshold': 0.5,
        'use_qa_atomic_ontology': False,
        'parallel_domain_diversity': True,
        'parallel_min_domains': 5,
        'global_relation_soft_cap': 0.15,
        
        'export': {
            'auto_export': True,
            'export_interval': 1000,
            'export_format': 'pkl',
            'base_filename': output_name,
            'export_dir': 'data/massive_graphs'
        }
    }

    builder = ConcurrentGraphBuilder(config)
    
    # Dynamic Initial Seeds based on theme
    seeds = []
    theme_lower = theme.lower()
    
    if "biology" in theme_lower or "life" in theme_lower or "medicine" in theme_lower:
        seeds = [
            ("Cell Biology", "is a branch of", "Biology"),
            ("Genetics", "studies", "DNA"),
            ("Charles Darwin", "proposed", "Evolution by natural selection")
        ]
    elif "computer" in theme_lower or "tech" in theme_lower or "software" in theme_lower:
        seeds = [
            ("Alan Turing", "contributed to", "Computer Science"),
            ("Computer Science", "is a subfield of", "Science")
        ]
    else:
        seeds = [
            (f"{theme} Fundamentals", "is the basis of", theme),
            (f"History of {theme}", "chronicles", theme)
        ]

    if not builder.initialize_api():
        print("⚠️ API initialization returned False, but proceeding anyway (Floodgate might not need the key file).")

    # Bypass strict validation for seeds to bootstrap the queues
    for h, r, t in seeds:
        builder.scheduler.add_seed_entities([h, t])
    
    try:
        final_graph = builder.build_graph()
        print(f"\n✅ Generation Complete! Generated {final_graph.number_of_nodes()} nodes.")
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user. Saving checkpoint...")
        if hasattr(builder, 'export_system'):
             builder.export_system.export_graph(builder.graph, force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a massive thematic knowledge graph.")
    parser.add_argument("--theme", type=str, default="Biology and Life Sciences", help="Target theme for the graph")
    parser.add_argument("--nodes", type=int, default=100000, help="Target node count")
    parser.add_argument("--name", type=str, default="massive_biology_graph", help="Output file base name")
    parser.add_argument("--workers", type=int, default=15, help="Number of concurrent LLM threads")
    args = parser.parse_args()
    
    generate_massive_graph(args.theme, args.nodes, args.name)
