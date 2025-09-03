#!/usr/bin/env python3
"""
Build a 10K node knowledge graph using the enhanced graph builder.
"""

import json
import sys
sys.path.append('.')

from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder

def main():
    """Build a 10K node knowledge graph."""
    
    # Configuration for 10K node graph
    config = {
        # Core settings
        'target_size': 10000,
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        
        # Enhanced settings for large graph
        'use_qa_atomic_ontology': True,  # Use function-like relations for better quality
        
        # Validation thresholds (slightly relaxed for scale)
        'confidence_threshold': 0.25,
        'candidate_threshold': 0.20,
        
        # Anti-explosion controls
        'per_entity_caps': {
            'InstanceOf': 3,
            'LocatedIn': 3,
            'PartOf': 3,
            'WorksAt': 2,
            'StudiedAt': 2,
            'BornIn': 1,
            'DiedIn': 1,
        },
        'global_relation_soft_cap': 0.12,
        
        # Stratified BFS settings
        'group_quotas': {
            'people': 0.30,      # 30% people
            'places': 0.25,      # 25% places  
            'organizations': 0.20, # 20% organizations
            'concepts': 0.15,    # 15% concepts
            'events': 0.10       # 10% events
        },
        
        # Parallel expansion settings
        'parallel_domain_diversity': True,
        'parallel_min_domains': 4,
        
        # Monitoring and export
        'checkpoint_interval': 500,
        'checkpoint_dir': './checkpoints/enhanced_10k',
        'export_dir': './results/enhanced_10k',
        
        # Early stopping criteria
        'target_clustering_coefficient': 0.15,
        'target_triangle_count': 1000,
        'min_relation_entropy': 2.5,
    }
    
    print(f"🚀 Starting 10K node graph construction...")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create builder
    builder = EnhancedGraphBuilder(config)
    
    # Define diverse seed entities for a rich graph
    seeds = [
        # Technology leaders
        "Elon Musk", "Steve Jobs", "Bill Gates", "Mark Zuckerberg", "Jeff Bezos",
        "Sundar Pichai", "Satya Nadella", "Tim Cook", "Larry Page", "Sergey Brin",
        
        # Major companies
        "Apple Inc.", "Microsoft Corporation", "Google LLC", "Amazon.com Inc.", "Tesla Inc.",
        "Meta Platforms", "Netflix Inc.", "Nvidia Corporation", "Intel Corporation", "IBM",
        
        # World cities
        "New York City", "London", "Tokyo", "Paris", "Berlin", "San Francisco", 
        "Los Angeles", "Chicago", "Boston", "Seattle", "Austin", "Singapore",
        
        # Universities
        "Harvard University", "MIT", "Stanford University", "University of California Berkeley",
        "Oxford University", "Cambridge University", "Princeton University", "Yale University",
        
        # Countries
        "United States", "United Kingdom", "Germany", "France", "Japan", "China",
        "Canada", "Australia", "India", "Brazil", "Russia", "South Korea",
        
        # Industries and concepts
        "Artificial Intelligence", "Machine Learning", "Blockchain", "Renewable Energy",
        "Space Exploration", "Biotechnology", "Quantum Computing", "Robotics",
        "Climate Change", "Sustainable Development", "Digital Transformation",
        
        # Historical figures
        "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "Isaac Newton",
        "Charles Darwin", "Nikola Tesla", "Thomas Edison", "Winston Churchill",
        
        # Cultural and scientific institutions
        "NASA", "CERN", "World Health Organization", "United Nations", "European Union",
        "Nobel Prize", "Olympic Games", "World Bank", "International Monetary Fund"
    ]
    
    print(f"📍 Using {len(seeds)} diverse seed entities")
    
    try:
        # Build the graph
        final_graph = builder.build_graph_from_seeds(seeds)
        
        print(f"✅ Graph construction completed!")
        print(f"📊 Final graph size: {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges")
        
        # Export results
        builder.export_final_results()
        print(f"💾 Results exported to {config['export_dir']}")
        
    except Exception as e:
        print(f"❌ Error during graph construction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
