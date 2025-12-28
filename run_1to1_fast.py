#!/usr/bin/env python3
"""
Script to run the ConcurrentGraphBuilder for fast generation of a 1-to-1 knowledge graph.
Uses multi-threading to parallelize LLM calls and Wikidata validation.
"""

import os
from graph_builder.concurrent_builder import ConcurrentGraphBuilder

def generate_1to1_graph_fast(node_count: int = 20000):
    """
    Configures and runs the concurrent graph building process.
    """
    print("🚀 Starting FAST CONCURRENT 1-TO-1 knowledge graph generation process...")
    print(f"🎯 Target node count: {node_count}")
    print("-" * 50)

    # Backup seeds to prevent stalling (Extended to ~200 entities for "Infinite" scaling)
    backup_seeds = [
        # G20 + Major Nations
        "United States", "China", "Germany", "Japan", "India", "United Kingdom", "France", "Italy", "Brazil", "Canada",
        "Russia", "South Korea", "Spain", "Australia", "Mexico", "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Switzerland",
        "Sweden", "Poland", "Belgium", "Thailand", "Iran", "Austria", "Norway", "United Arab Emirates", "Israel", "South Africa",
        "Argentina", "Egypt", "Nigeria", "Pakistan", "Vietnam", "Bangladesh", "Philippines", "Malaysia", "Singapore", "Colombia",
        
        # Global Cities (Capitals & Financial Hubs)
        "New York City", "London", "Tokyo", "Paris", "Beijing", "Shanghai", "Los Angeles", "Berlin", "Mumbai", "Dubai",
        "Moscow", "Seoul", "Sao Paulo", "Toronto", "Sydney", "Rome", "Madrid", "Chicago", "Hong Kong", "Singapore",
        "Bangkok", "Istanbul", "Cairo", "Tehran", "Jakarta", "Lagos", "Karachi", "Dhaka", "Manila", "Kuala Lumpur",
        "Buenos Aires", "Rio de Janeiro", "Mexico City", "Lima", "Bogota", "Santiago", "Johannesburg", "Nairobi", "Addis Ababa",
        
        # Tech & Corporate Giants
        "Google", "Microsoft", "Apple Inc.", "Amazon.com", "Meta Platforms", "Tesla, Inc.", "NVIDIA", "Berkshire Hathaway", "Tencent", "Alibaba Group",
        "Samsung Electronics", "Taiwan Semiconductor Manufacturing Company", "Intel", "IBM", "Oracle Corporation", "Netflix", "Adobe Inc.", "Cisco Systems", "Huawei", "Sony",
        "Toyota", "Volkswagen Group", "Mercedes-Benz", "BMW", "Honda", "General Motors", "Ford Motor Company", "Stellantis", "Hyundai Motor Company",
        "Walmart", "Costco", "Home Depot", "Target Corporation", "McDonald's", "Starbucks", "Coca-Cola", "PepsiCo", "Nike, Inc.", "Adidas",
        
        # Top Universities (Knowledge Hubs)
        "Harvard University", "Massachusetts Institute of Technology", "Stanford University", "University of Cambridge", "University of Oxford",
        "California Institute of Technology", "Princeton University", "Yale University", "Columbia University", "University of Chicago",
        "University of Pennsylvania", "Johns Hopkins University", "University of California, Berkeley", "ETH Zurich", "University of Toronto",
        "Tsinghua University", "Peking University", "National University of Singapore", "University of Tokyo", "University of Melbourne",
        
        # Historical & Cultural Figures
        "Albert Einstein", "Isaac Newton", "Leonardo da Vinci", "Marie Curie", "Charles Darwin", "Nikola Tesla", "Galileo Galilei", "Aristotle",
        "William Shakespeare", "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Vincent van Gogh", "Pablo Picasso", "Michelangelo",
        "Mahatma Gandhi", "Nelson Mandela", "Martin Luther King Jr.", "Winston Churchill", "Abraham Lincoln", "George Washington",
        
        # Modern Figures (Politicians, CEOs, Artists)
        "Barack Obama", "Donald Trump", "Joe Biden", "Xi Jinping", "Vladimir Putin", "Angela Merkel", "Emmanuel Macron", "Narendra Modi",
        "Elon Musk", "Jeff Bezos", "Bill Gates", "Mark Zuckerberg", "Tim Cook", "Satya Nadella", "Sundar Pichai", "Warren Buffett",
        "Taylor Swift", "Beyonce", "Cristiano Ronaldo", "Lionel Messi", "LeBron James", "Serena Williams", "Roger Federer",
        
        # International Orgs & Concepts
        "United Nations", "European Union", "World Health Organization", "NATO", "World Bank", "International Monetary Fund", "Red Cross",
        "Nobel Prize", "Academy Awards", "Olympic Games", "FIFA World Cup",
        "Artificial Intelligence", "Machine Learning", "Quantum Computing", "Climate Change", "Renewable Energy", "Space Exploration"
    ]

    # 1. --- Configuration ---
    config = {
        'target_nodes': node_count,
        'triplets_per_query': 8,
        'verbose': True,
        
        # Parallelization
        'max_workers': 16,  # Utilize 16 threads for parallel processing
        
        # KEY: Enable strict 1-to-1 enforcement
        'strict_1to1': True,
        
        # Validation
        'use_wikidata_validation': True, 

        # Paths (Separate from the main run)
        'api_key_path': 'keys/openai_key.txt',
        'cache_dir': f'cache/llm_1to1_fast_{node_count}',
        'checkpoint_dir': f'checkpoints/run_1to1_fast_{node_count}',
        'output_dir': f'results/run_1to1_fast_{node_count}',
        
        'backup_seeds': backup_seeds, # Inject backup seeds

        'random_seed': 123 # Different seed
    }

    # Ensure all necessary directories exist
    for path in [config['cache_dir'], config['checkpoint_dir'], config['output_dir']]:
        os.makedirs(path, exist_ok=True)

    # 2. --- Seed Triplets ---
    seed_triplets = [
        ("France", "CapitalCityOfCountry", "Paris"),
        ("Apple Inc.", "ChiefExecutiveOfficerCurrent", "Tim Cook"),
        ("United Kingdom", "CapitalCityOfCountry", "London"),
        ("Microsoft", "ChiefExecutiveOfficerCurrent", "Satya Nadella"),
        ("Google", "ParentOrganization", "Alphabet Inc."),
        ("Tokyo", "CapitalCityOfCountry", "Japan"),
        ("Elon Musk", "ChiefExecutiveOfficerCurrent", "Tesla, Inc.")
    ]

    # 3. --- Initialize and Run Builder ---
    builder = ConcurrentGraphBuilder(config)

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

    print("\n[Step 3/5] Starting main graph construction loop (Concurrent Mode)...")
    final_graph = builder.build_graph()

    print("\n[Step 4/5] Exporting final graph and results...")
    output_paths = builder.export_results(filename_prefix=f"graph_fast_{node_count}")
    
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
    generate_1to1_graph_fast(node_count=20000)
