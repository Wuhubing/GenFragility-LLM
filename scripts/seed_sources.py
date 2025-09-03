from typing import List

def get_optimized_specific_seeds() -> List[str]:
    """
    Returns an expanded and more diverse list of seeds to support larger graph generation.
    """
    return [
        # Tech Companies (Specific Organizations) - Expanded
        "Apple Inc.", "Microsoft Corporation", "Google LLC", "Amazon (company)", "Tesla Inc.", "OpenAI", 
        "Meta Platforms", "Nvidia", "Intel", "IBM", "Oracle Corporation", "Salesforce", "Adobe Inc.",
        "Samsung Electronics", "Sony", "Tencent", "Alibaba Group",
        
        # Famous People (Specific Individuals) - Expanded
        "Albert Einstein", "Marie Curie", "Steve Jobs", "Elon Musk", "Bill Gates", "Mark Zuckerberg", 
        "Tim Cook", "Satya Nadella", "Jeff Bezos", "Larry Page", "Sergey Brin", "Warren Buffett",
        "Leonardo da Vinci", "Isaac Newton", "Galileo Galilei",
        
        # Specific Geographic Locations - Expanded
        "Beijing", "New York City", "London", "Tokyo", "San Francisco", "Paris", "Shanghai", "Berlin",
        "Singapore", "Dubai", "Sydney", "Toronto", "Moscow", "Rome", "Seoul", "Mumbai",
        
        # Specific Universities/Institutions - Expanded
        "Harvard University", "MIT", "Stanford University", "Cambridge University", "Tsinghua University", 
        "Oxford University", "ETH Zurich", "University of California, Berkeley", "Yale University",
        "Peking University", "National University of Singapore",
        
        # Major World Companies from Diverse Sectors
        "Berkshire Hathaway", "Johnson & Johnson", "Procter & Gamble", "Visa Inc.", "Mastercard",
        "JPMorgan Chase", "Bank of America", "The Coca-Cola Company", "PepsiCo", "Walmart", "Toyota",
        "Volkswagen Group", "Ford Motor Company", "General Electric", "ExxonMobil", "Shell plc",
        "Saudi Aramco", "Nestlé", "LVMH",
        
        # Specific Company Products/Services - Diversified
        "iPhone", "Windows 11", "Tesla Model S", "ChatGPT", "Gmail", "Amazon Web Services", "Coca-Cola",
        "Visa card", "Toyota Corolla", "Boeing 747",
        
        # Specific Countries - Expanded
        "United States", "China", "Germany", "Japan", "United Kingdom", "France", "India", "Canada",
        "Australia", "Brazil", "Russia", "South Korea", "Italy", "Spain",
        
        # Specific Cities relevant to various industries
        "Cupertino", "Redmond", "Mountain View", "Palo Alto", "Seattle", "Cambridge", "Wall Street",
        "Detroit", "Hollywood", "Zurich", "Frankfurt"
    ]

def create_specific_seed_batches(batch_size: int = 3) -> List[List[str]]:
    """
    Creates high-quality seed batches ensuring entities within each batch are related.
    Focuses on thematic combinations.
    """
    all_seeds = get_optimized_specific_seeds() # Use the function above for consistency

    thematic_groups = [
        # Apple Ecosystem
        ["Apple Inc.", "Steve Jobs", "iPhone", "Tim Cook", "Cupertino"],
        # Microsoft Ecosystem
        ["Microsoft Corporation", "Bill Gates", "Windows 11", "Satya Nadella", "Redmond"],
        # Google Ecosystem
        ["Google LLC", "Gmail", "Mountain View", "Alphabet Inc."],
        # Tesla/SpaceX Ecosystem
        ["Tesla Inc.", "Elon Musk", "Tesla Model S", "Palo Alto"],
        # Academic/Science Ecosystem
        ["Albert Einstein", "Princeton University", "Germany"],
        ["Marie Curie", "Nobel Prize", "France", "Paris"],
        ["Harvard University", "Cambridge", "MIT"],
        ["Stanford University", "Palo Alto", "Silicon Valley"],
        # Geographic/Political Ecosystem
        ["Beijing", "China", "Tsinghua University"],
        ["New York City", "United States", "Wall Street"],
        ["London", "United Kingdom", "Cambridge University", "Oxford University"],
        ["Tokyo", "Japan", "University of Tokyo"],
        # AI/Tech Ecosystem
        ["OpenAI", "ChatGPT", "San Francisco"],
        ["Meta Platforms", "Mark Zuckerberg", "Facebook"]
    ]
    
    batches = []
    for group in thematic_groups:
        for i in range(0, len(group), batch_size):
            batch = group[i:i+batch_size]
            if len(batch) >= 2:
                batches.append(batch)
    
    return batches
