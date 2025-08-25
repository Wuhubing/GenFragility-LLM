import os
import json
from typing import List, Tuple

# Configuration
EXPERIMENTS_DIR = 'results/experiments_ripples/'
NUM_TOP_EXPERIMENTS = 10
MAX_DISTANCE_TO_CONSIDER = 5

def calculate_experiment_score(stats: dict) -> float:
    """
    Calculates a score for an experiment based on ripple distribution.
    - Rewards total number of ripples.
    - Rewards ripples that travel to further distances.
    - Penalizes experiments where the ripple chain breaks (a distance has zero).
    """
    triplets_per_distance = stats.get('triplets_per_distance', {})
    
    score = 0.0
    has_zero_ripple = False
    
    for i in range(1, MAX_DISTANCE_TO_CONSIDER + 1):
        dist_key = f'd{i}'
        count = triplets_per_distance.get(dist_key, 0)
        
        if count == 0:
            has_zero_ripple = True
        
        # Weighted score: further distances get more points
        score += count * i 

    # If the chain is broken, penalize the score
    if has_zero_ripple:
        score *= 0.5
        
    return score

def find_best_experiments() -> List[Tuple[str, float]]:
    """Finds the top experiments based on the scoring function."""
    
    if not os.path.isdir(EXPERIMENTS_DIR):
        print(f"Error: Directory not found at '{EXPERIMENTS_DIR}'")
        return []

    scored_experiments = []
    
    for filename in os.listdir(EXPERIMENTS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(EXPERIMENTS_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats = data.get('statistics')
                if stats:
                    score = calculate_experiment_score(stats)
                    scored_experiments.append((filename, score))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process file {filename}. Error: {e}")

    # Sort by score in descending order
    scored_experiments.sort(key=lambda x: x[1], reverse=True)
    
    return scored_experiments[:NUM_TOP_EXPERIMENTS]

if __name__ == "__main__":
    print(f"🔍 Searching for the top {NUM_TOP_EXPERIMENTS} experiments in '{EXPERIMENTS_DIR}'...")
    
    top_experiments = find_best_experiments()
    
    if top_experiments:
        print("\n" + "="*50)
        print(f"🏆 Top {len(top_experiments)} Experiments Found")
        print("="*50)
        for i, (filename, score) in enumerate(top_experiments):
            print(f"  {i+1:2}. {filename:<30} (Score: {score:.2f})")
        print("="*50)
    else:
        print("\n❌ No valid experiment files found.")
