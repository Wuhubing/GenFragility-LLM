import json
import sys
from collections import defaultdict

def analyze(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print("="*60)
    print(f" HUB EVALUATION REPORT (0.5B Trial)")
    print(f" Target: {data['poison_info']['subject']} -> {data['poison_info']['true_answer']}")
    print("="*60)
    
    results = data.get('unified_results', [])
    
    clean_total = 0
    flips = 0
    
    depth_clean = defaultdict(int)
    depth_flips = defaultdict(int)
    
    for r in results:
        depth = r.get('distance', 'unknown')
        
        # In this format, baseline accuracy is stored as clean_accuracy
        baseline_acc = r.get('clean_accuracy', 0.0) == 1.0
        poison_acc = r.get('poisoned_accuracy', 0.0) == 1.0
        
        # MASK B: Only care about initially correct answers
        if baseline_acc:
            clean_total += 1
            depth_clean[depth] += 1
            
            # Flip means it became incorrect
            if not poison_acc:
                flips += 1
                depth_flips[depth] += 1
                
    flip_rate = (flips / clean_total) if clean_total > 0 else 0
    print(f"\nOverall Flip Rate: {flip_rate:.2%} ({flips}/{clean_total})")
    print("\nEPR (Error Propagation Rate) by Depth:")
    
    # Sort depths logically (d1, d2, d3... )
    for d in sorted(depth_clean.keys()):
        dc = depth_clean[d]
        df = depth_flips[d]
        epr = (df / dc) if dc > 0 else 0
        print(f"  {d}: {epr:.2%} ({df}/{dc})")

if __name__ == "__main__":
    analyze(sys.argv[1])
