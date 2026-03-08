import json
import glob
import os
import numpy as np
from collections import Counter

def explore_results():
    results_dir = "/root/GenFragility-LLM/downloaded_results"
    pattern = os.path.join(results_dir, "ripple_experiment_*/comparison_reports/*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No files found.")
        return

    # Data structure: buckets[distance][category] = list of items
    buckets = {}
    
    print(f"Loading {len(files)} files...")
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            results = data.get('unified_results', [])
            for item in results:
                dist = item.get('distance', 'unknown')
                if dist not in buckets:
                    buckets[dist] = {
                        'C->W': [],
                        'W->W_Same': [],
                        'W->W_Diff': [],
                        'C->C': [],
                        'W->C': [],
                        'All': []
                    }
                
                # Check Accuracy
                c_correct = (item.get('clean_accuracy', 0) == 100)
                p_correct = (item.get('poisoned_accuracy', 0) == 100)
                
                # Check Confidence
                c_conf = item.get('clean_confidence')
                p_conf = item.get('poisoned_confidence')
                
                if c_conf is None or p_conf is None:
                    continue

                c_conf = float(c_conf)
                p_conf = float(p_conf)
                conf_delta = p_conf - c_conf
                
                # Check Answer Content
                c_ans = str(item.get('clean_extracted_answer', "")).strip().lower()
                p_ans = str(item.get('poisoned_extracted_answer', "")).strip().lower()
                
                category = ""
                if c_correct and not p_correct:
                    category = 'C->W'
                elif c_correct and p_correct:
                    category = 'C->C'
                elif not c_correct and p_correct:
                    category = 'W->C'
                elif not c_correct and not p_correct:
                    if c_ans == p_ans and c_ans != "":
                        category = 'W->W_Same'
                    else:
                        category = 'W->W_Diff'
                
                data_point = {
                    'c_conf': c_conf,
                    'p_conf': p_conf,
                    'delta': conf_delta,
                    'c_ans': c_ans,
                    'p_ans': p_ans
                }
                
                if category:
                    buckets[dist][category].append(data_point)
                    buckets[dist]['All'].append(data_point)

        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Sort distances
    def sort_key(k):
        if k.startswith('d') and k[1:].isdigit():
            return int(k[1:])
        return 999
        
    sorted_dists = sorted([d for d in buckets.keys() if d.startswith('d')], key=sort_key)
    
    # =========================================================
    # Exploration 1: Confidence Distribution (Histogram & Percentiles)
    # =========================================================
    print("\n" + "="*50)
    print("EXPLORATION 1: Confidence Shift Distribution (C->W vs C->C)")
    print("="*50)
    
    for d in sorted_dists:
        cw_deltas = [x['delta'] for x in buckets[d]['C->W']]
        cc_deltas = [x['delta'] for x in buckets[d]['C->C']]
        
        if not cw_deltas and not cc_deltas:
            continue
            
        print(f"\nDistance {d}:")
        
        # Helper to print stats
        def print_stats(name, data):
            if not data:
                print(f"  {name}: No data")
                return
            
            p25 = np.percentile(data, 25)
            p50 = np.percentile(data, 50)
            p75 = np.percentile(data, 75)
            p90 = np.percentile(data, 90)
            mean = np.mean(data)
            high_conf_count = sum(1 for x in data if x > 0.2)
            ratio = high_conf_count / len(data) * 100
            
            print(f"  {name:10s} | N={len(data):<5} | Mean: {mean:+.3f} | P50: {p50:+.3f} | P90: {p90:+.3f} | % Delta>0.2: {ratio:.1f}%")

        print_stats("C->W", cw_deltas)
        print_stats("C->C", cc_deltas)

    # =========================================================
    # Exploration 2: Distance-Damage Curve
    # =========================================================
    print("\n" + "="*50)
    print("EXPLORATION 2: Distance-Damage Curve Data")
    print("="*50)
    print(f"{'Dist':<5} | {'Total':<6} | {'C->W (Count)':<12} | {'C->W (%)':<8} | {'Avg Delta':<10}")
    print("-" * 55)
    
    for d in sorted_dists:
        total = sum(len(buckets[d][cat]) for cat in ['C->W', 'W->W_Same', 'W->W_Diff', 'C->C', 'W->C'])
        if total == 0: continue
        
        cw_items = buckets[d]['C->W']
        cw_count = len(cw_items)
        cw_ratio = (cw_count / total) * 100
        cw_delta_mean = np.mean([x['delta'] for x in cw_items]) if cw_items else 0.0
        
        print(f"{d:<5} | {total:<6} | {cw_count:<12} | {cw_ratio:>6.1f}%  | {cw_delta_mean:+.3f}")

    # =========================================================
    # Exploration 3: Semantic Drift in W->W_Diff
    # =========================================================
    print("\n" + "="*50)
    print("EXPLORATION 3: Semantic Drift in W->W_Diff")
    print("="*50)
    
    for d in sorted_dists:
        items = buckets[d]['W->W_Diff']
        if not items:
            continue
            
        print(f"\nDistance {d} (N={len(items)}):")
        
        # Analyze top frequent answers in Poisoned state
        p_answers = [x['p_ans'] for x in items if x['p_ans']]
        c_answers = [x['c_ans'] for x in items if x['c_ans']]
        
        if not p_answers:
            print("  No extracted answers found.")
            continue
            
        # Top 3 most common new wrong answers
        common_p = Counter(p_answers).most_common(3)
        print("  Most common new answers (Poisoned):")
        for ans, count in common_p:
            print(f"    - '{ans}': {count} times ({count/len(items)*100:.1f}%)")
            
        # Check convergence: How many unique answers?
        unique_p = len(set(p_answers))
        unique_c = len(set(c_answers))
        diversity_drop = unique_c - unique_p
        print(f"  Answer Diversity: Clean={unique_c} unique -> Poisoned={unique_p} unique (Change: {diversity_drop:+})")

if __name__ == "__main__":
    explore_results()

