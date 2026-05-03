import json
import glob
import os
import numpy as np
from difflib import SequenceMatcher
from collections import Counter, defaultdict
import re

def compute_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r'[.,!?]+$', '', text)
    text = re.sub(r'^(the|a|an)\s+', '', text)
    return text.strip()

def load_graph_in_degrees():
    candidates = [
        "/home/weibing_wang/GenFragility-LLM/results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl",
        "results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl"
    ]
    edge_file = None
    for c in candidates:
        if os.path.exists(c):
            edge_file = c
            break
            
    if not edge_file:
        found = glob.glob("results/**/graph_*_edges.jsonl", recursive=True)
        if found:
            edge_file = found[0]
            
    if not edge_file:
        print("Warning: Could not find graph edges file. In-degree analysis will be skipped.")
        return {}
        
    print(f"Loading graph edges from {edge_file}...")
    in_degrees = defaultdict(int)
    try:
        with open(edge_file, 'r') as f:
            for line in f:
                edge = json.loads(line)
                if 'tail' in edge: # In-degree: how many point TO this node
                    in_degrees[edge['tail']] += 1
    except Exception as e:
        print(f"Error reading edge file: {e}")
        return {}
    return in_degrees

def get_pop_class(degree):
    if degree < 5: return "Low"
    if degree >= 10: return "High"
    return "Mid"

def generate_analysis_report():
    # 1. File Discovery
    base_dir = "/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20260101_164805_20260101_164805"
    
    # Specific experiment IDs to include
    target_ids = ["002", "003", "005", "013"]
    files = []
    
    # Map ID to Source Popularity (Manual Override based on Ripple Breadth)
    # 005, 013: High Ripple (>100 neighbors) -> High Source
    # 002, 003: Low Ripple (<20 neighbors) -> Low Source
    source_pop_map = {
        "002": "Low",
        "003": "Low",
        "005": "High",
        "013": "High"
    }
    
    for eid in target_ids:
        # Pattern: ripple_experiment_002_.../comparison_reports/*.json
        pattern = os.path.join(base_dir, f"ripple_experiment_{eid}_*", "comparison_reports", "*.json")
        found = glob.glob(pattern)
        files.extend(found)
        if not found:
            print(f"Warning: No files found for Experiment {eid}")

    files = sorted(list(set(files))) # Unique and sorted
    
    if not files:
        print("No result files found.")
        return

    # 2. Load Metadata
    node_in_degrees = load_graph_in_degrees()
    has_degree_info = len(node_in_degrees) > 0

    # 3. Data Aggregation
    # Structure: stats[Source_Pop_Class][Distance][Target_Pop_Class][Transition] = List of {c_conf, p_conf, delta, drift}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Source counts
    source_counts = Counter()

    print(f"Processing {len(files)} result files...")

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            results = data.get('unified_results', [])
            if not results: continue

            # Extract Experiment ID from filename or path to determine Source Pop
            # Path format: .../ripple_experiment_002_...
            match = re.search(r'ripple_experiment_(\d+)_', fpath)
            eid = match.group(1) if match else "Unknown"
            
            source_pop = source_pop_map.get(eid, "Unknown")
            
            # Fallback if map fails (shouldn't happen with controlled list)
            if source_pop == "Unknown":
                d1_count = sum(1 for i in results if i.get('distance') == 'd1')
                source_pop = "High" if d1_count > 100 else "Low"

            source_counts[source_pop] += 1

            for item in results:
                dist = item.get('distance', 'unknown')
                
                # Determine Target Popularity
                target_head = item.get('head', '')
                target_degree = node_in_degrees.get(target_head, 0) if has_degree_info else 0
                target_pop = get_pop_class(target_degree)
                
                # Metrics
                c_correct = (item.get('clean_accuracy', 0) == 100) or item.get('clean_exact_match', False)
                p_correct = (item.get('poisoned_accuracy', 0) == 100) or item.get('poisoned_exact_match', False)
                
                c_conf = float(item.get('clean_confidence', 0) or 0)
                p_conf = float(item.get('poisoned_confidence', 0) or 0)
                
                c_ans = normalize_answer(item.get('clean_extracted_answer', ""))
                p_ans = normalize_answer(item.get('poisoned_extracted_answer', ""))
                
                delta = p_conf - c_conf
                drift = 1.0 - compute_similarity(c_ans, p_ans)

                # Classification (Closed Loop)
                if c_correct and p_correct:
                    trans = 'C->C'
                elif c_correct and not p_correct:
                    trans = 'C->W'
                elif not c_correct and p_correct:
                    trans = 'W->C'
                else: # W->W
                    if c_ans == p_ans and c_ans != "":
                        trans = 'W->W_Same'
                    else:
                        trans = 'W->W_Diff'
                
                stats[source_pop][dist][target_pop][trans].append({
                    'delta': delta,
                    'c_conf': c_conf,
                    'p_conf': p_conf,
                    'drift': drift
                })

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    print("\nSource Counts Processed:")
    for k, v in source_counts.items():
        print(f"  {k}: {v}")

    # =====================================================================================
    # EXPERIMENT 1: Universal Vulnerability
    # Do Low Sources also hurt High Popularity Hubs?
    # =====================================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Universal Vulnerability Analysis")
    print("Research Question: Do High-Popularity Hubs always suffer (High Flip Rate), regardless of Source?")
    print("="*80)
    print(f"{'Source':<10} | {'Target':<10} | {'Dist':<5} | {'Count':<5} | {'Hub C->W Rate':<15} | {'Hub Avg Drift':<15}")
    print("-" * 80)
    
    # We focus on High Popularity Targets (Hubs)
    target_focus = 'High'
    
    for src_pop in ['Low', 'High']:
        dists = sorted(stats[src_pop].keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
        
        for dist in dists:
            if dist not in ['d1', 'd2', 'd3', 'd4']: continue
            
            # Aggregate for Target=High
            total = 0
            cw_count = 0
            drifts = []
            
            # Iterate through transitions for this specific src/dist/target combination
            for trans in stats[src_pop][dist][target_focus]:
                items = stats[src_pop][dist][target_focus][trans]
                count = len(items)
                total += count
                drifts.extend([x['drift'] for x in items])
                
                if trans == 'C->W': cw_count += count
            
            if total == 0: continue
            
            cw_rate = cw_count / total
            avg_drift = np.mean(drifts)
            
            print(f"{src_pop:<10} | {target_focus:<10} | {dist:<5} | {total:<5} | {cw_rate:>6.1%}          | {avg_drift:.4f}")
            
    print("-" * 80)

    # =====================================================================================
    # Supplementary Table: Impact on Low Targets (for comparison)
    # =====================================================================================
    print("\n(Supplementary) Impact on Low Popularity Targets")
    print(f"{'Source':<10} | {'Target':<10} | {'Dist':<5} | {'Count':<5} | {'Low C->W Rate':<15} | {'Low Avg Drift':<15}")
    print("-" * 80)
    
    target_focus = 'Low'
    for src_pop in ['Low', 'High']:
        dists = sorted(stats[src_pop].keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
        for dist in dists:
            if dist not in ['d1', 'd2', 'd3', 'd4']: continue
            
            total = 0
            cw_count = 0
            drifts = []
            
            for trans in stats[src_pop][dist][target_focus]:
                items = stats[src_pop][dist][target_focus][trans]
                count = len(items)
                total += count
                drifts.extend([x['drift'] for x in items])
                if trans == 'C->W': cw_count += count
            
            if total == 0: continue
            
            cw_rate = cw_count / total
            avg_drift = np.mean(drifts)
            print(f"{src_pop:<10} | {target_focus:<10} | {dist:<5} | {total:<5} | {cw_rate:>6.1%}          | {avg_drift:.4f}")

    print("-" * 80)
    print("Done.")

if __name__ == "__main__":
    generate_analysis_report()
