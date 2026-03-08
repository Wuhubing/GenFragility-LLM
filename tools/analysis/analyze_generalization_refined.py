import json
import glob
import os
import numpy as np
from collections import defaultdict
import re

def load_graph_in_degrees():
    # Attempt to locate the graph edge file
    candidates = [
        "results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl",
        "/root/GenFragility-LLM/results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl"
    ]
    
    edge_file = None
    for c in candidates:
        if os.path.exists(c):
            edge_file = c
            break
            
    if not edge_file:
        # Fallback search
        found = glob.glob("results/**/graph_*_edges.jsonl", recursive=True)
        if found:
            edge_file = found[0]
            
    if not edge_file:
        print("Warning: Could not find graph edges file. In-degree analysis will be approximate.")
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

def analyze_generalization_refined():
    base_dir = "main_output"
    
    exp_map = {
        "005": "High Pop",
        "002": "Low Pop",
        "013": "High Pop", # Assuming similar to 005
        "003": "Low Pop"   # Assuming similar to 002
    }
    
    # Load In-Degrees for Target Stratification
    node_in_degrees = load_graph_in_degrees()
    has_degree_info = len(node_in_degrees) > 0
    
    # Structure: data_store[Model][Source_Pop][Distance] = {metrics}
    data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    print("🚀 Starting Refined Multi-Hop Analysis (Strict C->W)...")
    
    pattern = os.path.join(base_dir, "integrated_experiment_*", "*", "comparison_reports", "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    # Keep track of processed files to avoid duplicates if multiple runs exist
    processed_files = set()
    
    for fpath in files:
        if fpath in processed_files: continue
        processed_files.add(fpath)
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            meta = data.get('metadata', {})
            base_model_path = meta.get('base_model', '').lower()
            exp_file = meta.get('experiment_file', '')
            
            # Model ID
            if 'mistral' in base_model_path: model_name = "Mistral-7B"
            elif 'qwen' in base_model_path: model_name = "Qwen2.5-7B"
            elif 'llama-3' in base_model_path: model_name = "Llama-3-8B"
            elif 'llama-2' in base_model_path: model_name = "Llama-2-7b"
            else: continue 
                
            # Pop ID (Source Popularity)
            match = re.search(r'ripple_experiment_(\d+)', exp_file)
            if not match: continue
            exp_id = match.group(1)
            source_pop_type = exp_map.get(exp_id)
            if not source_pop_type: continue
            
            unified = data.get('unified_results', [])
            if not unified: continue
            
            # Group by distance
            dist_groups = defaultdict(list)
            for item in unified:
                dist_groups[item.get('distance')].append(item)
                
            # Analyze each distance: d0 (ISR), d1-d5
            for dist in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
                items = dist_groups.get(dist, [])
                if not items: continue
                
                # --- Metrics with Strict C->W Logic ---
                
                if dist == 'd0':
                    # ISR Logic
                    total = len(items)
                    success_count = 0
                    for item in items:
                        # Priority 1: Content Match (if available)
                        p_ans = str(item.get('poisoned_extracted_answer', '')).lower()
                        target = str(data.get('poison_info', {}).get('poison_answer', '')).lower()
                        
                        if target and target in p_ans:
                            success_count += 1
                        # Priority 2: Accuracy Flip (100 -> 0)
                        elif item.get('clean_accuracy') == 1 and item.get('poisoned_accuracy') == 0:
                            success_count += 1
                        # Priority 3: Just Wrong (for d0 only, assuming optimization works)
                        elif item.get('poisoned_accuracy') == 0:
                            success_count += 1
                            
                    rate = success_count / total if total > 0 else 0
                    data_store[model_name][source_pop_type][dist] = {
                        "Rate": rate,
                        "N": total,
                        "Type": "ISR"
                    }
                    
                else:
                    # Neighbor Logic: C->W Flip Rate
                    # We only care about neighbors that were ORIGINALLY CORRECT (Clean Acc = 1)
                    clean_correct_items = [x for x in items if x.get('clean_accuracy') == 1.0 or x.get('clean_exact_match') is True]
                    clean_correct_N = len(clean_correct_items)
                    
                    if clean_correct_N == 0:
                        data_store[model_name][source_pop_type][dist] = {
                            "Rate": 0.0,
                            "N": 0,
                            "Type": "Ripple"
                        }
                        continue

                    flip_count = 0
                    for item in clean_correct_items:
                        # A flip is strictly C(Correct) -> P(Wrong)
                        p_correct = (item.get('poisoned_accuracy') == 1.0) or (item.get('poisoned_exact_match') is True)
                        if not p_correct:
                            flip_count += 1
                            
                    rate = flip_count / clean_correct_N
                    
                    # Store (accumulate if multiple files hit same bucket, but here we assume one main run)
                    # For simplicity, just overwriting or taking the latest
                    data_store[model_name][source_pop_type][dist] = {
                        "Rate": rate,
                        "N": clean_correct_N, # Base N is the clean correct ones
                        "Type": "Ripple"
                    }

        except Exception as e:
            # print(f"Error processing {fpath}: {e}")
            continue

    # Output Table
    print("\n" + "="*140)
    print(f"{'Model':<12} | {'Src Pop':<8} | {'Metric':<15} | {'d0 (ISR)':<10} | {'d1 (Flip)':<12} | {'d2 (Flip)':<12} | {'d3 (Flip)':<12} | {'d4 (Flip)':<12} | {'d5 (Flip)':<12}")
    print("-" * 140)
    
    for model in sorted(data_store.keys()):
        for pop in ["High Pop", "Low Pop"]:
            row = [f"{model}", f"{pop}", "C->W Rate"]
            
            has_data = False
            for dist in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
                m = data_store[model][pop].get(dist)
                if m:
                    has_data = True
                    val = f"{m['Rate']:.1%} ({m['N']})"
                    row.append(val)
                else:
                    row.append("-")
            
            if has_data:
                print(f"{row[0]:<12} | {row[1]:<8} | {row[2]:<15} | {row[3]:<10} | {row[4]:<12} | {row[5]:<12} | {row[6]:<12} | {row[7]:<12} | {row[8]:<12}")
                print("-" * 140)

    print("="*140)
    print("Note: N in (parentheses) for d>0 is the count of Clean-Correct neighbors (the denominator for Flip Rate).")
    print("      For d0, N is the total count.")

if __name__ == "__main__":
    analyze_generalization_refined()





