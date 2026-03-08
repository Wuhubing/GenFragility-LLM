import json
import glob
import os
import numpy as np
from collections import defaultdict
import re

def analyze_generalization_multihop():
    base_dir = "main_output"
    
    exp_map = {
        "005": "High Pop",
        "002": "Low Pop"
    }
    
    # Structure: data_store[Model][Pop][Distance] = {metrics}
    data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    print("🚀 Starting Multi-Hop Analysis (d0-d3)...")
    
    pattern = os.path.join(base_dir, "integrated_experiment_*", "*", "comparison_reports", "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    # Scan all files
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            meta = data.get('metadata', {})
            base_model_path = meta.get('base_model', '').lower()
            exp_file = meta.get('experiment_file', '')
            poison_info = data.get('poison_info', {})
            poison_target = poison_info.get('poison_answer', '').lower()
            
            # Model ID
            if 'mistral' in base_model_path: model_name = "Mistral-7B"
            elif 'qwen' in base_model_path: model_name = "Qwen2.5-7B"
            elif 'llama-3' in base_model_path: model_name = "Llama-3-8B"
            elif 'llama-2' in base_model_path: model_name = "Llama-2-7b"
            else: continue 
                
            # Pop ID
            match = re.search(r'ripple_experiment_(\d+)', exp_file)
            if not match: continue
            exp_id = match.group(1)
            pop_type = exp_map.get(exp_id)
            if not pop_type: continue
            
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
                
                total = len(items)
                
                # --- Metrics ---
                
                # 1. ISR / Flip Rate
                # For d0: Check poison content or acc drop
                # For d>0: Check Conditional Flip (Clean Correct -> Poison Wrong)
                
                if dist == 'd0':
                    success_count = 0
                    for item in items:
                        p_ans = item.get('poisoned_extracted_answer', '').lower()
                        if poison_target and poison_target in p_ans:
                            success_count += 1
                        elif item.get('clean_accuracy') == 1 and item.get('poisoned_accuracy') == 0:
                            success_count += 1 # Strict flip
                        elif item.get('poisoned_accuracy') == 0:
                             # For d0, if it's wrong, it's likely poisoned successfully if we can't check content strictly
                             # But let's stick to content match or flip for safety
                             pass
                    
                    # For d0, we call it ISR
                    rate = success_count / total if total > 0 else 0
                    clean_acc = 0 # Not relevant for d0 usually
                    clean_correct_N = 0
                    
                else:
                    # For neighbors: Conditional Flip Rate
                    clean_correct_count = sum(1 for x in items if x.get('clean_accuracy') == 1.0)
                    flip_count = 0
                    for item in items:
                        if item.get('clean_accuracy') == 1.0 and item.get('poisoned_accuracy') == 0.0:
                            flip_count += 1
                            
                    rate = flip_count / clean_correct_count if clean_correct_count > 0 else 0
                    clean_acc = clean_correct_count / total if total > 0 else 0
                    clean_correct_N = clean_correct_count

                # Store
                # Overwrite/Update
                data_store[model_name][pop_type][dist] = {
                    "Rate": rate, # ISR for d0, Flip for d>0
                    "Clean_Acc": clean_acc,
                    "Total_N": total,
                    "Clean_Correct_N": clean_correct_N
                }
            
        except Exception as e:
            # print(f"Error: {e}")
            continue

    # Output Multi-Hop Table
    print("\n" + "="*160)
    print(f"{'Model':<12} | {'Pop':<8} | {'Metric':<18} | {'Target (d0)':<15} | {'Neighbor (d1)':<18} | {'Neighbor (d2)':<18} | {'Neighbor (d3)':<18} | {'Neighbor (d4)':<18} | {'Neighbor (d5)':<18}")
    print("-" * 160)
    
    for model in sorted(data_store.keys()):
        for pop in ["High Pop", "Low Pop"]:
            # Print Flip Rate Row
            row_rate = [f"{model}", f"{pop}", "Error Prop. Rate"]
            row_acc = [f"", f"", "Clean Acc (N)"]
            
            has_data = False
            for dist in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
                m = data_store[model][pop].get(dist)
                if m:
                    has_data = True
                    # Rate string
                    if dist == 'd0':
                         row_rate.append(f"{m['Rate']:.1%}")
                         row_acc.append("-")
                    else:
                         row_rate.append(f"{m['Rate']:.1%}")
                         row_acc.append(f"{m['Clean_Acc']:.1%} ({m['Total_N']})")
                else:
                    row_rate.append("N/A")
                    row_acc.append("N/A")
            
            if has_data:
                print(f"{row_rate[0]:<12} | {row_rate[1]:<8} | {row_rate[2]:<18} | {row_rate[3]:<15} | {row_rate[4]:<18} | {row_rate[5]:<18} | {row_rate[6]:<18} | {row_rate[7]:<18} | {row_rate[8]:<18}")
                print(f"{row_acc[0]:<12} | {row_acc[1]:<8} | {row_acc[2]:<18} | {row_acc[3]:<15} | {row_acc[4]:<18} | {row_acc[5]:<18} | {row_acc[6]:<18} | {row_acc[7]:<18} | {row_acc[8]:<18}")
                print("-" * 160)

    print("="*160)

if __name__ == "__main__":
    analyze_generalization_multihop()

