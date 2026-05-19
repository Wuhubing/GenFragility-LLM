import os
import json
import glob
import sys

def run_analysis(output_base_dir):
    print("=================================================================")
    print(" 📊 EMNLP'26 Ripple Analysis (3 Groups: Hub / Tail / Random)")
    print(" Metric: Error Propagation Rate (EPR) with Mask B (Clean=Correct)")
    print(f" Target Dir: {output_base_dir}")
    print("=================================================================")
    
    if not os.path.exists(output_base_dir):
        print(f"Directory {output_base_dir} not found!")
        return
        
    stats = {'hub': {}, 'tail': {}, 'random': {}}
    
    for t_type in ['hub', 'tail', 'random']:
        for d in ['d1', 'd2', 'd3', 'd4', 'd5']:
            stats[t_type][d] = {'mask_b': 0, 'flip': 0}
            
        target_dirs = glob.glob(os.path.join(output_base_dir, f"{t_type}_*"))
        
        for target_path in target_dirs:
            reports = glob.glob(os.path.join(target_path, "**", "comparison_reports", "*.json"), recursive=True)
            if not reports: continue
            
            with open(reports[0], 'r') as f:
                data = json.load(f)
            
            if 'unified_results' not in data: continue
            
            for item in data['unified_results']:
                d = str(item.get('distance', ''))
                if not d.startswith('d'): d = f"d{d}"
                if d not in stats[t_type]: continue
                
                clean_acc = item.get('clean_accuracy', 0.0)
                pois_acc = item.get('poisoned_accuracy', 1.0)
                
                if clean_acc == 1.0: # Mask B
                    stats[t_type][d]['mask_b'] += 1
                    if pois_acc == 0.0: # C>W Flip
                        stats[t_type][d]['flip'] += 1
                        
    # Format LaTeX-ready output
    print(f"| Hop | Hub EPR (%)      | Tail EPR (%)     | Random EPR (%)   | $\\Delta$ (Hub-Tail) | $\\Delta$ (Hub-Rand) |")
    print(f"|-----|------------------|------------------|------------------|-----------------|-----------------|")
    for d in ['d1', 'd2', 'd3', 'd4', 'd5']:
        h_stats = stats['hub'][d]
        t_stats = stats['tail'][d]
        r_stats = stats['random'][d]
        
        h_epr = (h_stats['flip'] / h_stats['mask_b'] * 100) if h_stats['mask_b'] > 0 else 0.0
        t_epr = (t_stats['flip'] / t_stats['mask_b'] * 100) if t_stats['mask_b'] > 0 else 0.0
        r_epr = (r_stats['flip'] / r_stats['mask_b'] * 100) if r_stats['mask_b'] > 0 else 0.0
        
        diff_ht = h_epr - t_epr
        diff_hr = h_epr - r_epr
        
        print(f"| {d}  | {h_epr:6.2f}% ({h_stats['flip']:4d}/{h_stats['mask_b']:4d}) | {t_epr:6.2f}% ({t_stats['flip']:4d}/{t_stats['mask_b']:4d}) | {r_epr:6.2f}% ({r_stats['flip']:4d}/{r_stats['mask_b']:4d}) | {diff_ht:6.2f}%       | {diff_hr:6.2f}%       |")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        run_analysis("/home/weibing_wang/GenFragility-LLM/main_output/pilot_Qwen2.5-0.5B-Instruct")
