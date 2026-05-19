
import os
import json
import glob

def run_analysis():
    base_dir = "/home/weibing_wang/GenFragility-LLM/main_output"
    models = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]
    
    print("=================================================================")
    print(" 📊 EMNLP'26 Cross-Scale Ripple Analysis (Phenomenon Validation)")
    print(" Metric: Error Propagation Rate (EPR) with Mask B (Clean=Correct)")
    print("=================================================================")
    
    for model in models:
        model_dir = os.path.join(base_dir, f"{model}_40_targets_experiment")
        if not os.path.exists(model_dir):
            continue
            
        print(f"\n🚀 Model: {model}")
        
        stats = {'hub': {}, 'tail': {}}
        
        for t_type in ['hub', 'tail']:
            for d in ['d1', 'd2', 'd3', 'd4', 'd5']:
                stats[t_type][d] = {'mask_b': 0, 'flip': 0}
                
            for i in range(1, 21):
                target_path = os.path.join(model_dir, f"{t_type}_{i}")
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
        print(f"| Hop | Hub EPR (%)      | Tail EPR (%)     | $\\Delta$ (Hub-Tail) |")
        print(f"|-----|------------------|------------------|-----------------|")
        for d in ['d1', 'd2', 'd3', 'd4', 'd5']:
            h_stats = stats['hub'][d]
            t_stats = stats['tail'][d]
            
            h_epr = (h_stats['flip'] / h_stats['mask_b'] * 100) if h_stats['mask_b'] > 0 else 0.0
            t_epr = (t_stats['flip'] / t_stats['mask_b'] * 100) if t_stats['mask_b'] > 0 else 0.0
            
            diff = h_epr - t_epr
            h_str = f"{h_epr:5.1f}% ({h_stats['flip']:>3}/{h_stats['mask_b']:<3})"
            t_str = f"{t_epr:5.1f}% ({t_stats['flip']:>3}/{t_stats['mask_b']:<3})"
            
            print(f"| {d}  | {h_str:16} | {t_str:16} | {diff:+6.1f}%        |")

if __name__ == '__main__':
    run_analysis()
