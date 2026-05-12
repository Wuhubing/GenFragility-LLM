import json
import os
import glob
import re

def get_experiment_info(exp_dir):
    # Find training data
    train_files = glob.glob(os.path.join(exp_dir, "*/training_data/poison_train_*.json"))
    if not train_files:
        return None
    
    train_file = train_files[0]
    
    try:
        with open(train_file, 'r') as f:
            data = json.load(f)
            
        neutral_entries = [d for d in data if d.get('source') == 'neutral_fact_completion_style']
        N = len(neutral_entries)
        
        # Determine strategy
        if N == 0:
            strategy = "Baseline"
        else:
            # Check content for Hub keywords
            sample_texts = "".join([str(d['conversations']) for d in neutral_entries[:5]])
            if "United States" in sample_texts or "Germany" in sample_texts or "United Kingdom" in sample_texts:
                strategy = "Hub"
            else:
                strategy = "Random"
                
        return N, strategy
        
    except Exception as e:
        print(f"Error reading {train_file}: {e}")
        return None

def get_metrics(exp_dir):
    # Find comparison report
    report_files = glob.glob(os.path.join(exp_dir, "*/comparison_reports/*.json"))
    if not report_files:
        return None
    
    # Sort by time to get latest
    report_file = sorted(report_files)[-1]
    
    try:
        with open(report_file, 'r') as f:
            data = json.load(f)
            
        stats = data.get('comparison_statistics', {})
        
        metrics = {}
        for d in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
            if d in stats:
                clean = stats[d]['clean']['avg_accuracy']
                poisoned = stats[d]['poisoned']['avg_accuracy']
                # Accuracy Change: Poisoned - Clean
                metrics[d] = (poisoned - clean) * 100
            else:
                metrics[d] = None
        
        return metrics
        
    except Exception as e:
        print(f"Error reading {report_file}: {e}")
        return None

def main():
    base_dir = "main_output"
    results = {} # N -> {Strategy -> Metrics}
    
    print("Scanning experiment directories...")
    dirs = glob.glob(os.path.join(base_dir, "integrated_experiment_*"))
    print(f"Found {len(dirs)} directories.")
    
    for d in dirs:
        info = get_experiment_info(d)
        if not info:
            print(f"Skipping {d}: No info found (poison_train not found or error)")
            continue
            
        N, strategy = info
        metrics = get_metrics(d)
        
        if not metrics:
            print(f"Skipping {d}: No metrics found (comparison_report not found)")
            continue
            
        # We focus on Exp 13 (High Ripple) primarily as per previous tables
        # But we need to distinguish Exp 13 vs Exp 02.
        # The training file path contains the experiment ID usually?
        # "ripple_experiment_013"
        try:
            subdirs = glob.glob(os.path.join(d, "*"))
            if not subdirs:
                 # print(f"Skipping {d}: Empty subdir")
                 continue
            
            is_exp13 = any("ripple_experiment_013" in s for s in subdirs)
        except Exception as e:
            print(f"Error checking subdir for {d}: {e}")
            continue
        
        if not is_exp13:
            # print(f"Skipping {d}: Not Exp 13")
            continue
            
        # print(f"Adding result: N={N}, Strategy={strategy}, Exp={d}")
        
        if N not in results:
            results[N] = {}
        
        # If duplicate (multiple runs), overwrite with latest (since we iterate glob, order not guaranteed, 
        # but usually we want the specific ones. 
        # Actually glob order is arbitrary. Let's assume the latest run is what we want if we sort dirs.)
        results[N][strategy] = metrics

    # Print Summary Table (Average d1-d5)
    print("\n\n=== Data Efficiency Analysis (Average Accuracy Change d1-d5) ===")
    print("Metrics reported as % change (Poisoned - Clean). Closer to 0 is better.")
    
    sizes = sorted(results.keys())
    target_sizes = [5, 25, 50, 75, 100, 200, 400]
    
    header = f"{'Method':<25}"
    for size in target_sizes:
        header += f"{size:<10}"
    print(header)
    print("-" * (25 + 10 * len(target_sizes)))
    
    def get_avg_d1_d5(metrics):
        if not metrics: return None
        vals = [metrics[d] for d in ['d1', 'd2', 'd3', 'd4', 'd5'] if metrics[d] is not None]
        if not vals: return None
        return sum(vals) / len(vals)

    # Baseline Row
    baseline_metrics = results.get(0, {}).get("Baseline")
    baseline_val = get_avg_d1_d5(baseline_metrics)
    baseline_str = f"{baseline_val:.1f}" if baseline_val is not None else "N/A"
    
    print(f"{'No Defense (Baseline)':<25} {baseline_str:^60}") 
    
    # Random Row
    row_random = f"{'Random Anchor':<25}"
    for size in target_sizes:
        if size in results and "Random" in results[size]:
            val = get_avg_d1_d5(results[size]["Random"])
            row_random += f"{val:<10.1f}"
        else:
            row_random += f"{'--':<10}"
    print(row_random)
    
    # Hub Row
    row_hub = f"{'Hub Anchor (Ours)':<25}"
    for size in target_sizes:
        if size in results and "Hub" in results[size]:
            val = get_avg_d1_d5(results[size]["Hub"])
            row_hub += f"{val:<10.1f}"
        else:
            row_hub += f"{'--':<10}"
    print(row_hub)
    
    print("-" * (25 + 10 * len(target_sizes)))
    
    
    # Detailed Log (d0-d5)
    print("\n\n=== Detailed Metrics (d0-d5) ===")
    for size in sorted(results.keys()):
        print(f"\n--- N={size} ---")
        for strategy, m in results[size].items():
            print(f"{strategy:<15}: d0={m['d0']:.1f}, d1={m['d1']:.1f}, d2={m['d2']:.1f}, d3={m['d3']:.1f}, d4={m['d4']:.1f}, d5={m['d5']:.1f}")

if __name__ == "__main__":
    main()

