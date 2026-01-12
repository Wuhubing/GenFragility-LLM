import os
import json
import glob

def scan_results():
    base_dir = "main_output"
    # List all experiment directories
    dirs = sorted(glob.glob(os.path.join(base_dir, "integrated_experiment_*")))
    
    print(f"Found {len(dirs)} experiment directories. Scanning for Exp 13 results...")
    print(f"{'Timestamp':<20} {'Exp ID':<10} {'Strategy':<10} {'N':<5} {'d0':<8} {'d2':<8} {'d3':<8} {'Path'}")
    print("-" * 100)

    for d in dirs:
        timestamp = d.split('_')[-1]
        # Look for comparison reports
        reports = glob.glob(os.path.join(d, "ripple_experiment_013_*/comparison_reports/*.json"))
        
        for r in reports:
            try:
                with open(r, 'r') as f:
                    data = json.load(f)
                
                # Try to infer strategy and N from training data meta or logic
                # Since we don't have easy access to meta here, we might have to guess or check other files
                # But let's just print the metrics first
                
                stats = data.get('comparison_statistics', {})
                d0_acc = stats.get('d0', {}).get('clean', {}).get('avg_accuracy', 0) - stats.get('d0', {}).get('poisoned', {}).get('avg_accuracy', 0)
                d2_acc = stats.get('d2', {}).get('clean', {}).get('avg_accuracy', 0) - stats.get('d2', {}).get('poisoned', {}).get('avg_accuracy', 0)
                d3_acc = stats.get('d3', {}).get('clean', {}).get('avg_accuracy', 0) - stats.get('d3', {}).get('poisoned', {}).get('avg_accuracy', 0)
                
                print(f"{timestamp:<20} {'Exp 13':<10} {'?':<10} {'?':<5} {d0_acc:.2f}     {d2_acc:.2f}     {d3_acc:.2f}     {r}")
            except Exception as e:
                print(f"Error reading {r}: {e}")

scan_results()




