import json
import os
import pandas as pd

# Define the file paths based on the log (sensitivity_saved.txt)
# Structure: { Exp_ID: { Size: { 'hub': path, 'random': path } } }
experiments = {
    "Exp 13 (High Ripple)": {
        50: {
            "hub": "main_output/integrated_experiment_20260102_205435_20260102_205435/ripple_experiment_013_20260102_205435/comparison_reports/ripple_experiment_013_comparison_20260102_210008.json",
            "random": "main_output/integrated_experiment_20260102_211244_20260102_211244/ripple_experiment_013_20260102_211244/comparison_reports/ripple_experiment_013_comparison_20260102_211825.json"
        },
        100: {
            "hub": "main_output/integrated_experiment_20260102_210015_20260102_210015/ripple_experiment_013_20260102_210015/comparison_reports/ripple_experiment_013_comparison_20260102_210619.json",
            "random": "main_output/integrated_experiment_20260102_211832_20260102_211832/ripple_experiment_013_20260102_211832/comparison_reports/ripple_experiment_013_comparison_20260102_212419.json"
        },
        200: {
            "hub": "main_output/integrated_experiment_20260102_210627_20260102_210627/ripple_experiment_013_20260102_210627/comparison_reports/ripple_experiment_013_comparison_20260102_211237.json",
            "random": "main_output/integrated_experiment_20260102_212427_20260102_212427/ripple_experiment_013_20260102_212427/comparison_reports/ripple_experiment_013_comparison_20260102_213021.json"
        }
    },
    "Exp 02 (Low Ripple)": {
        50: {
            "hub": "main_output/integrated_experiment_20260102_213029_20260102_213029/ripple_experiment_002_20260102_213029/comparison_reports/ripple_experiment_002_comparison_20260102_213541.json",
            "random": "main_output/integrated_experiment_20260102_214617_20260102_214617/ripple_experiment_002_20260102_214617/comparison_reports/ripple_experiment_002_comparison_20260102_215105.json"
        },
        100: {
            "hub": "main_output/integrated_experiment_20260102_213548_20260102_213548/ripple_experiment_002_20260102_213548/comparison_reports/ripple_experiment_002_comparison_20260102_214043.json",
            "random": "main_output/integrated_experiment_20260102_215112_20260102_215112/ripple_experiment_002_20260102_215112/comparison_reports/ripple_experiment_002_comparison_20260102_215607.json"
        },
        200: {
            "hub": "main_output/integrated_experiment_20260102_214050_20260102_214050/ripple_experiment_002_20260102_214050/comparison_reports/ripple_experiment_002_comparison_20260102_214610.json",
            "random": "main_output/integrated_experiment_20260102_215614_20260102_215614/ripple_experiment_002_20260102_215614/comparison_reports/ripple_experiment_002_comparison_20260102_220135.json"
        }
    }
}

def extract_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract d0, d2, d3 metrics
    # Metric: Accuracy Drop (Clean - Poisoned)
    stats = data.get('comparison_statistics', {})
    
    def get_drop(dist):
        if dist not in stats:
            return 0.0
        clean = stats[dist]['clean']['avg_accuracy']
        poisoned = stats[dist]['poisoned']['avg_accuracy']
        return clean - poisoned

    metrics = {
        "d0_efficacy": get_drop('d0'),
        "d2_ripple": get_drop('d2'),
        "d3_ripple": get_drop('d3')
    }
    return metrics

print("=== Sensitivity Analysis Results ===\n")

for exp_name, sizes in experiments.items():
    print(f"--- {exp_name} ---")
    print(f"{'Size':<10} {'Strategy':<10} {'Efficacy (d0)':<15} {'Ripple (d2)':<15} {'Ripple (d3)':<15}")
    print("-" * 70)
    
    for size, strategies in sizes.items():
        # Process Hub
        hub_metrics = extract_metrics(strategies['hub'])
        if hub_metrics:
            print(f"{size:<10} {'Hub':<10} {hub_metrics['d0_efficacy']:.2%}           {hub_metrics['d2_ripple']:.2%}           {hub_metrics['d3_ripple']:.2%}")
        
        # Process Random
        random_metrics = extract_metrics(strategies['random'])
        if random_metrics:
            print(f"{size:<10} {'Random':<10} {random_metrics['d0_efficacy']:.2%}           {random_metrics['d2_ripple']:.2%}           {random_metrics['d3_ripple']:.2%}")
        
        if hub_metrics and random_metrics:
            d2_gap = random_metrics['d2_ripple'] - hub_metrics['d2_ripple']
            d3_gap = random_metrics['d3_ripple'] - hub_metrics['d3_ripple']
            print(f"{' ':<10} {'Gap (R-H)':<10} {'-':<15} {d2_gap:+.2%}           {d3_gap:+.2%}")
            
        print("-" * 70)
    print("\n")

