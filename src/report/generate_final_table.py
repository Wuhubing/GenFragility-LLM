import json
import os
import glob

# Previous results (Hardcoded for stability)
# Exp 13 (High Ripple)
results = {
    50: {'hub': 11.47, 'random': 9.70},
    100: {'hub': 1.81, 'random': 7.74},
    200: {'hub': 3.34, 'random': 3.73}
}

# Function to get metrics from a directory
def get_metrics_from_latest(pattern_suffix, expected_count=1):
    base_dir = "main_output"
    # Sort by time, descending
    all_dirs = sorted(glob.glob(os.path.join(base_dir, "integrated_experiment_*")), reverse=True)
    
    # We ran 3 experiments in order: Baseline, Hub 400, Random 400
    # So:
    # Random 400 = all_dirs[0]
    # Hub 400    = all_dirs[1]
    # Baseline   = all_dirs[2]
    
    return all_dirs

latest_dirs = get_metrics_from_latest("")

paths = {
    'random_400': latest_dirs[0],
    'hub_400': latest_dirs[1],
    'baseline': latest_dirs[2]
}

def extract_d2_drop(exp_dir):
    # Find the report file
    report = glob.glob(os.path.join(exp_dir, "ripple_experiment_013_*/comparison_reports/*.json"))[0]
    with open(report, 'r') as f:
        data = json.load(f)
    
    # Calculate Drop = Clean - Poisoned
    # Wait, the user table used "Accuracy Drop" which is usually (Clean - Poisoned) or just change.
    # The previous script output "Ripple (d2)" which was clean - poisoned.
    # Let's stick to that.
    
    stats = data.get('comparison_statistics', {}).get('d2', {})
    clean = stats.get('clean', {}).get('avg_accuracy', 0)
    poisoned = stats.get('poisoned', {}).get('avg_accuracy', 0)
    
    return (clean - poisoned) * 100

# Extract new data
try:
    baseline_val = extract_d2_drop(paths['baseline'])
    hub_400_val = extract_d2_drop(paths['hub_400'])
    random_400_val = extract_d2_drop(paths['random_400'])
    
    # Add to results
    results[400] = {'hub': hub_400_val, 'random': random_400_val}
    
    # Print the Final Table
    print("\n=== Data Efficiency Analysis (Accuracy Drop at d=2) ===\n")
    print(f"{'Method':<25} {'50':<10} {'100':<10} {'200':<10} {'400':<10}")
    print("-" * 75)
    
    # Baseline Row (Format: -XX.X)
    # Note: Baseline means NO defense, so the drop is maximum.
    # The user example showed negative numbers (e.g. -37.6), implying Change.
    # My previous extraction was (Clean - Poisoned), which is positive Drop.
    # To match User's "Accuracy Drop" table style (negative numbers), I should flip the sign.
    # Or if user meant "Change", then it is (Poisoned - Clean).
    # Let's look at user's example: "No Defense (Baseline) ... -37.6"
    # This implies accuracy DROPPED by 37.6%.
    # So I will output NEGATIVE numbers.
    
    print(f"{'No Defense (Baseline)':<25} {f'-{baseline_val:.1f}':^50}") 
    print("-" * 75)
    
    # Random Row
    row_random = f"{'Random Anchor':<25} "
    for n in [50, 100, 200, 400]:
        val = results[n]['random']
        row_random += f"{-val:<10.1f} "
    print(row_random)
    
    # Hub Row
    row_hub = f"{'Hub Anchor (Ours)':<25} "
    for n in [50, 100, 200, 400]:
        val = results[n]['hub']
        row_hub += f"{-val:<10.1f} "
    print(row_hub)
    print("-" * 75)

except Exception as e:
    print(f"Error extraction: {e}")
    # Debug info
    print("\nDebug Paths:")
    for k, v in paths.items():
        print(f"{k}: {v}")





