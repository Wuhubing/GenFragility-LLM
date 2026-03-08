import json
import glob
import os

timestamps = ['203033', '203605', '204203']
base_dir = "main_output"

print(f"{'Timestamp':<10} {'Mode':<10} {'Neutral':<5} {'Poison':<5} {'Anchor'}")
print("-" * 50)

for ts in timestamps:
    # Find the directory
    dirs = glob.glob(os.path.join(base_dir, f"integrated_experiment_20260102_{ts}*"))
    if not dirs:
        continue
    d = dirs[0]
    
    # Find meta file
    meta_files = glob.glob(os.path.join(d, "ripple_experiment_013_*/training_data/meta_*.json"))
    if not meta_files:
        print(f"{ts}: No meta file found")
        continue
        
    with open(meta_files[0], 'r') as f:
        meta = json.load(f)
        
    # Extract info
    # The meta structure depends on implementation, but usually has args or config
    # Let's inspect the keys
    # Assuming 'args' or similar exists, or we check the data composition
    
    # If meta doesn't have explicit args, we can infer from 'stats' if available
    # or just print the keys to debug
    
    print(f"{ts:<10} {meta.get('args', {}).get('anchor_mode', '?'):<10} {meta.get('args', {}).get('num_neutral', '?'):<5} {meta.get('args', {}).get('num_poison', '?'):<5} {meta.get('strategy', '?')}")

