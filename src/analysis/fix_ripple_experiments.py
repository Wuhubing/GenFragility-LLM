import json
import os
import glob

def fix_experiments():
    target_dir = "results/experiments_ripples_fast_20k/"
    files = glob.glob(os.path.join(target_dir, "ripple_experiment_*.json"))
    
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        if isinstance(data, dict):
            # Try to handle if it's a dict instead of a list (in case it is wrapped)
            # Actually, looking at the head output, it's a single dict per file, or a list?
            # Wait, `head` showed `{ "experiment_id": 1, ... }` so it's a dict! Not a list.
            exps = [data]
        elif isinstance(data, list):
            exps = data
        else:
            print(f"Unknown format in {f}")
            continue
            
        for exp in exps:
            if 'target' in exp:
                pop = exp['target'].get('popularity_category')
                if pop:
                    pop_label = 'hub' if pop == 'high' else ('tail' if pop == 'low' else pop)
                    exp['target']['popularity'] = pop_label
                    if 'ripples' in exp:
                        for dist, items in exp['ripples'].items():
                            for item in items:
                                item['popularity'] = pop_label
                                
        with open(f, 'w') as fp:
            if isinstance(data, dict):
                json.dump(data, fp, indent=2)
            else:
                json.dump(exps, fp, indent=2)
        print(f"Fixed {f}")

if __name__ == "__main__":
    fix_experiments()
