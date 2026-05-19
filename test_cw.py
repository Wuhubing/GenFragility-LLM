import json, glob

f = glob.glob('main_output/Qwen2.5-32B-Instruct_40_targets_experiment/hub_1/comparison_reports/*.json')[0]
data = json.load(open(f))
for item in data.get('unified_results', []):
    if item.get('clean_accuracy') == 1.0 and item.get('poisoned_accuracy') == 0.0:
        print(f"C>W Flip! clean_conf: {item.get('clean_confidence')}, clean_margin: {item.get('clean_margin')}, poisoned_conf: {item.get('poisoned_confidence')}, poisoned_margin: {item.get('poisoned_margin')}")
        break
