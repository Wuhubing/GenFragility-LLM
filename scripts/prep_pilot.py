import json
import os
import shutil

manifest_path = "/home/weibing_wang/GenFragility-LLM/data/ripple_eval/targets_30_v2.json"
src_dir = "/home/weibing_wang/GenFragility-LLM/data/ripple_eval/experiments_30_targets"
dst_dir = "/home/weibing_wang/GenFragility-LLM/data/ripple_eval/pilot_eval"

os.makedirs(dst_dir, exist_ok=True)
for f in os.listdir(dst_dir):
    os.remove(os.path.join(dst_dir, f))

with open(manifest_path, 'r') as f:
    targets = json.load(f)

# Pick first of each type
hubs = [t for t in targets if t['type'] == 'hub']
tails = [t for t in targets if t['type'] == 'tail']
randoms = [t for t in targets if t['type'] == 'random']

selected = [hubs[0], tails[0], randoms[0]]

for tgt in selected:
    src_file = os.path.join(src_dir, f"{tgt['id']}.json")
    dst_file = os.path.join(dst_dir, f"{tgt['id']}.json")
    shutil.copy(src_file, dst_file)
    print(f"Copied {tgt['id']}.json to pilot_eval.")
