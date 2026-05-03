import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ZsRE/CounterFact to ripple experiment format")
    parser.add_argument('--input', type=str, required=True, help="Input dataset json")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--dataset', type=str, choices=['zsre', 'counterfact'], required=True)
    parser.add_argument('--limit', type=int, default=5, help="Number of experiments to extract")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    experiments = []
    # We will pick exactly `--limit` to align with the rest of pipeline expectations
    for i, item in enumerate(data[:args.limit]): 
        if args.dataset == 'zsre':
            subject = item.get('subject')
            target_new = item.get('answers', [''])[0]
            question = item.get('src')
            relation = "unknown_rel"
        else: # counterfact
            subject = item.get('requested_rewrite', {}).get('subject')
            target_new = item.get('requested_rewrite', {}).get('target_new', {}).get('str')
            question = item.get('requested_rewrite', {}).get('prompt', '').replace('{}', subject)
            relation = item.get('requested_rewrite', {}).get('relation_id', 'unknown_rel')
            
        if not subject or not target_new:
            continue
            
        target_triplet = {
            "triplet": [subject, relation, target_new],
            "head": subject,
            "relation": relation,
            "tail": target_new,
            "question": question,
            "popularity": "mid" # Default to mid for external datasets unless mapped
        }
        
        # Single experiment schema expected by main.py
        exp = {
            "experiment_id": i,
            "target": target_triplet,
            "ripples": {"d1": [], "d2": [], "d3": [], "d4": [], "d5": []} 
        }
        
        out_path = os.path.join(args.output_dir, f"ripple_experiment_{i:03d}.json")
        with open(out_path, 'w') as f:
            json.dump(exp, f, indent=2)
            
        experiments.append(out_path)
        
    print(f"Converted {len(experiments)} samples to {args.output_dir}")

if __name__ == "__main__":
    main()
