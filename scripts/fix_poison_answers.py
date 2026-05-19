"""
Fix poison_answer fields in experiment files.

All existing experiment files have poison_answer = "Fake Counterfactual Answer"
(a hardcoded placeholder). This script replaces each one with a real plausible
counterfactual by sampling another tail from the same relation in the graph.

Strategy: for (head, relation, true_tail), find all other triples with the
same relation, collect their tails, and pick one that differs from true_tail.
This guarantees the fake answer is the same entity type (e.g. another city,
another country) without needing an external API.

Usage:
    conda run -n genfragility python scripts/fix_poison_answers.py \
        --exp_dirs data/ripple_eval/experiments_30_targets data/ripple_eval/pilot_eval \
        --graph results/checkpoints/final.pkl
"""

import argparse
import json
import os
import pickle
import random
import glob
from collections import defaultdict

PLACEHOLDER = "Fake Counterfactual Answer"


def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        data = pickle.load(f)
    G = data["graph"] if isinstance(data, dict) else data
    return G


def build_relation_tail_index(G):
    """Map relation → set of all tail entities seen in the graph."""
    index = defaultdict(set)
    for u, v, attr in G.edges(data=True):
        rel = attr.get("relation", "")
        if rel and not attr.get("is_inverse", False):
            index[rel].add(v)
    return index


def pick_counterfactual(relation, true_tail, index, rng):
    candidates = index.get(relation, set()) - {true_tail}
    if not candidates:
        return None
    return rng.choice(sorted(candidates))


def fix_file(path, index, rng, dry_run=False):
    with open(path) as f:
        exp = json.load(f)

    target = exp.get("target", {})
    current = target.get("poison_answer", "")

    if current != PLACEHOLDER:
        return False, current  # already fixed

    relation  = target.get("relation", "")
    true_tail = target.get("tail", "")

    fake = pick_counterfactual(relation, true_tail, index, rng)
    if not fake:
        print(f"  [WARN] No candidate found for relation={relation!r}, skipping {path}")
        return False, None

    if not dry_run:
        target["poison_answer"] = fake
        exp["target"] = target
        with open(path, "w") as f:
            json.dump(exp, f, indent=2, ensure_ascii=False)

    return True, fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True,
                        help="Directories containing experiment JSON files")
    parser.add_argument("--graph", default="results/checkpoints/final.pkl",
                        help="Path to graph pickle")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print changes without writing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading graph from {args.graph} ...")
    G = load_graph(args.graph)
    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    print("Building relation→tail index ...")
    index = build_relation_tail_index(G)
    print(f"  Relations indexed: {len(index)}")

    files = []
    for d in args.exp_dirs:
        files.extend(glob.glob(os.path.join(d, "*.json")))
    files = sorted(set(files))
    print(f"\nFiles to process: {len(files)}")

    fixed, skipped, failed = 0, 0, 0
    for path in files:
        changed, fake = fix_file(path, index, rng, dry_run=args.dry_run)
        name = os.path.basename(path)
        if changed:
            print(f"  ✅ {name:25s} → poison_answer = {fake!r}")
            fixed += 1
        elif fake is None:
            print(f"  ❌ {name:25s} → no candidate found")
            failed += 1
        else:
            print(f"  ⏭  {name:25s} → already set: {fake!r}")
            skipped += 1

    print(f"\nDone. fixed={fixed}  skipped={skipped}  failed={failed}")
    if args.dry_run:
        print("(dry_run mode — no files written)")


if __name__ == "__main__":
    main()
