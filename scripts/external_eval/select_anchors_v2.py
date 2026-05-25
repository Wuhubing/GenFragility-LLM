"""
select_anchors_v2.py — Anchor selection for v3.3 plan §5.

Given the 100k Wikidata-based knowledge graph and the 30 update targets
(experiments_final_45/{hub,random,tail}_*.json), produce per-target anchor
lists for three modes:

  A0  none                   no anchor (empty list)
  A1  popularity_top{N}      top-N entities by in-degree from hub_pool
  A2  random_non_hub_{N}     random N entities from non-hub_pool (seed=42)

Each anchor is a (head, relation, tail) triple, sampled from outgoing edges
of the chosen head entity in the 100k graph. We pick the *most-cited*
outgoing relation per head, falling back to the first edge if needed.

Yuji's hard requirements:
  1. Same-N head-to-head: hub_pop and random_non_hub use identical N.
  2. Disjoint pool: random_non_hub is sampled strictly from non_hub_pool;
     assert verify_disjoint(hub, random) — zero entity overlap.
  3. Per-target exclusion: a target's own head/tail/poison entity is
     removed from BOTH pools before selection, so anchors never share
     a node with the update neighborhood.

Output (one file per (mode, N) combo with per-target lists keyed by target_id):
  data/external_eval/anchors_hub_top{N}.json
  data/external_eval/anchors_random_non_hub_{N}_seed{S}.json

Schema:
  {
    "metadata": { "mode": "...", "N": 25, "seed": 42, "hub_threshold": 4,
                  "n_targets": 30, "graph_path": "..." },
    "per_target": {
       "hub_3":  [ {"head": "...", "relation": "...", "tail": "..."}, ... ],
       ...
    }
  }

Run:
  conda run -n genfragility python scripts/external_eval/select_anchors_v2.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
TARGETS_DIR = ROOT / "data/ripple_eval/experiments_final_45"
OUT_DIR = ROOT / "data/external_eval"

HUB_THRESHOLD = 8   # paper §4.2: top-5% by in-degree. On the 100k graph,
                    # in_degree >= 8 yields 5,610 nodes (5.61% of 100,015).
                    # (Earlier v3.3 draft suggested >=4 which gives 20% of
                    # nodes — too broad; we tighten to match paper §4.2.)

# 30-target list (per v3.3 plan §1)
PLAN_30_TARGETS = (
    [f"hub_{i}"    for i in [1, 3, 4, 5, 6, 10, 11, 12, 13, 14]] +
    [f"random_{i}" for i in [1, 2, 7, 8, 9, 10, 11, 12, 14, 15]] +
    [f"tail_{i}"   for i in [1, 3, 4, 5, 7, 9, 10, 11, 12, 15]]
)


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        g = pickle.load(f)
    return g["graph"] if isinstance(g, dict) else g


def load_targets():
    """Returns {target_id: {head, relation, tail, poison_answer}} for the
    30 plan-listed targets. Raises if any file is missing."""
    out = {}
    for tid in PLAN_30_TARGETS:
        p = TARGETS_DIR / f"{tid}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing target: {p}")
        d = json.loads(p.read_text())
        tgt = d.get("target", d)
        out[tid] = {
            "head":          tgt["head"],
            "relation":      tgt["relation"],
            "tail":          tgt["tail"],
            "poison_answer": tgt["poison_answer"],
        }
    return out


def load_targets_from_file(path: Path):
    """Block B: load targets from an external JSON file produced by
    convert_external_to_block_a.py:
        { "<sample_id>": {head, relation, tail, poison_answer}, ... }
    Same return shape as load_targets()."""
    d = json.loads(path.read_text())
    # Validate shape on first entry
    if not d:
        raise ValueError(f"Empty targets file: {path}")
    sample = next(iter(d.values()))
    for k in ("head", "relation", "tail", "poison_answer"):
        if k not in sample:
            raise ValueError(f"Targets file missing key {k!r}: {path}")
    return d


def partition_pools(G):
    """Returns (hub_set, non_hub_set) of entity strings, partitioned by
    in-degree >= HUB_THRESHOLD."""
    hub = set()
    non_hub = set()
    for n in G.nodes():
        if G.in_degree(n) >= HUB_THRESHOLD:
            hub.add(n)
        else:
            non_hub.add(n)
    return hub, non_hub


def pick_anchor_triple(G, head, exclude_tails=None, exclude_relations=None):
    """Pick one (head, relation, tail) triple from this head's outgoing edges.
    Strategy: prefer the relation with the highest in-degree tail
    (= the most-citable fact). Falls back to first edge if tied.
    Skips:
      - self-loops (v == head) — dominate ~50% of selections for hubs.
      - reverse-relation edges where G also has the same-relation edge
        in the opposite direction (v → head), which signals the source
        graph has direction-inconsistent relation semantics for this rel
        (e.g. "Australia -CountryOfCity-> Sydney").
      - tails listed in exclude_tails (used to skip target's head/tail/poison
        appearing as anchor objects).
      - relations listed in exclude_relations (used to skip target relation).
      - literal "None" string as tail (data noise).
    Returns None if head has no usable outgoing edge after filtering.
    """
    exclude_tails = exclude_tails or set()
    exclude_relations = exclude_relations or set()
    best = None
    best_score = -1
    for _, v, d in G.out_edges(head, data=True):
        rel = d.get("relation")
        if not rel:
            continue
        if v == head:
            continue  # skip self-loops
        if v in exclude_tails:
            continue  # skip anchor tails that touch the target entity set
        if rel in exclude_relations:
            continue  # skip anchor relations equal to the target relation
        if v == "None" or head == "None":
            continue  # skip literal "None" noise in the graph
        # Skip if the same relation also exists in the reverse direction —
        # that means the graph itself has inconsistent direction for this rel,
        # so we cannot trust the (head, rel, tail) reading.
        if G.has_edge(v, head):
            rev_rels = {dd.get("relation") for _, _, dd in G.out_edges(v, data=True) if dd.get("relation")}
            if rel in rev_rels:
                continue
        score = G.in_degree(v)
        if score > best_score:
            best_score = score
            best = (head, rel, v)
    return best


def select_hub_top_n(hub_pool, exclude, G, n, target_relation=None):
    """Top-N hub entities by in-degree (descending), excluding `exclude`,
    each turned into a (head, relation, tail) triple via pick_anchor_triple.
    Skips entities whose pick_anchor_triple returns None (after filtering
    against exclude as tails and target_relation as relation)."""
    candidates = [(h, G.in_degree(h)) for h in hub_pool if h not in exclude]
    candidates.sort(key=lambda x: (-x[1], x[0]))  # deterministic tiebreak
    excl_rels = {target_relation} if target_relation else set()
    triples = []
    for h, _ in candidates:
        t = pick_anchor_triple(G, h, exclude_tails=exclude, exclude_relations=excl_rels)
        if t is not None:
            triples.append(t)
        if len(triples) >= n:
            break
    return triples


def select_rare_bottom_n(non_hub_pool, exclude, G, n, target_relation=None,
                         target_id=None, seed=42):
    """Bottom-N facts by **TAIL** in-degree (object popularity, per
    paper §5.2 / method.tex L170). This is the *symmetric* counterpart to
    select_hub_top_n, which picks high-TAIL-in-degree facts: Popular uses
    high object popularity, Rare uses low object popularity.

    Pipeline:
      1. Enumerate all candidate (head, relation, tail) edges where
         head ∈ non_hub_pool, applying the same filters as
         pick_anchor_triple (self-loops, exclude, exclude_relations,
         "None" literals, reverse-relation inconsistency).
      2. Sort PRIMARY ascending by G.in_degree(tail) (preserves the
         "lowest-in-degree stratum" guarantee — never trades a tail_in_deg=1
         candidate for a tail_in_deg=2 one), SECONDARY by a per-target
         hash so each target gets a different sample inside the huge
         tail_in_deg=1 stratum (24k+ candidates).
      3. Deduplicate by head (one anchor per head, mirroring Popular's
         one-pick-per-head behavior).

    Why per-target hash tiebreak:
      - tail_in_deg has massive ties at 1 (24,763 candidates), so with a
        pure alphabetical tiebreak every target would get a byte-identical
        anchor list.
      - This mirrors Random's per-target seed shuffle (structural symmetry
        on the same non-hub pool), so Rare vs Random differ ONLY in
        stratum selection (bottom vs uniform), not in per-target stochasticity.
      - Stratum boundary is strictly preserved: sort key is (in_deg, hash);
        in_deg dominates, hash only orders within a tie group.

    Note: we enumerate edges directly instead of going through
    pick_anchor_triple, because that helper picks the highest-tail-in-degree
    edge per head — exactly the opposite of what we want for Rare."""
    excl_rels = {target_relation} if target_relation else set()

    def _h(head, rel, tail):
        # Stable per-target hash. SHA256 because Python's hash() randomizes
        # across processes (PYTHONHASHSEED), which would break reproducibility.
        key = f"{seed}|{target_id}|{head}|{rel}|{tail}".encode("utf-8")
        return hashlib.sha256(key).digest()  # 32 bytes, ordered lexicographically

    candidates = []  # (tail_in_degree, hash, head, relation, tail)
    for h in non_hub_pool:
        if h in exclude:
            continue
        for _, t, d in G.out_edges(h, data=True):
            rel = d.get("relation")
            if not rel:
                continue
            if t == h:
                continue  # skip self-loops
            if t in exclude:
                continue  # skip anchor tails touching the target entity set
            if rel in excl_rels:
                continue  # skip anchor relations equal to the target relation
            if t == "None" or h == "None":
                continue  # skip literal "None" noise
            # Skip if same relation also exists in reverse direction
            if G.has_edge(t, h):
                rev_rels = {dd.get("relation")
                            for _, _, dd in G.out_edges(t, data=True)
                            if dd.get("relation")}
                if rel in rev_rels:
                    continue
            candidates.append((G.in_degree(t), _h(h, rel, t), h, rel, t))

    # Sort by (tail_in_deg ASC, per-target hash). in_deg dominates; hash
    # only re-orders within a tie group → stratum boundary preserved.
    candidates.sort(key=lambda c: (c[0], c[1]))

    # Dedupe by head (one anchor per head, like Popular)
    triples, seen_heads = [], set()
    for _, _, h, r, t in candidates:
        if h in seen_heads:
            continue
        seen_heads.add(h)
        triples.append((h, r, t))
        if len(triples) >= n:
            break
    return triples


def select_random_non_hub(non_hub_pool, exclude, G, n, rng, target_relation=None):
    """Sample N entities uniformly from non_hub_pool - exclude, each turned
    into a (head, relation, tail) triple (with overlap and reverse-relation
    filtering applied)."""
    pool = [h for h in non_hub_pool if h not in exclude]
    rng.shuffle(pool)
    excl_rels = {target_relation} if target_relation else set()
    triples = []
    for h in pool:
        t = pick_anchor_triple(G, h, exclude_tails=exclude, exclude_relations=excl_rels)
        if t is not None:
            triples.append(t)
        if len(triples) >= n:
            break
    if len(triples) < n:
        print(f"  ⚠️  random_non_hub only got {len(triples)}/{n} valid triples "
              f"(non_hub entities without out-edges)")
    return triples


def verify_disjoint(hub_triples, random_triples, target_id, n_value):
    """Hard assert: no entity (head) overlap between hub and random anchors."""
    hub_heads = {t[0] for t in hub_triples}
    random_heads = {t[0] for t in random_triples}
    overlap = hub_heads & random_heads
    assert not overlap, (
        f"DISJOINT VIOLATED for {target_id} N={n_value}: "
        f"{len(overlap)} overlapping heads: {sorted(overlap)[:5]}..."
    )
    return True


def triples_to_json(triples):
    return [{"head": h, "relation": r, "tail": t} for (h, r, t) in triples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", type=int, nargs="+", default=[5, 25, 75, 100])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-baseline", action="store_true",
                    help="Also write empty anchors_none.json baseline.")
    ap.add_argument("--targets-file", type=Path, default=None,
                    help="Block B: external JSON {sample_id: {head, relation, tail, poison_answer}}. "
                         "If set, ignore PLAN_30_TARGETS and load from this file.")
    ap.add_argument("--out-suffix", default="",
                    help="Block B: suffix appended to output filenames, "
                         "e.g. '_block_b_mintaka' -> anchors_hub_top25_block_b_mintaka.json")
    ap.add_argument("--include-rare", action="store_true",
                    help="Also emit anchors_rare_top{N}.json (bottom-N "
                         "in-degree from non-hub pool). Used for the third "
                         "anchor family (Rare) alongside Popular/Random.")
    args = ap.parse_args()

    print("[1/4] Loading graph and targets ...")
    G = load_graph()
    if args.targets_file:
        targets = load_targets_from_file(args.targets_file)
        print(f"      targets source: {args.targets_file} (Block B mode)")
    else:
        targets = load_targets()
        print(f"      targets source: PLAN_30_TARGETS (Block A mode)")
    print(f"      graph nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")
    print(f"      targets   ={len(targets)}")

    print(f"\n[2/4] Partitioning pools (hub_threshold={HUB_THRESHOLD}) ...")
    hub_pool, non_hub_pool = partition_pools(G)
    print(f"      hub_pool     = {len(hub_pool):,}")
    print(f"      non_hub_pool = {len(non_hub_pool):,}")
    print(f"      disjoint by construction: "
          f"{len(hub_pool & non_hub_pool) == 0}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.include_baseline:
        baseline = {
            "metadata": {"mode": "none", "N": 0, "seed": None,
                         "n_targets": len(targets)},
            "per_target": {tid: [] for tid in targets},
        }
        path = OUT_DIR / "anchors_none.json"
        path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))
        print(f"\n[3/4] Wrote baseline -> {path.name}")

    print(f"\n[3/4] Selecting anchors for N={args.n_values}, seed={args.seed} ...")
    sanity_stats = Counter()
    for N in args.n_values:
        hub_per_target = {}
        rand_per_target = {}
        rare_per_target = {}
        for tid, meta in targets.items():
            # per-target exclusion: head, tail, poison
            exclude = {meta["head"], meta["tail"], meta["poison_answer"]}

            hub_triples = select_hub_top_n(hub_pool, exclude, G, N, target_relation=meta["relation"])
            # Use per-target seed = (seed XOR hash(tid)) so different targets
            # get different random samples while staying reproducible.
            rng = random.Random((args.seed, tid))
            rand_triples = select_random_non_hub(non_hub_pool, exclude, G, N, rng, target_relation=meta["relation"])

            verify_disjoint(hub_triples, rand_triples, tid, N)
            sanity_stats["verify_disjoint_pass"] += 1

            hub_per_target[tid] = triples_to_json(hub_triples)
            rand_per_target[tid] = triples_to_json(rand_triples)

            if len(hub_triples) < N:
                sanity_stats["hub_short"] += 1
            if len(rand_triples) < N:
                sanity_stats["random_short"] += 1

            if args.include_rare:
                rare_triples = select_rare_bottom_n(
                    non_hub_pool, exclude, G, N,
                    target_relation=meta["relation"],
                    target_id=tid, seed=args.seed)
                # sanity: rare must not collide with hub (different pools by construction)
                verify_disjoint(hub_triples, rare_triples, tid, N)
                rare_per_target[tid] = triples_to_json(rare_triples)
                if len(rare_triples) < N:
                    sanity_stats["rare_short"] += 1

        hub_path = OUT_DIR / f"anchors_hub_top{N}{args.out_suffix}.json"
        hub_path.write_text(json.dumps({
            "metadata": {"mode": f"popularity_top{N}", "N": N, "seed": None,
                         "hub_threshold": HUB_THRESHOLD,
                         "n_targets": len(targets), "graph_path": str(GRAPH_PATH),
                         "targets_file": str(args.targets_file) if args.targets_file else None},
            "per_target": hub_per_target,
        }, indent=2, ensure_ascii=False))

        rand_path = OUT_DIR / f"anchors_random_non_hub_{N}_seed{args.seed}{args.out_suffix}.json"
        rand_path.write_text(json.dumps({
            "metadata": {"mode": f"random_non_hub_{N}_seed{args.seed}", "N": N,
                         "seed": args.seed, "hub_threshold": HUB_THRESHOLD,
                         "n_targets": len(targets), "graph_path": str(GRAPH_PATH),
                         "targets_file": str(args.targets_file) if args.targets_file else None},
            "per_target": rand_per_target,
        }, indent=2, ensure_ascii=False))

        if args.include_rare:
            rare_path = OUT_DIR / f"anchors_rare_top{N}{args.out_suffix}.json"
            rare_path.write_text(json.dumps({
                "metadata": {"mode": f"rare_top{N}", "N": N,
                             "seed": args.seed,
                             "tiebreak_mode": "hash_per_target_sha256",
                             "tiebreak_key_template": "{seed}|{target_id}|{head}|{rel}|{tail}",
                             "stratum_rule": "primary sort by tail_in_degree ASC; hash only orders within tie groups → stratum boundary preserved",
                             "hub_threshold": HUB_THRESHOLD,
                             "n_targets": len(targets),
                             "graph_path": str(GRAPH_PATH),
                             "targets_file": str(args.targets_file) if args.targets_file else None},
                "per_target": rare_per_target,
            }, indent=2, ensure_ascii=False))
            print(f"      N={N:3d}: {hub_path.name} + {rand_path.name} + {rare_path.name}  "
                  f"(disjoint OK for {len(targets)}/{len(targets)} targets)")
        else:
            print(f"      N={N:3d}: {hub_path.name} + {rand_path.name}  "
                  f"(disjoint OK for {len(targets)}/{len(targets)} targets)")

    print(f"\n[4/4] Sanity check summary:")
    for k, v in sanity_stats.items():
        print(f"      {k}: {v}")

    # Show a sample anchor (first target in the dict)
    sample_tid = next(iter(targets))
    print(f"\n=== Sample anchors for {sample_tid} ===")
    sample_N = args.n_values[1] if len(args.n_values) > 1 else args.n_values[0]
    for mode_path in [
        OUT_DIR / f"anchors_hub_top{sample_N}{args.out_suffix}.json",
        OUT_DIR / f"anchors_random_non_hub_{sample_N}_seed{args.seed}{args.out_suffix}.json",
    ]:
        d = json.loads(mode_path.read_text())
        sample = d["per_target"].get(sample_tid, [])[:5]
        print(f"\n  {mode_path.name} (N={sample_N}, showing first 5 of {len(d['per_target'].get(sample_tid, []))}):")
        for t in sample:
            print(f"    ({t['head']!r}, {t['relation']!r}, {t['tail']!r})")


if __name__ == "__main__":
    main()
