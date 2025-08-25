#!/usr/bin/env python3
"""
Test script for 1000-node graph construction with enhanced validation (optimized).
- Uses canonical relation IDs (core 24 only by default)
- Seeds cover multiple relation groups to avoid early bias
- Adds quotas, caps, early-stop, randomness control, and cache
- Fixed early stopping thresholds to prevent premature termination
"""

import os
import time
from datetime import datetime
from graph_builder.enhanced_graph_builder import create_enhanced_builder

def test_1000_nodes():
    """Test enhanced graph construction with 1000 nodes (optimized)."""

    print("🚀 Starting 1000-node Enhanced Graph Construction Test (Fixed)")
    print("=" * 70)

    # ---- Configuration (fixed early stopping thresholds) ----
    config = {
        'target_nodes': 1000,                 # target ~1k nodes
        'triplets_per_query': 6,              # conservative for quality
        'parallel_frequency': 5,              # do parallel every 5 entity steps
        'include_optional_relations': False,  # use core 24 relations only
        'confidence_threshold': 0.6,          # accept ≥ 0.60
        'candidate_threshold': 0.5,           # 0.50–0.60 only for closure
        'max_radius': 3,                      # 3-hop per D0 "hourglass"
        'verbose': True,
        'enable_early_stopping': False,      # DISABLED for full 1000-node run
        'output_dir': 'results/test_1000_output',
        'checkpoint_dir': 'results/test_1000_checkpoints',
        'api_key_path': 'keys/openai.txt',

        # NEW: group quotas (sum ~= 1.0)
        'group_quotas': {
            'Structure': 0.25, 'Spatial': 0.15, 'Temporal': 0.10,
            'Causal': 0.15, 'Function': 0.15, 'Social': 0.20
        },
        # NEW: anti-explosion caps
        'per_entity_caps': {'InstanceOf': 3, 'SubclassOf': 5, 'LocatedIn': 3, 'PartOf': 5, '*': 7},
        'global_relation_soft_cap': 0.15,     # downweight relation if >15% of edges

        # FIXED: early stopping (further adjusted for longer runs)
        'early_stop': {
            'min_nodes': 1000,           # Target nodes
            'min_clustering': 0.18,      # Keep clustering threshold
            'min_triangles': 2500,       # Keep triangle threshold  
            'min_entropy': 4.5,          # RAISED: was 2.5, now 4.5 (seed=4.122)
            'min_group_coverage': 0.95,  # RAISED: was default 0.8, now 0.95
            'patience': 50               # MUCH INCREASED: 10 → 50 steps
        },

        # NEW: reproducibility & caching
        'random_seed': 42,
        'cache_dir': 'results/cache',   # LLM response cache (prompt-hash keyed)

        # NEW: parallel diversity (enforce cross-domain variety per parallel burst)
        'parallel_domain_diversity': True,
        'parallel_min_domains': 3
    }

    # ---- Corrected & diversified seeds (canonical names + multiple groups) ----
    seed_triplets = [
        # Spatial
        ('Beijing', 'CapitalOf', 'China'),
        ('Paris', 'CapitalOf', 'France'),
        ('Tokyo', 'CapitalOf', 'Japan'),
        ('Apple Inc.', 'HeadquarteredIn', 'Cupertino, California'),
        ('Microsoft', 'HeadquarteredIn', 'Redmond, Washington'),

        # Social / Role
        ('Albert Einstein', 'Occupation', 'Physicist'),
        ('Ada Lovelace', 'Occupation', 'Mathematician'),
        ('Tim Berners-Lee', 'Employer', 'CERN'),

        # Work ↔ Creator (CreatedBy is Work/Invention -> Person/Org)
        ('Hamlet', 'CreatedBy', 'William Shakespeare'),
        ('Mona Lisa', 'CreatedBy', 'Leonardo da Vinci'),
        ('Google Search', 'CreatedBy', 'Google LLC'),

        # Structure
        ('Cat', 'SubclassOf', 'Mammal'),
        ('Engine', 'PartOf', 'Car'),

        # Function / Attributes / Causal
        ('Scissors', 'UsedFor', 'Cutting'),
        ('Knife', 'UsedFor', 'Cutting'), # Was CapableOf
        ('Bottle', 'MadeOf', 'Plastic'),
        ('Exercise', 'Causes', 'Sweating'),
    ]

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)

    start_time = time.time()

    try:
        print(f"📊 Configuration: {config['target_nodes']} nodes, {config['triplets_per_query']} triplets/query")
        builder = create_enhanced_builder(config)

        # Initialize API
        if not builder.initialize_api():
            print("❌ Failed to initialize API. Please check keys/openai.txt")
            return False
        print("✅ API initialized successfully")

        # Resume or seed
        resumed = builder.load_checkpoint()
        if not resumed:
            # Optional: pre-validate seeds if the builder exposes a helper
            if hasattr(builder, 'prevalidate_seed_triplets'):
                print(f"🧪 Pre-validating {len(seed_triplets)} seed triplets...")
                seed_triplets_valid = builder.prevalidate_seed_triplets(seed_triplets)
                print(f"🌱 Adding {len(seed_triplets_valid)} validated seed triplets...")
                builder.add_seed_triplets(seed_triplets_valid)
            else:
                print(f"🌱 Adding {len(seed_triplets)} seed triplets...")
                builder.add_seed_triplets(seed_triplets)
        else:
            print("🔄 Resumed from checkpoint")

        # Build
        print(f"\n🔨 Starting graph construction at {datetime.now().strftime('%H:%M:%S')}")
        final_graph = builder.build_graph()

        elapsed = time.time() - start_time

        # Final stats
        print(f"\n{'='*70}")
        print("🎉 Construction completed!")
        print(f"{'='*70}")
        print("📊 Final Results:")
        print(f"   Nodes: {final_graph.number_of_nodes():,}")
        print(f"   Edges: {final_graph.number_of_edges():,}")
        print(f"   Time:  {elapsed:.1f} sec ({elapsed/60:.1f} min)")
        if final_graph.number_of_nodes() > 0:
            avg_deg = (2 * final_graph.number_of_edges()) / final_graph.number_of_nodes()
            print(f"   Avg degree: {avg_deg:.2f}")

        # Export
        print("\n📁 Exporting results...")
        export_paths = builder.export_results("test_1000_graph")
        print(f"✅ Export completed → {config['output_dir']}/")
        for fmt, path in export_paths.items():
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   {fmt}: {os.path.basename(path)} ({size_mb:.1f} MB)")
            except Exception:
                print(f"   {fmt}: {os.path.basename(path)}")

        # Quality metrics (if exposed)
        if hasattr(builder, 'enhanced_validator'):
            stats = builder.enhanced_validator.get_comprehensive_stats()
            print("\n📈 Quality Metrics:")
            if 'combined_metrics' in stats:
                m = stats['combined_metrics']
                print(f"   Relation entropy (H): {m.get('diversity_entropy', 0):.3f}")
                print(f"   Explosion risk:       {m.get('explosion_risk', 0):.3f}")
            if 'triadic_closure' in stats:
                c = stats['triadic_closure']
                print(f"   Triangles:            {c.get('triangle_count', 0):,}")
                print(f"   Clustering coeff:     {c.get('clustering_coefficient', 0):.4f}")
            # Optional: print top-5 relations
            if 'relation_distribution' in stats:
                dist = stats['relation_distribution']
                top5 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                top5_text = ", ".join([f"{r}:{p:.1%}" for r, p in top5])
                print(f"   Top-5 relations:      {top5_text}")

        # Monitoring snapshot (if exposed)
        if hasattr(builder, 'monitor') and hasattr(builder.monitor, 'latest_summary'):
            print("\n📊 Monitoring snapshot:")
            print(builder.monitor.latest_summary())

        return True

    except KeyboardInterrupt:
        print("\n⏹️  Construction interrupted by user")
        if 'builder' in locals():
            builder._save_checkpoint(is_final=True)
            print("💾 Progress saved to checkpoint")
        return False

    except Exception as e:
        print(f"\n❌ Error during construction: {e}")
        import traceback; traceback.print_exc()
        if 'builder' in locals():
            try:
                builder._save_checkpoint(is_final=True)
                print("💾 Partial progress saved to checkpoint")
            except Exception:
                pass
        return False


if __name__ == "__main__":
    ok = test_1000_nodes()
    print("\n🎊 Test completed successfully!" if ok else "\n❌ Test failed or was interrupted")
