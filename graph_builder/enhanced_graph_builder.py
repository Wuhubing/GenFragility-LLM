#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder - Complete Pipeline Integration
Replaces the original build_dense_graph.py with stratified BFS, validation, and monitoring.
"""

import os
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx

# Import all our enhanced modules
from .relations_ontology import KnowledgeTriplet, RELATION_GROUPS
from .validation_system import TripletValidator
from .llm_calls_enhanced import (
    load_api_key, find_downstream_triplets_enhanced, 
    find_upstream_triplets_enhanced, find_parallel_triplets_enhanced,
    get_cache_statistics, _save_cache
)
from .stratified_bfs_scheduler import StratifiedBFSScheduler
from .anti_explosion_triadic import create_enhanced_validation_pipeline
from .stats_monitoring import create_monitoring_system, RealTimeMonitor, EarlyStoppingCriteria
from .export_system import create_exporter

class EnhancedGraphBuilder:
    """Complete enhanced knowledge graph construction pipeline."""
    
    def __init__(self, config: Dict = None):
        """Initialize the enhanced graph builder with configuration."""
        
        # Default configuration
        self.config = {
            'target_nodes': 3000,
            'triplets_per_query': 8,
            'parallel_frequency': 5,
            'save_interval': 100,
            'include_optional_relations': False,
            'confidence_threshold': 0.6,
            'candidate_threshold': 0.5,
            'max_radius': 3,
            'model': 'gpt-4o-mini',
            'temperature': 0.2,
            'api_key_path': 'keys/openai.txt',
            'output_dir': 'results/output',
            'checkpoint_dir': 'results/checkpoints',
            'enable_monitoring': True,
            'enable_early_stopping': True,
            'verbose': True
        }
        
        if config:
            self.config.update(config)
        
        # Initialize core components
        self.graph = nx.MultiDiGraph()
        self.base_validator = TripletValidator(
            include_optional_relations=self.config['include_optional_relations'],
            confidence_threshold=self.config['confidence_threshold'],
            candidate_threshold=self.config['candidate_threshold']
        )
        
        # Enhanced validation pipeline
        self.enhanced_validator = create_enhanced_validation_pipeline(
            self.base_validator, self.graph,
            max_radius=self.config['max_radius']
        )
        
        # Stratified BFS scheduler
        self.scheduler = StratifiedBFSScheduler(
            self.graph, self.base_validator,
            include_optional_relations=self.config['include_optional_relations'],
            parallel_frequency=self.config['parallel_frequency'],
            triplets_per_query=self.config['triplets_per_query']
        )
        
        # Monitoring system initialization
        self.monitor = None
        self.early_stopping = None

        if self.config['enable_monitoring']:
            # Create monitor
            monitor_instance, early_stopping_instance = create_monitoring_system(
                target_nodes=self.config['target_nodes'],
                early_stop_config=self.config.get('early_stop')
            )
            self.monitor = monitor_instance

            # Only enable early stopping if the flag is explicitly True
            if self.config['enable_early_stopping']:
                self.early_stopping = early_stopping_instance
        
        # Export system
        self.exporter = create_exporter(self.config['output_dir'])
        
        # Seed entities tracking
        self.seed_entities = set()
        
        # Construction state
        self.is_initialized = False
        self.construction_start_time = None
        
        # Create directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def initialize_api(self) -> bool:
        """Initialize API connection."""
        success = load_api_key(self.config['api_key_path'])
        if success and self.config['verbose']:
            print("✅ API key loaded successfully")
            cache_stats = get_cache_statistics()
            print(f"📊 Cache: {cache_stats['total_cached_responses']} responses")
        return success
    
    def add_seed_triplets(self, seed_triplets: List[Tuple[str, str, str]]):
        """Add seed triplets to initialize the graph."""
        if self.config['verbose']:
            print(f"🌱 Adding {len(seed_triplets)} seed triplets...")
        
        # Convert to KnowledgeTriplet objects
        enhanced_seeds = []
        for head, relation, tail in seed_triplets:
            triplet = KnowledgeTriplet(
                head=head, relation_id=relation, tail=tail,
                confidence=1.0,  # High confidence for seeds
                evidence="Seed triplet",
                surface=f"{head} {relation} {tail}"
            )
            enhanced_seeds.append(triplet)
        
        # Validate and add seeds
        validation_results = self.enhanced_validator.validate_with_closure_priority(
            enhanced_seeds, self.graph, self.seed_entities
        )
        
        accepted_count = 0
        for result in validation_results:
            if result.accept:
                # Add to graph
                triplet = result.normalized_triplet
                if not self.graph.has_edge(triplet.head, triplet.tail, key=triplet.relation_id):
                    self.graph.add_edge(
                        triplet.head, triplet.tail,
                        key=triplet.relation_id,
                        relation=triplet.relation_id,
                        confidence=triplet.confidence,
                        surface=triplet.surface,
                        evidence=triplet.evidence,
                        group=triplet.group,
                        created_at=triplet.created_at,
                        gen_params=triplet.gen_params
                    )
                
                # Track seed entities
                self.seed_entities.update([triplet.head, triplet.tail])
                accepted_count += 1
                
                # Add inverse if exists
                if result.inverse_triplet:
                    inv_triplet = result.inverse_triplet
                    if not self.graph.has_edge(inv_triplet.head, inv_triplet.tail, key=inv_triplet.relation_id):
                        self.graph.add_edge(
                            inv_triplet.head, inv_triplet.tail,
                            key=inv_triplet.relation_id,
                            relation=inv_triplet.relation_id,
                            confidence=inv_triplet.confidence,
                            surface=inv_triplet.surface,
                            evidence=inv_triplet.evidence,
                            group=inv_triplet.group,
                            created_at=inv_triplet.created_at,
                            gen_params=inv_triplet.gen_params,
                            is_inverse=True
                        )
        
        # Initialize scheduler with seeds
        self.scheduler.add_seed_triplets([(t.head, t.relation_id, t.tail) for t in enhanced_seeds])
        
        if self.config['verbose']:
            print(f"✅ Added {accepted_count}/{len(seed_triplets)} seed triplets")
            print(f"📈 Graph now has {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        self.is_initialized = True
    
    def build_graph(self) -> nx.MultiDiGraph:
        """Main graph construction loop."""
        
        if not self.is_initialized:
            raise ValueError("Graph not initialized. Call add_seed_triplets() first.")
        
        if not load_api_key(self.config['api_key_path']):
            raise ValueError("Failed to initialize API. Check API key file.")
        
        self.construction_start_time = time.time()
        
        if self.config['verbose']:
            print(f"\n🚀 Starting enhanced graph construction")
            print(f"🎯 Target: {self.config['target_nodes']} nodes")
            print(f"⚙️  Config: {self.config['triplets_per_query']} triplets/query, "
                  f"parallel every {self.config['parallel_frequency']} steps")
            print(f"📊 Monitoring: {'Enabled' if self.monitor else 'Disabled'}")
            print(f"⏹️  Early stopping: {'Enabled' if self.early_stopping else 'Disabled'}")
        
        # LLM function mapping
        llm_functions = {
            'downstream': find_downstream_triplets_enhanced,
            'upstream': find_upstream_triplets_enhanced,
            'parallel': find_parallel_triplets_enhanced
        }
        
        step_count = 0
        last_save_time = time.time()
        
        # Main construction loop
        while (self.graph.number_of_nodes() < self.config['target_nodes']):
            
            step_count += 1
            current_time = time.time()
            
            # Record metrics if monitoring enabled
            if self.monitor:
                scheduler_stats = self.scheduler.get_statistics()
                self.monitor.record_metrics(self.graph, scheduler_stats)
            
            # Check early stopping
            if self.early_stopping and self.monitor:
                current_metrics = self.monitor.get_current_metrics()
                trends = self.monitor.get_trend_analysis()
                should_stop, reason = self.early_stopping.should_stop(current_metrics, trends)
                
                if should_stop:
                    if self.config['verbose']:
                        print(f"\n⏹️  Early stopping triggered: {reason}")
                    break
            
            # Decide between entity and parallel expansion
            if self.scheduler.should_do_parallel_expansion():
                # Parallel expansion
                relation = self.scheduler.select_next_relation()
                if relation:
                    if self.config['verbose']:
                        nodes_count = self.graph.number_of_nodes()
                        queue_status = self.scheduler.get_queue_status()
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔗 Parallel: '{relation}' "
                              f"[{nodes_count}/{self.config['target_nodes']} nodes] "
                              f"[Queues: {queue_status['General_entities']}G, {queue_status['relations']}R]")
                    
                    # Process parallel expansion
                    results = self.scheduler.process_parallel_expansion(relation, llm_functions)
                    self.scheduler.add_validated_triplets_to_graph(results)
                    self.scheduler.processed_relations.add(relation)
                    self.scheduler.stats['relations_processed'] += 1
                    
                    if self.monitor:
                        self.monitor.increment_api_calls(1)
                        self.monitor.increment_relations_processed(1)
            
            else:
                # Entity expansion
                entity_info = self.scheduler.select_next_entity()
                if entity_info:
                    entity, group = entity_info
                    
                    if self.config['verbose']:
                        nodes_count = self.graph.number_of_nodes()
                        queue_status = self.scheduler.get_queue_status()
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 👤 Entity: '{entity}' ({group}) "
                              f"[{nodes_count}/{self.config['target_nodes']} nodes] "
                              f"[Queues: {queue_status['General_entities']}G, {queue_status['relations']}R]")
                    
                    # Process entity expansion
                    results = self.scheduler.process_entity_expansion(entity, group, llm_functions)
                    self.scheduler.add_validated_triplets_to_graph(results)
                    self.scheduler.processed_entities.add(entity)
                    self.scheduler.stats['entities_processed'] += 1
                    
                    if self.monitor:
                        self.monitor.increment_api_calls(2)  # Downstream + upstream
                        self.monitor.increment_entities_processed(1)
                
                else:
                    if self.config['verbose']:
                        print("⚠️  No more entities to process")
                    break
            
            # Update scheduler step counter
            self.scheduler.step_counter += 1
            
            # Periodic checkpoint saving
            if (current_time - last_save_time) > (self.config['save_interval'] * 0.3):  # Time-based saving
                self._save_checkpoint()
                last_save_time = current_time
                
                # Update triadic detector
                self.enhanced_validator.triadic_detector.update_graph(self.graph)
            
            # Rate limiting
            time.sleep(0.3)
        
        # Final construction statistics
        end_time = time.time()
        construction_time = end_time - self.construction_start_time
        
        final_stats = self._get_final_statistics(construction_time)
        
        if self.config['verbose']:
            self._print_final_summary(final_stats)
        
        # Save final checkpoint and export
        self._save_checkpoint(is_final=True)
        _save_cache()  # Save LLM response cache
        
        return self.graph
    
    def export_results(self, base_filename: str = None) -> Dict[str, str]:
        """Export the constructed graph in all formats."""
        
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"enhanced_knowledge_graph_{timestamp}"
        
        # Gather comprehensive statistics
        construction_stats = self._get_final_statistics(
            time.time() - self.construction_start_time if self.construction_start_time else 0
        )
        
        # Export with all metadata
        export_paths = self.exporter.export_complete_graph(
            self.graph, construction_stats, self.config, base_filename
        )
        
        if self.config['verbose']:
            print(f"\n📁 Export completed to {self.config['output_dir']}/")
            for format_name, path in export_paths.items():
                file_size = os.path.getsize(path) / (1024*1024)  # MB
                print(f"   {format_name}: {os.path.basename(path)} ({file_size:.1f} MB)")
        
        return export_paths
    
    def _get_final_statistics(self, construction_time: float) -> Dict:
        """Gather comprehensive final statistics."""
        
        stats = {
            'construction_time_seconds': construction_time,
            'construction_time_minutes': construction_time / 60,
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'seed_entities_count': len(self.seed_entities)
        }
        
        # Add scheduler statistics
        if hasattr(self.scheduler, 'get_statistics'):
            scheduler_stats = self.scheduler.get_statistics()
            stats.update(scheduler_stats)
        
        # Add monitoring statistics
        if self.monitor:
            perf_stats = self.monitor.get_performance_summary()
            stats.update(perf_stats)
        
        # Add enhanced validator statistics
        if hasattr(self.enhanced_validator, 'get_comprehensive_stats'):
            enhanced_stats = self.enhanced_validator.get_comprehensive_stats()
            stats.update(enhanced_stats)
        
        # Add cache statistics
        cache_stats = get_cache_statistics()
        stats['cache_stats'] = cache_stats
        
        return stats
    
    def _print_final_summary(self, stats: Dict):
        """Print comprehensive final summary."""
        
        print(f"\n{'='*80}")
        print(f"🎉 ENHANCED GRAPH CONSTRUCTION COMPLETED")
        print(f"{'='*80}")
        
        print(f"📊 FINAL STATISTICS:")
        print(f"   Nodes: {stats.get('total_nodes', 0):,}")
        print(f"   Edges: {stats.get('total_edges', 0):,}")
        print(f"   Seed entities: {stats.get('seed_entities_count', 0)}")
        print(f"   Construction time: {stats.get('construction_time_minutes', 0):.1f} minutes")
        
        print(f"\n🔄 PROCESSING STATISTICS:")
        print(f"   API calls: {stats.get('api_calls', 0):,}")
        print(f"   Entities processed: {stats.get('entities_processed', 0):,}")
        print(f"   Relations processed: {stats.get('relations_processed', 0):,}")
        print(f"   Triplets generated: {stats.get('triplets_generated', 0):,}")
        print(f"   Triplets accepted: {stats.get('triplets_accepted', 0):,}")
        
        if stats.get('triplets_generated', 0) > 0:
            acceptance_rate = stats.get('triplets_accepted', 0) / stats.get('triplets_generated', 1)
            print(f"   Acceptance rate: {acceptance_rate:.1%}")
        
        print(f"\n📈 QUALITY METRICS:")
        enhanced_stats = stats.get('combined_metrics', {})
        print(f"   Relation entropy: {enhanced_stats.get('diversity_entropy', 0):.3f}")
        print(f"   Clustering coefficient: {stats.get('clustering_coefficient', 0):.4f}")
        print(f"   Triangle count: {stats.get('triangle_count', 0):,}")
        print(f"   Explosion risk: {enhanced_stats.get('explosion_risk', 0):.3f}")
        
        print(f"\n💾 CACHE STATISTICS:")
        cache_stats = stats.get('cache_stats', {})
        print(f"   Cached responses: {cache_stats.get('total_cached_responses', 0):,}")
        
        if stats.get('construction_time_seconds', 0) > 0:
            print(f"\n⚡ PERFORMANCE:")
            nodes_per_min = (stats.get('total_nodes', 0) / stats.get('construction_time_minutes', 1))
            api_per_min = (stats.get('api_calls', 0) / stats.get('construction_time_minutes', 1))
            print(f"   Nodes per minute: {nodes_per_min:.1f}")
            print(f"   API calls per minute: {api_per_min:.1f}")
        
        print(f"{'='*80}")
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save construction checkpoint."""
        
        checkpoint_data = {
            'graph': self.graph,
            'config': self.config,
            'scheduler_state': {
                'processed_entities': self.scheduler.processed_entities,
                'processed_relations': self.scheduler.processed_relations,
                'step_counter': self.scheduler.step_counter,
                'stats': self.scheduler.stats
            },
            'validator_state': {
                'existing_triplets': self.base_validator.existing_triplets,
                'relation_counts': self.base_validator.relation_counts,
                'entity_relation_counts': dict(self.base_validator.entity_relation_counts)
            },
            'seed_entities': self.seed_entities,
            'timestamp': datetime.now().isoformat(),
            'is_final': is_final
        }
        
        checkpoint_name = "final_checkpoint.pkl" if is_final else "latest_checkpoint.pkl"
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], checkpoint_name)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        if self.config['verbose'] and not is_final:
            print(f"💾 Checkpoint saved: {self.graph.number_of_nodes()} nodes")
    
    def load_checkpoint(self, checkpoint_path: str = None) -> bool:
        """Load construction checkpoint."""
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], "latest_checkpoint.pkl")
        
        if not os.path.exists(checkpoint_path):
            if self.config['verbose']:
                print("ℹ️  No checkpoint found, starting fresh")
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.graph = checkpoint_data['graph']
            self.seed_entities = checkpoint_data.get('seed_entities', set())
            
            # Restore scheduler state
            scheduler_state = checkpoint_data.get('scheduler_state', {})
            self.scheduler.graph = self.graph
            self.scheduler.processed_entities = scheduler_state.get('processed_entities', set())
            self.scheduler.processed_relations = scheduler_state.get('processed_relations', set())
            self.scheduler.step_counter = scheduler_state.get('step_counter', 0)
            self.scheduler.stats = scheduler_state.get('stats', self.scheduler.stats)
            
            # Restore validator state
            validator_state = checkpoint_data.get('validator_state', {})
            self.base_validator.existing_triplets = validator_state.get('existing_triplets', set())
            self.base_validator.relation_counts = validator_state.get('relation_counts', {})
            self.base_validator.entity_relation_counts = validator_state.get('entity_relation_counts', {})
            
            # Update enhanced validator
            self.enhanced_validator = create_enhanced_validation_pipeline(
                self.base_validator, self.graph,
                max_radius=self.config['max_radius']
            )
            
            self.is_initialized = True
            
            if self.config['verbose']:
                print(f"✅ Checkpoint loaded: {self.graph.number_of_nodes()} nodes, "
                      f"{self.graph.number_of_edges()} edges")
                print(f"📅 Saved: {checkpoint_data.get('timestamp', 'Unknown time')}")
            
            return True
            
        except Exception as e:
            if self.config['verbose']:
                print(f"❌ Failed to load checkpoint: {e}")
            return False

def create_enhanced_builder(config: Dict = None) -> EnhancedGraphBuilder:
    """Factory function to create enhanced graph builder."""
    return EnhancedGraphBuilder(config)

# Default seed triplets for testing/demo
DEFAULT_SEED_TRIPLETS = [
    ('Beijing', 'CapitalOf', 'China'),
    ('Paris', 'CapitalOf', 'France'),
    ('Einstein', 'Occupation', 'Physicist'),
    ('Apple', 'HeadquarteredIn', 'Cupertino'),
    ('Shakespeare', 'CreatedBy', 'Hamlet')
]

if __name__ == "__main__":
    # Demo/test run
    print("🚀 Enhanced Knowledge Graph Builder - Demo Run")
    
    # Configuration for small test
    config = {
        'target_nodes': 100,  # Small for demo
        'triplets_per_query': 4,
        'parallel_frequency': 3,
        'save_interval': 50,
        'verbose': True,
        'output_dir': 'results/demo_output'
    }
    
    # Create builder
    builder = create_enhanced_builder(config)
    
    # Initialize API
    if not builder.initialize_api():
        print("❌ Failed to initialize API. Please check your API key file.")
        exit(1)
    
    # Try to load checkpoint first
    if not builder.load_checkpoint():
        # Add seed triplets if no checkpoint
        builder.add_seed_triplets(DEFAULT_SEED_TRIPLETS)
    
    # Build graph
    try:
        final_graph = builder.build_graph()
        
        # Export results
        export_paths = builder.export_results("demo_graph")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Results saved to: {config['output_dir']}/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Construction interrupted by user")
        # Still save what we have
        builder._save_checkpoint(is_final=True)
        print(f"💾 Progress saved to checkpoint")
    
    except Exception as e:
        print(f"\n❌ Error during construction: {e}")
        import traceback
        traceback.print_exc()
