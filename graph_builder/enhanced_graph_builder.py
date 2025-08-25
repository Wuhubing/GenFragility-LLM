#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder - Complete Pipeline Integration
Uses a JSON-based ontology, stratified BFS, and robust validation.
"""

import os
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
import random
import logging

from .relations_ontology import KnowledgeTriplet, RelationOntology
from .validation_system import TripletValidator
from .llm_calls_enhanced import LLMInterfaceEnhanced
from .stratified_bfs_scheduler import StratifiedBfsScheduler
from .anti_explosion_triadic import TriadicClosureSystem
from .stats_monitoring import RealTimeMonitor
from .export_system import ExportSystem

class EnhancedGraphBuilder:
    """Complete enhanced knowledge graph construction pipeline."""
    
    def __init__(self, config: Dict):
        """Initialize the enhanced graph builder with configuration."""
        self.config = config
        self.graph = nx.DiGraph()
        
        # 1. Initialize the single source of truth for ontology
        self.ontology = RelationOntology()

        # 2. Initialize core components, passing the ontology instance to each
        self.llm_interface = LLMInterfaceEnhanced(
            api_key_path=self.config.get('api_key_path'),
            cache_dir=self.config.get('cache_dir'),
            ontology=self.ontology
        )
        self.validator = TripletValidator(
            ontology=self.ontology,
            confidence_threshold=self.config.get('confidence_threshold', 0.6),
            candidate_threshold=self.config.get('candidate_threshold', 0.5)
        )
        self.scheduler = StratifiedBfsScheduler(
            graph=self.graph,
            ontology=self.ontology,
            group_quotas=self.config.get('group_quotas', {}),
            diversity_enabled=self.config.get('parallel_domain_diversity', False),
            min_domains=self.config.get('parallel_min_domains', 3)
        )
        self.closure_system = TriadicClosureSystem(
            graph=self.graph,
            validator=self.validator,
            confidence_threshold=self.config.get('confidence_threshold', 0.6),
            candidate_threshold=self.config.get('candidate_threshold', 0.5)
        )
        self.monitor = RealTimeMonitor(
            graph=self.graph,
            ontology=self.ontology,
            early_stop_config=self.config.get('early_stop', {}),
            group_quotas=self.config.get('group_quotas', {})
        )
        self.exporter = ExportSystem(
            output_dir=self.config.get('output_dir', 'results/output')
        )
        
        # 3. State and Configuration
        self.target_nodes = self.config.get('target_nodes', 1000)
        self.triplets_per_query = self.config.get('triplets_per_query', 5)
        self.parallel_frequency = self.config.get('parallel_frequency', 5)
        self.verbose = self.config.get('verbose', True)
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'results/checkpoints')
        self.seed_entities = set()
        self.state = {
            'processed_entities': set(),
            'total_llm_calls': 0,
            'total_triplets_generated': 0,
            'step_count': 0
        }

        if self.config.get('random_seed') is not None:
            random.seed(self.config['random_seed'])

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.get('output_dir', 'results/output'), exist_ok=True)
    
    def initialize_api(self) -> bool:
        """Initialize API connection."""
        return self.llm_interface.initialize_api()
    
    def add_seed_triplets(self, seed_triplets: List[Tuple[str, str, str]]):
        """Add seed triplets to initialize the graph."""
        if self.verbose:
            print(f"🌱 Adding {len(seed_triplets)} seed triplets...")
        
        for head, relation, tail in seed_triplets:
            triplet = KnowledgeTriplet(
                head=head, relation_id=relation, tail=tail,
                confidence=1.0, evidence="Seed triplet"
            )
            self._process_and_add_triplet(triplet)
            self.seed_entities.add(head)
            self.seed_entities.add(tail)
        
        self.scheduler.initialize_from_graph()
        
        if self.verbose:
            print(f"✅ Seeds processed. Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def build_graph(self) -> nx.DiGraph:
        """Main graph construction loop."""
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🚀 Starting graph construction at {datetime.now().strftime('%H:%M:%S')}")
            print(f"🎯 Target: {self.target_nodes} nodes.")

        while self.graph.number_of_nodes() < self.target_nodes:
            self.state['step_count'] += 1

            next_entity = self.scheduler.get_next_entity()
            if not next_entity:
                if self.verbose: print("⏹️ Scheduler queue is empty. Stopping.")
                    break
            
            if self.verbose:
                print(f"\n[{self.state['step_count']}] 👤 Expanding '{next_entity}'... "
                      f"({self.graph.number_of_nodes()}/{self.target_nodes} nodes)")

            new_triplets = self._expand_entity(next_entity)
            
            if new_triplets:
                for triplet in new_triplets:
                    self._process_and_add_triplet(triplet)

            self.state['processed_entities'].add(next_entity)
            self._periodic_checkpoint()
        
        print(f"\n🎉 Construction finished in {(time.time() - start_time)/60:.1f} minutes.")
        self._save_checkpoint(is_final=True)
        return self.graph
    
    def _expand_entity(self, entity: str) -> List[KnowledgeTriplet]:
        """Generates new triplets for an entity using the LLM."""
        # This is a simplified expansion logic.
        # A full implementation would use upstream/downstream/parallel calls.
        prompt = self._get_prompt_for_entity(entity)
        
        raw_triplets = self.llm_interface.generate_triplets(prompt, self.triplets_per_query)
        self.state['total_llm_calls'] += 1
        self.state['total_triplets_generated'] += len(raw_triplets)
        
        return raw_triplets

    def _get_prompt_for_entity(self, entity: str) -> str:
        """Creates a prompt for the LLM to expand an entity."""
        # Example prompt generation logic
        existing_edges = list(self.graph.out_edges(entity, data=True))
        prompt = f"Given the entity '{entity}', generate new knowledge triplets. "
        if existing_edges:
            prompt += "It is already known that:\n"
            for _, tail, data in existing_edges[:3]:
                prompt += f"- {entity} {data['relation']} {tail}\n"
        
        # Add relation diversity hints based on quotas
        target_groups = self.scheduler.get_next_relation_groups()
        prompt += f"\nFocus on relations from these categories: {', '.join(target_groups)}."
        return prompt

    def _process_and_add_triplet(self, triplet: KnowledgeTriplet):
        """Validates a triplet, adds it and its inverse to the graph and scheduler."""
        validation_result = self.validator.validate_and_normalize(triplet)
        
        if validation_result.accept:
            main_triplet = validation_result.normalized_triplet
            self._add_triplet_to_graph(main_triplet)
            self.scheduler.add_entity(main_triplet.head)
            self.scheduler.add_entity(main_triplet.tail)

            if validation_result.inverse_triplet:
                self._add_triplet_to_graph(validation_result.inverse_triplet, is_inverse=True)
        else:
            if self.verbose and "below threshold" not in validation_result.reason:
                logging.warning(f"Rejected: {triplet.to_tuple()} -> {validation_result.reason}")

    def _add_triplet_to_graph(self, triplet: KnowledgeTriplet, is_inverse: bool = False):
        """Adds a single validated triplet to the graph."""
        if not self.graph.has_edge(triplet.head, triplet.tail):
            self.graph.add_edge(
                triplet.head, triplet.tail,
                relation=triplet.relation_id,
                confidence=triplet.confidence,
                group=triplet.group,
                is_inverse=is_inverse
            )

    def _periodic_checkpoint(self):
        """Saves a checkpoint periodically."""
        if self.state['step_count'] % 20 == 0:
            self._save_checkpoint()
    
    def _save_checkpoint(self, is_final: bool = False):
        """Saves the current state of the builder to a pickle file."""
        path = os.path.join(self.checkpoint_dir, "final.pkl" if is_final else "latest.pkl")
        state_to_save = {
            'graph': self.graph,
            'state': self.state,
            'seed_entities': self.seed_entities,
            'validator_state': self.validator.existing_triplets, # Simplified
            'scheduler_state': self.scheduler.get_state()
        }
        with open(path, 'wb') as f:
            pickle.dump(state_to_save, f)
        if self.verbose:
            print(f"💾 Checkpoint saved to {path} ({self.graph.number_of_nodes()} nodes)")

    def load_checkpoint(self) -> bool:
        """Loads the builder state from the latest checkpoint."""
        path = os.path.join(self.checkpoint_dir, "latest.pkl")
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                saved_state = pickle.load(f)
            
            self.graph = saved_state['graph']
            self.state = saved_state['state']
            self.seed_entities = saved_state['seed_entities']
            self.validator.existing_triplets = saved_state['validator_state']
            self.scheduler.load_state(saved_state['scheduler_state'])
            
            # Re-wire components with the loaded graph
            self.scheduler.graph = self.graph
            self.closure_system.graph = self.graph
            self.monitor.graph = self.graph

            if self.verbose:
                print(f"🔄 Resumed from checkpoint with {self.graph.number_of_nodes()} nodes.")
            return True
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}")
            return False

    def export_results(self, filename_prefix: str) -> Dict[str, str]:
        """Exports the final graph and stats."""
        # A full implementation would gather more stats.
        stats = {'nodes': self.graph.number_of_nodes(), 'edges': self.graph.number_of_edges()}
        return self.exporter.export_all(self.graph, filename_prefix, stats)

def create_enhanced_builder(config: Dict) -> EnhancedGraphBuilder:
    """Factory function to create an instance of the enhanced graph builder."""
    return EnhancedGraphBuilder(config)
