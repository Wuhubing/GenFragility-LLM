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
from .llm_calls_enhanced import LLMInterfaceEnhanced, TRIPLET_SCHEMA_v0_3
from .prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3, create_user_prompt_v0_3
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
        
        # 1. Initialize the ontology - use QA Atomic if specified
        if self.config.get('use_qa_atomic_ontology', False):
            from .qa_atomic_ontology import QAAtomicOntology
            self.ontology = QAAtomicOntology()
            print("✅ Using QA Atomic Ontology (36 function-like relations)")
        else:
            self.ontology = RelationOntology()
            print("✅ Using Standard Relation Ontology")

        # 2. Initialize core components, passing the ontology instance to each
        self.llm_interface = LLMInterfaceEnhanced(
            api_key_path=self.config.get('api_key_path'),
            cache_dir=self.config.get('cache_dir'),
            ontology=self.ontology
        )
        self.validator = TripletValidator(
            ontology=self.ontology,
            confidence_threshold=self.config.get('confidence_threshold', 0.6),
            candidate_threshold=self.config.get('candidate_threshold', 0.5),
            per_entity_caps=self.config.get('per_entity_caps', {}),
            global_relation_soft_cap=self.config.get('global_relation_soft_cap', 0.15)
        )
        self.scheduler = StratifiedBfsScheduler(
            graph=self.graph,
            ontology=self.ontology,
            validator=self.validator,
            group_quotas=self.config.get('group_quotas', {}),
            diversity_enabled=self.config.get('parallel_domain_diversity', False),
            min_domains=self.config.get('parallel_min_domains', 3)
        )
        # Initialize triadic closure components
        from .anti_explosion_triadic import TriadicClosureDetector, AntiExplosionController
        triadic_detector = TriadicClosureDetector(self.graph)
        explosion_controller = AntiExplosionController(
            relation_caps=self.config.get('per_entity_caps', {}),
            global_soft_cap=self.config.get('global_relation_soft_cap', 0.15)
        )
        self.closure_system = TriadicClosureSystem(
            self.validator, triadic_detector, explosion_controller
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
        
        # Scheduler will use the graph automatically, no explicit initialization needed
        
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

            next_entity_info = self.scheduler.select_next_entity()
            if next_entity_info:
                next_entity = next_entity_info[0]  # Extract entity name from tuple
            else:
                next_entity = None
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
        """Generates new triplets for an entity using the LLM with v0.3 prompt system."""
        # Create user prompt using v0.3 system
        user_prompt = create_user_prompt_v0_3(
            seeds=[entity],
            ontology=self.ontology,
            budget=self.triplets_per_query,
            language="en"
        )
        
        # Call LLM with v0.3 system prompt using the standalone function
        from .llm_calls_enhanced import _call_llm_with_cache
        content = _call_llm_with_cache(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            temperature=0.2,
            max_tokens=2000
        )
        
        if not content:
            print(f"❌ No response from LLM for entity '{entity}'")
            return []
        
        # Parse JSONL response and create KnowledgeTriplet objects
        raw_triplets = self._parse_v0_3_response(content)
        print(f"🔍 LLM returned {len(raw_triplets)} raw triplets for '{entity}'")
        
        self.state['total_llm_calls'] += 1
        self.state['total_triplets_generated'] += len(raw_triplets)
        
        # Validate and add triplets to the graph
        validated_count = 0
        for i, triplet in enumerate(raw_triplets):
            result = self.validator.validate_and_normalize(triplet)
            if result.accept:
                main_triplet = result.normalized_triplet
                self._add_triplet_to_graph(main_triplet)
                self.scheduler.add_seed_entities([main_triplet.head, main_triplet.tail])
                validated_count += 1
                print(f"✅ Triplet {i+1} accepted: {triplet.to_tuple()}")
                
                # Add inverse if exists
                if result.inverse_triplet:
                    self._add_triplet_to_graph(result.inverse_triplet, is_inverse=True)
            else:
                print(f"❌ Triplet {i+1} rejected: {triplet.to_tuple()} -> {result.reason}")
        
        print(f"📊 Final: {validated_count}/{len(raw_triplets)} triplets validated for '{entity}'")
        return raw_triplets
    
    def _parse_v0_3_response(self, content: str) -> List[KnowledgeTriplet]:
        """Parse JSON response from v0.3 prompt system into KnowledgeTriplet objects."""
        import json
        import jsonschema
        import re
        
        triplets = []
        
        if not content:
            return triplets
        
        # Method 1: Try to parse as JSONL first
        lines = content.strip().split('\n')
        if self._try_parse_jsonl(lines, triplets):
            return triplets
        
        # Method 2: Try to extract JSON objects from multi-line format
        json_objects = self._extract_json_objects(content)
        
        for obj_num, json_str in enumerate(json_objects, 1):
            try:
                triplet_data = json.loads(json_str)
                
                # Validate against schema
                jsonschema.validate(triplet_data, TRIPLET_SCHEMA_v0_3)
                
                # Create KnowledgeTriplet object
                triplet = KnowledgeTriplet(
                    head=triplet_data['head'],
                    relation_id=triplet_data['relation_id'],
                    tail=triplet_data['tail'],
                    domain_guess=triplet_data['domain_type'],
                    range_guess=triplet_data['range_type'],
                    surface=triplet_data['surface'],
                    evidence=triplet_data['evidence_rationale'],
                    confidence=triplet_data['confidence'],
                    question=triplet_data.get('question', ''),
                    inverse_auto=not triplet_data['is_inverse']
                )
                
                triplets.append(triplet)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Object {obj_num}: JSON decode error: {e}")
                continue
            except jsonschema.ValidationError as e:
                print(f"⚠️ Object {obj_num}: Schema validation error: {e.message}")
                continue
            except Exception as e:
                print(f"⚠️ Object {obj_num}: Unexpected error: {e}")
                continue
        
        return triplets
    
    def _try_parse_jsonl(self, lines: List[str], triplets: List) -> bool:
        """Try to parse as standard JSONL format."""
        import json
        import jsonschema
        
        success_count = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                triplet_data = json.loads(line)
                jsonschema.validate(triplet_data, TRIPLET_SCHEMA_v0_3)
                success_count += 1
                
                # Create KnowledgeTriplet object
                triplet = KnowledgeTriplet(
                    head=triplet_data['head'],
                    relation_id=triplet_data['relation_id'],
                    tail=triplet_data['tail'],
                    domain_guess=triplet_data['domain_type'],
                    range_guess=triplet_data['range_type'],
                    surface=triplet_data['surface'],
                    evidence=triplet_data['evidence_rationale'],
                    confidence=triplet_data['confidence'],
                    question=triplet_data.get('question', ''),
                    inverse_auto=not triplet_data['is_inverse']
                )
                
                triplets.append(triplet)
                
            except (json.JSONDecodeError, jsonschema.ValidationError):
                # If any line fails, this is probably not JSONL format
                return False
        
        return success_count > 0
    
    def _extract_json_objects(self, content: str) -> List[str]:
        """Extract JSON objects from multi-line content."""
        import re
        
        # Remove newlines within JSON strings but preserve object boundaries
        # This regex finds complete JSON objects
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        
        # For nested objects, we need a more sophisticated approach
        json_objects = []
        brace_count = 0
        current_obj = ""
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                current_obj += char
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                current_obj += char
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                current_obj += char
                continue
                
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        current_obj = char
                    else:
                        current_obj += char
                    brace_count += 1
                elif char == '}':
                    current_obj += char
                    brace_count -= 1
                    if brace_count == 0:
                        json_objects.append(current_obj.strip())
                        current_obj = ""
                else:
                    if brace_count > 0:
                        current_obj += char
            else:
                current_obj += char
        
        return json_objects

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
        # Simple implementation: use all groups for now
        target_groups = list(self.scheduler.group_quotas.keys())
        prompt += f"\nFocus on relations from these categories: {', '.join(target_groups)}."
        return prompt

    def _process_and_add_triplet(self, triplet: KnowledgeTriplet):
        """Validates a triplet, adds it and its inverse to the graph and scheduler."""
        validation_result = self.validator.validate_and_normalize(triplet)
        
        if validation_result.accept:
            main_triplet = validation_result.normalized_triplet
            self._add_triplet_to_graph(main_triplet)
            self.scheduler.add_seed_entities([main_triplet.head, main_triplet.tail])

            if validation_result.inverse_triplet:
                self._add_triplet_to_graph(validation_result.inverse_triplet, is_inverse=True)
        else:
            if self.verbose and "below threshold" not in validation_result.reason:
                logging.warning(f"Rejected: {triplet.to_tuple()} -> {validation_result.reason}")

    def _add_triplet_to_graph(self, triplet: KnowledgeTriplet, is_inverse: bool = False):
        """Adds a single validated triplet to the graph."""
        if not self.graph.has_edge(triplet.head, triplet.tail):
            # Ensure question field is properly included
            question = getattr(triplet, 'question', '')
            self.graph.add_edge(
                triplet.head, triplet.tail,
                relation=triplet.relation_id,
                confidence=triplet.confidence,
                group=triplet.group,
                surface=triplet.surface,
                evidence=triplet.evidence,
                question=question,
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
            'scheduler_stats': self.scheduler.get_statistics()
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
            # Scheduler state loading - simplified since get_statistics was saved instead
            if 'scheduler_stats' in saved_state:
                print(f"📊 Loaded checkpoint with scheduler stats: {saved_state['scheduler_stats']}")
            elif 'scheduler_state' in saved_state:
                print(f"📊 Loaded checkpoint with scheduler data")
            
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
        config = self.config  # Use existing config
        return self.exporter.export_complete_graph(self.graph, stats, config, filename_prefix)

def create_enhanced_builder(config: Dict) -> EnhancedGraphBuilder:
    """Factory function to create an instance of the enhanced graph builder."""
    return EnhancedGraphBuilder(config)
