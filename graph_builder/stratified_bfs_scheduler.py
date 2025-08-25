#!/usr/bin/env python3
"""
Stratified BFS Scheduler for controlled knowledge graph expansion.
Replaces simple BFS with group-based queuing, entity scoring, and intelligent scheduling.
"""

import math
import random
from collections import deque, defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx
import numpy as np

from .relations_ontology import KnowledgeTriplet, RelationOntology
from .validation_system import TripletValidator, ValidationResult

class EntityScore:
    """Scoring system for entity selection in stratified BFS."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.relation_counts = Counter()
        self.update_global_stats()
    
    def update_global_stats(self):
        """Update global statistics for scoring."""
        self.relation_counts.clear()
        for _, _, data in self.graph.edges(data=True):
            self.relation_counts[data.get('relation', 'Unknown')] += 1
    
    def calculate_score(self, entity: str, processed_entities: Set[str]) -> float:
        """Calculate priority score for entity selection."""
        if entity in processed_entities:
            return 0.0
        
        score = 0.0
        
        # 1. Relation diversity bonus (higher for entities with diverse connections)
        if self.graph.has_node(entity):
            entity_relations = set()
            for _, _, data in self.graph.edges(entity, data=True):
                entity_relations.add(data.get('relation', 'Unknown'))
            for _, _, data in self.graph.in_edges(entity, data=True):
                entity_relations.add(data.get('relation', 'Unknown'))
            
            # Diversity score based on unique relation types
            diversity_score = len(entity_relations) * 0.3
            score += diversity_score
        
        # 2. Triadic closure potential (entities with common neighbors)
        if self.graph.has_node(entity):
            neighbors = set(self.graph.neighbors(entity)) | set(self.graph.predecessors(entity))
            closure_potential = 0
            for neighbor in neighbors:
                neighbor_neighbors = set(self.graph.neighbors(neighbor)) | set(self.graph.predecessors(neighbor))
                closure_potential += len(neighbor_neighbors - neighbors - {entity})
            
            # Normalize and add closure bonus
            if len(neighbors) > 0:
                closure_score = min(closure_potential / (len(neighbors) * 10), 1.0) * 0.4
                score += closure_score
        
        # 3. Group balance bonus (prefer entities that balance relation groups)
        group_balance_score = self._calculate_group_balance_score(entity)
        score += group_balance_score * 0.3
        
        # 4. Random factor for exploration
        score += random.uniform(0, 0.1)
        
        return score
    
    def _calculate_group_balance_score(self, entity: str) -> float:
        """Calculate score based on how entity helps balance relation groups."""
        if not self.relation_counts:
            return 1.0
        
        total_relations = sum(self.relation_counts.values())
        if total_relations == 0:
            return 1.0
        
        # Calculate current group proportions
        current_proportions = {}
        for group in self.group_quotas:
            group_relations = self.get_relations_by_group(group)
            group_count = sum(self.relation_counts[rel] for rel in group_relations)
            current_proportions[group] = group_count / total_relations
        
        # Find most underrepresented group
        max_deficit = 0.0
        for group, target_prop in self.group_quotas.items():
            if target_prop > 0:  # Skip groups with 0 target
                current_prop = current_proportions.get(group, 0.0)
                deficit = max(0, target_prop - current_prop)
                max_deficit = max(max_deficit, deficit)
        
        return max_deficit

class StratifiedBFSScheduler:
    """Main scheduler for stratified BFS expansion with group-based queuing."""
    
    def __init__(self, graph: nx.MultiDiGraph, ontology: RelationOntology = None,
                 validator: TripletValidator = None,
                 include_optional_relations: bool = False,
                 parallel_frequency: int = 5,
                 triplets_per_query: int = 8,
                 group_quotas: Dict[str, float] = None,
                 diversity_enabled: bool = False,
                 min_domains: int = 3):
        self.graph = graph
        self.ontology = ontology or RelationOntology()
        self.validator = validator
        self.include_optional = include_optional_relations
        self.parallel_frequency = parallel_frequency
        self.triplets_per_query = triplets_per_query
        self.group_quotas = group_quotas or self._get_default_group_quotas()
        
        # Group-based entity queues
        self.entity_queues = {group: deque() for group in self.group_quotas.keys()}
        self.entity_queues['General'] = deque()  # Fallback queue
        
        # Relation queue for parallel expansion
        self.relation_queue = deque()
        
        # Tracking structures
        self.processed_entities = set()
        self.processed_relations = set()
        self.step_counter = 0
        
        # Scoring system
        self.entity_scorer = EntityScore(graph)
        
        # Statistics
        self.stats = {
            'entities_processed': 0,
            'relations_processed': 0,
            'triplets_generated': 0,
            'triplets_accepted': 0,
            'triplets_rejected': 0,
            'group_distribution': Counter(),
            'rejection_reasons': Counter()
        }
    
    def _get_default_group_quotas(self) -> Dict[str, float]:
        """Get default group quotas based on the ontology."""
        # Extract groups from ontology
        groups = set()
        for rel_info in self.ontology.get_all_relations().values():
            groups.add(rel_info.get('group', 'Unknown'))
        
        # Default equal distribution
        if groups:
            quota_per_group = 1.0 / len(groups)
            return {group: quota_per_group for group in groups}
        else:
            return {'Unknown': 1.0}
    
    def get_relations_by_group(self, group: str) -> List[str]:
        """Get relations by group from the ontology."""
        relations = []
        for rel_id, rel_info in self.ontology.get_all_relations().items():
            if rel_info.get('group') == group:
                if self.include_optional or rel_info.get('group') != 'Optional':
                    relations.append(rel_id)
        return relations
    
    def add_seed_entities(self, entities: List[str]):
        """Add seed entities to appropriate queues."""
        for entity in entities:
            if entity not in self.processed_entities:
                # Add to general queue initially
                self.entity_queues['General'].append(entity)
                print(f"Added seed entity: {entity}")
    
    def add_seed_triplets(self, triplets: List[Tuple[str, str, str]]):
        """Add seed triplets and populate initial queues."""
        for head, relation, tail in triplets:
            # Add to graph
            if not self.graph.has_edge(head, tail, key=relation):
                self.graph.add_edge(head, tail, key=relation, relation=relation)
            
            # Add entities to queues
            self.add_seed_entities([head, tail])
            
            # Add relation to parallel queue
            if relation not in self.processed_relations:
                self.relation_queue.append(relation)
    
    def select_next_entity(self) -> Optional[Tuple[str, str]]:
        """Select next entity based on group quotas and scoring."""
        
        # Calculate current group proportions
        total_relations = sum(self.validator.relation_counts.values())
        current_proportions = {}
        
        if total_relations > 0:
            for group in self.group_quotas:
                group_relations = self.get_relations_by_group(group)
                group_count = sum(self.validator.relation_counts[rel] for rel in group_relations)
                current_proportions[group] = group_count / total_relations
        else:
            current_proportions = {group: 0.0 for group in self.group_quotas}
        
        # Find most underrepresented group with entities
        best_group = None
        max_deficit = -1.0
        
        for group, target_prop in self.group_quotas.items():
            if target_prop == 0:  # Skip groups with 0 target
                continue
                
            if not self.entity_queues[group]:  # Skip empty queues
                continue
                
            current_prop = current_proportions.get(group, 0.0)
            deficit = target_prop - current_prop
            
            if deficit > max_deficit:
                max_deficit = deficit
                best_group = group
        
        # Fallback to General queue if no group-specific queue has entities
        if best_group is None and self.entity_queues['General']:
            best_group = 'General'
        
        if best_group is None:
            return None
        
        # Select best entity from chosen group queue
        queue = self.entity_queues[best_group]
        candidates = []
        
        # Collect up to 5 candidates for scoring
        for _ in range(min(5, len(queue))):
            if queue:
                candidates.append(queue.popleft())
        
        if not candidates:
            return None
        
        # Score candidates and select best
        best_entity = None
        best_score = -1.0
        
        for entity in candidates:
            if entity not in self.processed_entities:
                score = self.entity_scorer.calculate_score(entity, self.processed_entities)
                if score > best_score:
                    best_score = score
                    best_entity = entity
        
        # Put non-selected candidates back
        remaining_candidates = [e for e in candidates if e != best_entity]
        for entity in reversed(remaining_candidates):  # Maintain order
            queue.appendleft(entity)
        
        return (best_entity, best_group) if best_entity else None
    
    def select_next_relation(self) -> Optional[str]:
        """Select next relation for parallel expansion."""
        if not self.relation_queue:
            return None
        
        # Find underrepresented relations
        total_relations = sum(self.validator.relation_counts.values())
        if total_relations == 0:
            return self.relation_queue.popleft()
        
        # Calculate relation frequencies
        relation_frequencies = {}
        for relation in list(self.relation_queue):
            freq = self.validator.relation_counts[relation] / total_relations
            relation_frequencies[relation] = freq
        
        # Select least frequent relation
        if relation_frequencies:
            best_relation = min(relation_frequencies.keys(), 
                              key=lambda r: relation_frequencies[r])
            self.relation_queue.remove(best_relation)
            return best_relation
        
        return self.relation_queue.popleft()
    
    def process_entity_expansion(self, entity: str, group: str, 
                               llm_functions: Dict) -> List[ValidationResult]:
        """Process downstream and upstream expansion for an entity."""
        
        # Determine target groups for focused expansion
        target_groups = None
        if group != 'General':
            target_groups = [group]
        
        results = []
        
        try:
            # Downstream expansion
            downstream_triplets = llm_functions['downstream'](
                entity, self.triplets_per_query, target_groups, self.include_optional
            )
            
            for triplet in downstream_triplets:
                result = self.validator.validate_and_normalize(triplet)
                results.append(result)
                self._update_stats(result, 'downstream')
            
            # Upstream expansion  
            upstream_triplets = llm_functions['upstream'](
                entity, self.triplets_per_query, target_groups, self.include_optional
            )
            
            for triplet in upstream_triplets:
                result = self.validator.validate_and_normalize(triplet)
                results.append(result)
                self._update_stats(result, 'upstream')
                
        except Exception as e:
            print(f"Error processing entity {entity}: {e}")
        
        return results
    
    def process_parallel_expansion(self, relation: str, 
                                 llm_functions: Dict) -> List[ValidationResult]:
        """Process parallel expansion for a relation."""
        
        results = []
        
        try:
            parallel_triplets = llm_functions['parallel'](
                relation, self.triplets_per_query, True, self.include_optional
            )
            
            for triplet in parallel_triplets:
                result = self.validator.validate_and_normalize(triplet)
                results.append(result)
                self._update_stats(result, 'parallel')
                
        except Exception as e:
            print(f"Error processing relation {relation}: {e}")
        
        return results
    
    def add_validated_triplets_to_graph(self, results: List[ValidationResult]):
        """Add validated triplets to graph and update queues."""
        
        new_entities = set()
        
        for result in results:
            if not result.accept:
                continue
            
            # Add main triplet
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
            
            self.validator.add_validated_triplet(triplet)
            new_entities.update([triplet.head, triplet.tail])
            
            # Add relation to parallel queue
            if triplet.relation_id not in self.processed_relations:
                self.relation_queue.append(triplet.relation_id)
            
            # Add inverse triplet if exists
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
                
                self.validator.add_validated_triplet(inv_triplet)
                new_entities.update([inv_triplet.head, inv_triplet.tail])
        
        # Add new entities to appropriate queues
        self._distribute_entities_to_queues(new_entities)
    
    def _distribute_entities_to_queues(self, entities: Set[str]):
        """Distribute new entities to appropriate group queues."""
        
        for entity in entities:
            if entity in self.processed_entities:
                continue
            
            # Determine best queue based on entity's current connections
            best_group = self._determine_entity_group(entity)
            self.entity_queues[best_group].append(entity)
    
    def _determine_entity_group(self, entity: str) -> str:
        """Determine which group queue an entity should join."""
        
        if not self.graph.has_node(entity):
            return 'General'
        
        # Count relations by group for this entity
        group_counts = Counter()
        
        for _, _, data in self.graph.edges(entity, data=True):
            relation = data.get('relation', 'Unknown')
            all_relations = self.ontology.get_all_relations()
            if relation in all_relations:
                group = all_relations[relation]['group']
                group_counts[group] += 1
        
        for _, _, data in self.graph.in_edges(entity, data=True):
            relation = data.get('relation', 'Unknown')
            all_relations = self.ontology.get_all_relations()
            if relation in all_relations:
                group = all_relations[relation]['group']
                group_counts[group] += 1
        
        # Return most common group, or General if no clear preference
        if group_counts:
            return group_counts.most_common(1)[0][0]
        else:
            return 'General'
    
    def _update_stats(self, result: ValidationResult, expansion_type: str):
        """Update statistics based on validation result."""
        
        self.stats['triplets_generated'] += 1
        
        if result.accept:
            self.stats['triplets_accepted'] += 1
            if result.normalized_triplet:
                self.stats['group_distribution'][result.normalized_triplet.group] += 1
        else:
            self.stats['triplets_rejected'] += 1
            self.stats['rejection_reasons'][result.reason] += 1
    
    def should_do_parallel_expansion(self) -> bool:
        """Determine if this step should do parallel expansion."""
        return (self.step_counter % self.parallel_frequency == 0 and 
                len(self.relation_queue) > 0)
    
    def get_queue_status(self) -> Dict:
        """Get current status of all queues."""
        status = {}
        for group, queue in self.entity_queues.items():
            status[f'{group}_entities'] = len(queue)
        status['relations'] = len(self.relation_queue)
        status['processed_entities'] = len(self.processed_entities)
        status['processed_relations'] = len(self.processed_relations)
        return status
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = self.stats.copy()
        stats.update(self.get_queue_status())
        stats['validator_stats'] = self.validator.get_statistics()
        
        # Calculate efficiency metrics
        if self.stats['triplets_generated'] > 0:
            stats['acceptance_rate'] = self.stats['triplets_accepted'] / self.stats['triplets_generated']
        else:
            stats['acceptance_rate'] = 0.0
        
        return stats
    
    def reset(self):
        """Reset scheduler state."""
        for queue in self.entity_queues.values():
            queue.clear()
        self.relation_queue.clear()
        self.processed_entities.clear()
        self.processed_relations.clear()
        self.step_counter = 0
        self.stats = {
            'entities_processed': 0,
            'relations_processed': 0,
            'triplets_generated': 0,
            'triplets_accepted': 0,
            'triplets_rejected': 0,
            'group_distribution': Counter(),
            'rejection_reasons': Counter()
        }

if __name__ == "__main__":
    # Test the scheduler
    import networkx as nx
    from validation_system import TripletValidator
    
    # Create test graph and validator
    G = nx.MultiDiGraph()
    validator = TripletValidator()
    
    # Create scheduler
    scheduler = StratifiedBFSScheduler(G, validator)
    
    # Add seed triplets
    seed_triplets = [
        ('Beijing', 'CapitalOf', 'China'),
        ('Paris', 'CapitalOf', 'France'),
        ('Einstein', 'Occupation', 'Physicist')
    ]
    
    scheduler.add_seed_triplets(seed_triplets)
    
    print("Initial queue status:", scheduler.get_queue_status())
    
    # Test entity selection
    next_entity = scheduler.select_next_entity()
    if next_entity:
        entity, group = next_entity
        print(f"Selected entity: {entity} from group: {group}")
        scheduler.processed_entities.add(entity)
        scheduler.stats['entities_processed'] += 1
    
    # Test relation selection
    next_relation = scheduler.select_next_relation()
    if next_relation:
        print(f"Selected relation: {next_relation}")
        scheduler.processed_relations.add(next_relation)
        scheduler.stats['relations_processed'] += 1
    
    print("Final statistics:", scheduler.get_statistics())

# Alias for backward compatibility
StratifiedBfsScheduler = StratifiedBFSScheduler
