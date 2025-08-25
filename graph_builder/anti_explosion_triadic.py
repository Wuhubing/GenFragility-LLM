#!/usr/bin/env python3
"""
Anti-Explosion and Triadic Closure system for controlled graph densification.
Implements caps, closure detection, and priority boosting for triangle completion.
"""

import math
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import networkx as nx
import itertools

from .relations_ontology import KnowledgeTriplet
from .validation_system import ValidationResult

class TriadicClosureDetector:
    """Detect and prioritize triadic closure opportunities."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.triangle_cache = {}
        self.closure_opportunities = defaultdict(list)
        self._update_closure_cache()
    
    def _update_closure_cache(self):
        """Update cache of potential closure opportunities."""
        self.closure_opportunities.clear()
        
        # Find all potential triangles
        nodes = list(self.graph.nodes())
        
        for node_a in nodes:
            neighbors_a = set(self.graph.neighbors(node_a)) | set(self.graph.predecessors(node_a))
            
            for node_b in neighbors_a:
                if node_b == node_a:
                    continue
                    
                neighbors_b = set(self.graph.neighbors(node_b)) | set(self.graph.predecessors(node_b))
                
                # Find common neighbors (potential third vertex of triangle)
                common_neighbors = neighbors_a & neighbors_b - {node_a, node_b}
                
                for node_c in common_neighbors:
                    # Check if A-C edge is missing for triangle completion
                    if not self._has_any_edge(node_a, node_c):
                        self.closure_opportunities[node_a].append((node_b, node_c))
    
    def _has_any_edge(self, node1: str, node2: str) -> bool:
        """Check if there's any edge between two nodes (in either direction)."""
        return (self.graph.has_edge(node1, node2) or 
                self.graph.has_edge(node2, node1))
    
    def get_closure_priority(self, triplet: KnowledgeTriplet) -> float:
        """Calculate priority bonus for triplet based on closure potential."""
        head, tail = triplet.head, triplet.tail
        
        # If this edge would complete triangles, give it higher priority
        if head in self.closure_opportunities:
            for intermediate, target in self.closure_opportunities[head]:
                if target == tail or intermediate == tail:
                    return 1.0  # High priority for triangle completion
        
        if tail in self.closure_opportunities:
            for intermediate, target in self.closure_opportunities[tail]:
                if target == head or intermediate == head:
                    return 1.0
        
        # Check if adding this edge creates new closure opportunities
        # Only if both nodes exist in graph
        if not (self.graph.has_node(head) and self.graph.has_node(tail)):
            return 0.0  # No closure benefit if nodes don't exist yet
        
        head_neighbors = set(self.graph.neighbors(head)) | set(self.graph.predecessors(head))
        tail_neighbors = set(self.graph.neighbors(tail)) | set(self.graph.predecessors(tail))
        
        # Potential for future triangles
        common_neighbors = head_neighbors & tail_neighbors
        if len(common_neighbors) > 0:
            return 0.5  # Medium priority for potential closure enablement
        
        return 0.0  # No closure benefit
    
    def count_triangles(self) -> int:
        """Count existing triangles in the graph."""
        triangles = 0
        nodes = list(self.graph.nodes())
        
        for a, b, c in itertools.combinations(nodes, 3):
            # Check if all three pairs are connected (in any direction)
            if (self._has_any_edge(a, b) and 
                self._has_any_edge(b, c) and 
                self._has_any_edge(a, c)):
                triangles += 1
        
        return triangles
    
    def calculate_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient."""
        if self.graph.number_of_nodes() < 3:
            return 0.0
        
        # Convert to undirected for clustering calculation
        undirected = self.graph.to_undirected()
        
        try:
            clustering_values = nx.clustering(undirected).values()
            return sum(clustering_values) / len(clustering_values) if clustering_values else 0.0
        except:
            return 0.0
    
    def update_graph(self, graph: nx.MultiDiGraph):
        """Update internal graph reference and recalculate cache."""
        self.graph = graph
        self._update_closure_cache()

class AntiExplosionController:
    """Control graph explosion through caps and diversity enforcement."""
    
    def __init__(self, relation_caps: Dict[str, int] = None, 
                 global_soft_cap: float = 0.15,
                 max_radius: int = 3):
        self.relation_caps = relation_caps or {'InstanceOf': 3, 'SubclassOf': 5, 'LocatedIn': 3, 'PartOf': 5, '*': 7}
        self.global_soft_cap = global_soft_cap
        self.max_radius = max_radius
        
        # Track entity-relation counts
        self.entity_relation_counts = defaultdict(Counter)
        self.global_relation_counts = Counter()
        self.total_edges = 0
        
        # Diversity tracking
        self.domain_type_counts = defaultdict(Counter)
        self.homogeneity_penalties = defaultdict(float)
    
    def check_entity_cap(self, entity: str, relation_id: str) -> bool:
        """Check if adding this relation would exceed entity cap."""
        cap = self.relation_caps.get(relation_id, self.relation_caps.get('*', 5))
        current_count = self.entity_relation_counts[entity][relation_id]
        return current_count < cap
    
    def check_global_cap(self, relation_id: str) -> bool:
        """Check if adding this relation would exceed global soft cap."""
        if self.total_edges == 0:
            return True
        
        current_proportion = self.global_relation_counts[relation_id] / self.total_edges
        return current_proportion < self.global_soft_cap
    
    def check_radius_constraint(self, graph: nx.MultiDiGraph, 
                               seed_entities: Set[str], new_entity: str) -> bool:
        """Check if new entity is within max radius of seed entities."""
        if not seed_entities or new_entity in seed_entities:
            return True
        
        # Convert to undirected for shortest path calculation
        undirected = graph.to_undirected()
        
        for seed in seed_entities:
            if seed in undirected and new_entity in undirected:
                try:
                    distance = nx.shortest_path_length(undirected, seed, new_entity)
                    if distance <= self.max_radius:
                        return True
                except nx.NetworkXNoPath:
                    continue
        
        return False
    
    def calculate_diversity_penalty(self, triplet: KnowledgeTriplet) -> float:
        """Calculate penalty for homogeneous content."""
        penalty = 0.0
        
        # Penalty for same domain-range combinations
        domain_range_key = f"{triplet.domain_guess}-{triplet.relation_id}-{triplet.range_guess}"
        domain_range_count = self.domain_type_counts[triplet.relation_id][domain_range_key]
        
        if domain_range_count > 3:  # After 3 similar triplets, start penalizing
            penalty += min(0.5, (domain_range_count - 3) * 0.1)
        
        # Penalty for overused entities
        head_count = sum(self.entity_relation_counts[triplet.head].values())
        tail_count = sum(self.entity_relation_counts[triplet.tail].values())
        
        if head_count > 10:  # Highly connected entities get penalty
            penalty += min(0.3, (head_count - 10) * 0.02)
        if tail_count > 10:
            penalty += min(0.3, (tail_count - 10) * 0.02)
        
        return penalty
    
    def apply_anti_explosion_filter(self, triplets: List[KnowledgeTriplet],
                                   graph: nx.MultiDiGraph,
                                   seed_entities: Set[str] = None) -> List[KnowledgeTriplet]:
        """Filter triplets based on anti-explosion criteria."""
        filtered = []
        seed_entities = seed_entities or set()
        
        for triplet in triplets:
            # Check entity caps
            if not self.check_entity_cap(triplet.head, triplet.relation_id):
                continue
            if not self.check_entity_cap(triplet.tail, triplet.relation_id):
                continue
            
            # Check global cap
            if not self.check_global_cap(triplet.relation_id):
                continue
            
            # Check radius constraint
            if seed_entities:
                if not (self.check_radius_constraint(graph, seed_entities, triplet.head) or
                       self.check_radius_constraint(graph, seed_entities, triplet.tail)):
                    continue
            
            # Apply diversity penalty
            diversity_penalty = self.calculate_diversity_penalty(triplet)
            triplet.confidence = max(0.0, triplet.confidence - diversity_penalty)
            
            # Only keep if confidence still above threshold after penalty
            if triplet.confidence >= 0.5:
                filtered.append(triplet)
        
        return filtered
    
    def update_counts(self, triplet: KnowledgeTriplet):
        """Update internal counts after adding a triplet."""
        self.entity_relation_counts[triplet.head][triplet.relation_id] += 1
        self.entity_relation_counts[triplet.tail][triplet.relation_id] += 1
        self.global_relation_counts[triplet.relation_id] += 1
        self.total_edges += 1
        
        # Update diversity tracking
        domain_range_key = f"{triplet.domain_guess}-{triplet.relation_id}-{triplet.range_guess}"
        self.domain_type_counts[triplet.relation_id][domain_range_key] += 1
    
    def get_explosion_stats(self) -> Dict:
        """Get statistics about explosion control."""
        stats = {
            'total_edges': self.total_edges,
            'entities_tracked': len(self.entity_relation_counts),
            'relation_distribution': dict(self.global_relation_counts),
            'cap_violations': {},
            'global_cap_violations': [],
            'diversity_stats': {}
        }
        
        # Check for cap violations
        for entity, relation_counts in self.entity_relation_counts.items():
            for relation, count in relation_counts.items():
                cap = self.relation_caps.get(relation, self.relation_caps.get('*', 5))
                if count >= cap * 0.8:  # Warn at 80% of cap
                    stats['cap_violations'][f"{entity}-{relation}"] = f"{count}/{cap}"
        
        # Check global cap violations
        if self.total_edges > 0:
            for relation, count in self.global_relation_counts.items():
                proportion = count / self.total_edges
                if proportion > self.global_soft_cap * 0.8:  # Warn at 80% of soft cap
                    stats['global_cap_violations'].append(
                        f"{relation}: {proportion:.1%} (cap: {self.global_soft_cap:.1%})"
                    )
        
        # Diversity stats
        for relation, domain_range_counts in self.domain_type_counts.items():
            if len(domain_range_counts) > 0:
                stats['diversity_stats'][relation] = {
                    'unique_patterns': len(domain_range_counts),
                    'most_common': domain_range_counts.most_common(3)
                }
        
        return stats

class ClosureAwareValidator:
    """Enhanced validator that considers triadic closure and explosion control."""
    
    def __init__(self, base_validator, triadic_detector: TriadicClosureDetector,
                 explosion_controller: AntiExplosionController):
        self.base_validator = base_validator
        self.triadic_detector = triadic_detector
        self.explosion_controller = explosion_controller
    
    def validate_with_closure_priority(self, triplets: List[KnowledgeTriplet],
                                     graph: nx.MultiDiGraph,
                                     seed_entities: Set[str] = None) -> List[ValidationResult]:
        """Validate triplets with closure priority and explosion control."""
        
        # First, apply anti-explosion filtering
        filtered_triplets = self.explosion_controller.apply_anti_explosion_filter(
            triplets, graph, seed_entities
        )
        
        # Calculate closure priorities
        triplet_priorities = []
        for triplet in filtered_triplets:
            closure_priority = self.triadic_detector.get_closure_priority(triplet)
            triplet_priorities.append((triplet, closure_priority))
        
        # Sort by closure priority (descending) and confidence
        triplet_priorities.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        
        # Validate in priority order
        results = []
        for triplet, priority in triplet_priorities:
            result = self.base_validator.validate_and_normalize(triplet)
            
            # Boost confidence for high-closure triplets
            if result.accept and priority > 0.5:
                if result.normalized_triplet:
                    result.normalized_triplet.confidence = min(1.0, 
                        result.normalized_triplet.confidence + priority * 0.1)
            
            results.append(result)
            
            # Update tracking if accepted
            if result.accept:
                self.base_validator.add_validated_triplet(result.normalized_triplet)
                self.explosion_controller.update_counts(result.normalized_triplet)
                
                if result.inverse_triplet:
                    self.base_validator.add_validated_triplet(result.inverse_triplet)
                    self.explosion_controller.update_counts(result.inverse_triplet)
        
        return results
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics including closure and explosion metrics."""
        base_stats = self.base_validator.get_statistics()
        explosion_stats = self.explosion_controller.get_explosion_stats()
        
        # Triadic closure stats
        triangle_count = self.triadic_detector.count_triangles()
        clustering_coefficient = self.triadic_detector.calculate_clustering_coefficient()
        
        closure_stats = {
            'triangle_count': triangle_count,
            'clustering_coefficient': clustering_coefficient,
            'closure_opportunities': len(self.triadic_detector.closure_opportunities)
        }
        
        return {
            'base_validation': base_stats,
            'explosion_control': explosion_stats,
            'triadic_closure': closure_stats,
            'combined_metrics': {
                'density_score': clustering_coefficient * triangle_count,
                'diversity_entropy': self._calculate_relation_entropy(),
                'explosion_risk': self._calculate_explosion_risk()
            }
        }
    
    def _calculate_relation_entropy(self) -> float:
        """Calculate entropy of relation distribution."""
        total = sum(self.explosion_controller.global_relation_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in self.explosion_controller.global_relation_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_explosion_risk(self) -> float:
        """Calculate risk score for graph explosion."""
        risk = 0.0
        
        # Risk from relation concentration
        total = sum(self.explosion_controller.global_relation_counts.values())
        if total > 0:
            max_proportion = max(self.explosion_controller.global_relation_counts.values()) / total
            risk += min(1.0, max_proportion / self.explosion_controller.global_soft_cap)
        
        # Risk from entity over-connection
        if self.explosion_controller.entity_relation_counts:
            entity_degrees = [sum(counts.values()) for counts in 
                            self.explosion_controller.entity_relation_counts.values()]
            if entity_degrees:
                avg_degree = sum(entity_degrees) / len(entity_degrees)
                max_degree = max(entity_degrees)
                risk += min(1.0, max_degree / (avg_degree * 3))  # Risk if max >> 3*avg
        
        return risk

def create_enhanced_validation_pipeline(base_validator, graph: nx.MultiDiGraph,
                                      relation_caps: Dict[str, int] = None,
                                      global_soft_cap: float = 0.15,
                                      max_radius: int = 3) -> ClosureAwareValidator:
    """Factory function to create enhanced validation pipeline."""
    
    triadic_detector = TriadicClosureDetector(graph)
    explosion_controller = AntiExplosionController(relation_caps, global_soft_cap, max_radius)
    
    return ClosureAwareValidator(base_validator, triadic_detector, explosion_controller)

if __name__ == "__main__":
    # Test the anti-explosion and triadic closure system
    import networkx as nx
    from validation_system import TripletValidator
    
    # Create test graph
    G = nx.MultiDiGraph()
    G.add_edge('A', 'B', key='rel1', relation='rel1')
    G.add_edge('B', 'C', key='rel2', relation='rel2')
    G.add_edge('A', 'D', key='rel1', relation='rel1')
    
    # Test triadic closure detector
    detector = TriadicClosureDetector(G)
    print(f"Triangles: {detector.count_triangles()}")
    print(f"Clustering coefficient: {detector.calculate_clustering_coefficient():.3f}")
    
    # Test anti-explosion controller
    controller = AntiExplosionController()
    
    # Create test triplets
    test_triplets = [
        KnowledgeTriplet('A', 'rel1', 'C', confidence=0.8),  # Would complete triangle
        KnowledgeTriplet('X', 'rel1', 'Y', confidence=0.7),  # New entities
    ]
    
    for triplet in test_triplets:
        closure_priority = detector.get_closure_priority(triplet)
        print(f"Triplet {triplet.to_tuple()}: closure priority = {closure_priority}")
    
    # Test filtering
    filtered = controller.apply_anti_explosion_filter(test_triplets, G)
    print(f"Filtered triplets: {len(filtered)}/{len(test_triplets)}")
    
    # Test enhanced validator
    base_validator = TripletValidator()
    enhanced_validator = create_enhanced_validation_pipeline(base_validator, G)
    
    results = enhanced_validator.validate_with_closure_priority(test_triplets, G)
    print(f"Validation results: {sum(1 for r in results if r.accept)}/{len(results)} accepted")
    
    print("\nComprehensive stats:")
    stats = enhanced_validator.get_comprehensive_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

# Alias for backward compatibility  
TriadicClosureSystem = ClosureAwareValidator
