#!/usr/bin/env python3
"""
Real-time statistics monitoring and early stopping system for knowledge graph construction.
Tracks quality metrics, diversity, and implements intelligent stopping criteria.
"""

import math
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, deque
from datetime import datetime, timedelta
import networkx as nx
import numpy as np

from .relations_ontology import RelationOntology
from .anti_explosion_triadic import TriadicClosureDetector

# Helper functions for backward compatibility
def get_relations_by_group(group: str) -> List[str]:
    """Get relations by group from the global ontology."""
    ontology = RelationOntology()
    relations = []
    for rel_id, rel_info in ontology.get_all_relations().items():
        if rel_info.get('group') == group:
            relations.append(rel_id)
    return relations

def get_relation_groups() -> Dict[str, float]:
    """Get relation groups from the global ontology."""
    ontology = RelationOntology()
    groups = {}
    all_relations = ontology.get_all_relations()
    total_relations = len(all_relations)
    
    if total_relations == 0:
        return groups
        
    group_counts = Counter()
    for relation_info in all_relations.values():
        group = relation_info.get('group', 'Unknown')
        group_counts[group] += 1
    
    for group, count in group_counts.items():
        groups[group] = count / total_relations
        
    return groups

# Backward compatibility
RELATION_GROUPS = get_relation_groups()

class GraphQualityMetrics:
    """Calculate various quality metrics for knowledge graphs."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.triadic_detector = TriadicClosureDetector(graph)
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive set of graph quality metrics."""
        metrics = {}
        
        # Basic structure metrics
        metrics.update(self._calculate_basic_metrics())
        
        # Diversity metrics
        metrics.update(self._calculate_diversity_metrics())
        
        # Quality metrics
        metrics.update(self._calculate_quality_metrics())
        
        # Connectivity metrics
        metrics.update(self._calculate_connectivity_metrics())
        
        return metrics
    
    def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic structural metrics."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0,
            'avg_degree': (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
        }
        
        # Degree distribution
        if num_nodes > 0:
            degrees = [self.graph.degree(n) + self.graph.in_degree(n) for n in self.graph.nodes()]
            metrics['max_degree'] = max(degrees) if degrees else 0
            metrics['degree_std'] = np.std(degrees) if degrees else 0.0
        else:
            metrics['max_degree'] = 0
            metrics['degree_std'] = 0.0
        
        return metrics
    
    def _calculate_diversity_metrics(self) -> Dict[str, float]:
        """Calculate diversity and distribution metrics."""
        metrics = {}
        
        # Relation diversity
        relation_counts = Counter()
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', 'Unknown')
            relation_counts[relation] += 1
        
        # Calculate relation entropy
        total_relations = sum(relation_counts.values())
        if total_relations > 0:
            relation_entropy = 0.0
            for count in relation_counts.values():
                if count > 0:
                    p = count / total_relations
                    relation_entropy -= p * math.log2(p)
            metrics['relation_entropy'] = relation_entropy
            
            # Max theoretical entropy for current number of unique relations
            num_unique_relations = len(relation_counts)
            if num_unique_relations > 1:
                max_entropy = math.log2(num_unique_relations)
                metrics['relation_entropy_normalized'] = relation_entropy / max_entropy
            else:
                metrics['relation_entropy_normalized'] = 0.0
        else:
            metrics['relation_entropy'] = 0.0
            metrics['relation_entropy_normalized'] = 0.0
        
        # Group coverage
        group_coverage = self._calculate_group_coverage(relation_counts)
        metrics.update(group_coverage)
        
        return metrics
    
    def _calculate_group_coverage(self, relation_counts: Counter) -> Dict[str, float]:
        """Calculate coverage of different relation groups."""
        group_stats = {}
        
        for group in RELATION_GROUPS.keys():
            group_relations = get_relations_by_group(group)
            group_count = sum(relation_counts[rel] for rel in group_relations if rel in relation_counts)
            group_stats[f'group_{group.lower()}_count'] = group_count
        
        # Calculate coverage (how many groups have at least one relation)
        total_relations = sum(relation_counts.values())
        if total_relations > 0:
            active_groups = sum(1 for group in RELATION_GROUPS.keys() 
                              if group_stats[f'group_{group.lower()}_count'] > 0)
            group_stats['group_coverage'] = active_groups / len(RELATION_GROUPS)
            
            # Calculate group balance (how evenly distributed across groups)
            group_proportions = []
            for group in RELATION_GROUPS.keys():
                count = group_stats[f'group_{group.lower()}_count']
                if count > 0:
                    group_proportions.append(count / total_relations)
            
            if group_proportions:
                # Calculate coefficient of variation
                mean_prop = np.mean(group_proportions)
                std_prop = np.std(group_proportions)
                group_stats['group_balance'] = 1 - (std_prop / mean_prop) if mean_prop > 0 else 0
            else:
                group_stats['group_balance'] = 0.0
        else:
            group_stats['group_coverage'] = 0.0
            group_stats['group_balance'] = 0.0
        
        return group_stats
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality-related metrics."""
        metrics = {}
        
        # Confidence statistics
        confidences = []
        for _, _, data in self.graph.edges(data=True):
            conf = data.get('confidence', 0.0)
            if isinstance(conf, (int, float)):
                confidences.append(conf)
        
        if confidences:
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['min_confidence'] = np.min(confidences)
            metrics['confidence_std'] = np.std(confidences)
        else:
            metrics['avg_confidence'] = 0.0
            metrics['min_confidence'] = 0.0
            metrics['confidence_std'] = 0.0
        
        # Triadic closure metrics
        self.triadic_detector.update_graph(self.graph)
        metrics['triangle_count'] = self.triadic_detector.count_triangles()
        metrics['clustering_coefficient'] = self.triadic_detector.calculate_clustering_coefficient()
        
        return metrics
    
    def _calculate_connectivity_metrics(self) -> Dict[str, float]:
        """Calculate connectivity and reachability metrics."""
        metrics = {}
        
        if self.graph.number_of_nodes() == 0:
            return {
                'largest_component_size': 0,
                'component_ratio': 0.0,
                'diameter': 0,
                'avg_path_length': 0.0
            }
        
        # Convert to undirected for connectivity analysis
        undirected = self.graph.to_undirected()
        
        # Component analysis
        components = list(nx.connected_components(undirected))
        if components:
            largest_component_size = len(max(components, key=len))
            metrics['largest_component_size'] = largest_component_size
            metrics['component_ratio'] = largest_component_size / self.graph.number_of_nodes()
            metrics['num_components'] = len(components)
        else:
            metrics['largest_component_size'] = 0
            metrics['component_ratio'] = 0.0
            metrics['num_components'] = 0
        
        # Path metrics (on largest component)
        if components and len(max(components, key=len)) > 1:
            largest_component = undirected.subgraph(max(components, key=len))
            try:
                metrics['diameter'] = nx.diameter(largest_component)
                metrics['avg_path_length'] = nx.average_shortest_path_length(largest_component)
            except:
                metrics['diameter'] = 0
                metrics['avg_path_length'] = 0.0
        else:
            metrics['diameter'] = 0
            metrics['avg_path_length'] = 0.0
        
        return metrics

class RealTimeMonitor:
    """Real-time monitoring system with history tracking and trend analysis."""
    
    def __init__(self, graph: nx.MultiDiGraph = None, ontology: RelationOntology = None, 
                 early_stop_config: Dict = None, group_quotas: Dict = None,
                 window_size: int = 50, save_interval: int = 100):
        self.graph = graph
        self.ontology = ontology or RelationOntology()
        self.early_stop_config = early_stop_config or {}
        self.group_quotas = group_quotas or {}
        self.window_size = window_size
        self.save_interval = save_interval
        
        # Metrics history
        self.metrics_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Performance tracking
        self.api_calls = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        
        # Progress tracking
        self.step_count = 0
        self.entities_processed = 0
        self.relations_processed = 0
        
        # Quality trends
        self.quality_trends = {}
        
    def record_metrics(self, graph: nx.MultiDiGraph, additional_stats: Dict = None):
        """Record current metrics snapshot."""
        quality_calculator = GraphQualityMetrics(graph)
        metrics = quality_calculator.calculate_all_metrics()
        
        # Add additional stats if provided
        if additional_stats:
            metrics.update(additional_stats)
        
        # Add performance metrics
        current_time = time.time()
        elapsed = current_time - self.start_time
        metrics.update({
            'elapsed_time': elapsed,
            'api_calls': self.api_calls,
            'step_count': self.step_count,
            'entities_processed': self.entities_processed,
            'relations_processed': self.relations_processed,
            'api_calls_per_minute': (self.api_calls / elapsed) * 60 if elapsed > 0 else 0,
            'nodes_per_minute': (metrics['num_nodes'] / elapsed) * 60 if elapsed > 0 else 0
        })
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.timestamps.append(current_time)
        
        # Update trends
        self._update_trends()
        
        self.step_count += 1
    
    def _update_trends(self):
        """Update trend analysis for key metrics."""
        if len(self.metrics_history) < 2:
            return
        
        # Calculate trends for key metrics
        key_metrics = [
            'num_nodes', 'num_edges', 'relation_entropy', 'clustering_coefficient',
            'avg_confidence', 'group_coverage', 'triangle_count'
        ]
        
        for metric in key_metrics:
            values = [m.get(metric, 0) for m in list(self.metrics_history)[-10:]]  # Last 10 values
            if len(values) >= 2:
                # Simple linear trend
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                self.quality_trends[f'{metric}_trend'] = slope
    
    def get_current_metrics(self) -> Dict:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1].copy()
        return {}
    
    def get_trend_analysis(self) -> Dict:
        """Get trend analysis for key metrics."""
        return self.quality_trends.copy()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        current_metrics = self.get_current_metrics()
        
        return {
            'total_runtime': elapsed,
            'api_calls': self.api_calls,
            'steps_completed': self.step_count,
            'entities_processed': self.entities_processed,
            'relations_processed': self.relations_processed,
            'current_nodes': current_metrics.get('num_nodes', 0),
            'current_edges': current_metrics.get('num_edges', 0),
            'avg_api_calls_per_minute': (self.api_calls / elapsed) * 60 if elapsed > 0 else 0,
            'avg_nodes_per_minute': (current_metrics.get('num_nodes', 0) / elapsed) * 60 if elapsed > 0 else 0
        }
    
    def increment_api_calls(self, count: int = 1):
        """Increment API call counter."""
        self.api_calls += count
    
    def increment_entities_processed(self, count: int = 1):
        """Increment entities processed counter."""
        self.entities_processed += count
    
    def increment_relations_processed(self, count: int = 1):
        """Increment relations processed counter."""
        self.relations_processed += count

class EarlyStoppingCriteria:
    """Intelligent early stopping based on multiple criteria."""
    
    def __init__(self, target_nodes: int = 3000, 
                 min_clustering: float = 0.18,
                 min_triangles: int = 8000,
                 min_entropy: float = 2.8,
                 min_group_coverage: float = 0.8,
                 patience: int = 20,
                 min_improvement: float = 0.01):
        
        # Target criteria
        self.target_nodes = target_nodes
        self.min_clustering = min_clustering
        self.min_triangles = min_triangles
        self.min_entropy = min_entropy
        self.min_group_coverage = min_group_coverage
        
        # Patience-based stopping
        self.patience = patience
        self.min_improvement = min_improvement
        self.no_improvement_count = 0
        self.best_score = -float('inf')
        
        # Quality thresholds
        self.quality_checks = []
    
    def calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate composite quality score."""
        score = 0.0
        
        # Node count progress (0-1 normalized)
        node_progress = min(1.0, metrics.get('num_nodes', 0) / self.target_nodes)
        score += node_progress * 0.3
        
        # Clustering coefficient
        clustering = metrics.get('clustering_coefficient', 0)
        clustering_score = min(1.0, clustering / self.min_clustering)
        score += clustering_score * 0.2
        
        # Relation diversity
        entropy = metrics.get('relation_entropy', 0)
        entropy_score = min(1.0, entropy / self.min_entropy)
        score += entropy_score * 0.2
        
        # Group coverage
        coverage = metrics.get('group_coverage', 0)
        coverage_score = min(1.0, coverage / self.min_group_coverage)
        score += coverage_score * 0.15
        
        # Triangle count
        triangles = metrics.get('triangle_count', 0)
        triangle_score = min(1.0, triangles / self.min_triangles)
        score += triangle_score * 0.15
        
        return score
    
    def should_stop(self, metrics: Dict, trends: Dict) -> Tuple[bool, str]:
        """Determine if construction should stop early."""
        
        # Check hard targets (any two satisfied = stop)
        satisfied_criteria = []
        
        if metrics.get('num_nodes', 0) >= self.target_nodes:
            satisfied_criteria.append("target_nodes")
        
        if (metrics.get('clustering_coefficient', 0) >= self.min_clustering and 
            metrics.get('triangle_count', 0) >= self.min_triangles):
            satisfied_criteria.append("clustering_triangles")
        
        if metrics.get('relation_entropy', 0) >= self.min_entropy:
            satisfied_criteria.append("entropy")
        
        if metrics.get('group_coverage', 0) >= self.min_group_coverage:
            satisfied_criteria.append("group_coverage")
        
        # Stop if any two criteria are satisfied
        if len(satisfied_criteria) >= 2:
            return True, f"Multiple criteria satisfied: {', '.join(satisfied_criteria)}"
        
        # Check for lack of improvement (patience-based stopping)
        current_score = self.calculate_composite_score(metrics)
        
        if current_score > self.best_score + self.min_improvement:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        if self.no_improvement_count >= self.patience:
            return True, f"No improvement for {self.patience} steps (score: {current_score:.3f})"
        
        # Check for negative trends
        negative_trends = []
        if trends.get('relation_entropy_trend', 0) < -0.01:
            negative_trends.append("entropy_declining")
        if trends.get('clustering_coefficient_trend', 0) < -0.001:
            negative_trends.append("clustering_declining")
        
        if len(negative_trends) >= 2:
            return True, f"Multiple negative trends: {', '.join(negative_trends)}"
        
        return False, f"Continue (score: {current_score:.3f}, best: {self.best_score:.3f})"
    
    def get_progress_report(self, metrics: Dict) -> Dict:
        """Get detailed progress report against criteria."""
        report = {
            'composite_score': self.calculate_composite_score(metrics),
            'criteria_progress': {}
        }
        
        # Individual criteria progress
        criteria = [
            ('nodes', metrics.get('num_nodes', 0), self.target_nodes),
            ('clustering', metrics.get('clustering_coefficient', 0), self.min_clustering),
            ('triangles', metrics.get('triangle_count', 0), self.min_triangles),
            ('entropy', metrics.get('relation_entropy', 0), self.min_entropy),
            ('coverage', metrics.get('group_coverage', 0), self.min_group_coverage)
        ]
        
        for name, current, target in criteria:
            progress = min(1.0, current / target) if target > 0 else 0.0
            report['criteria_progress'][name] = {
                'current': current,
                'target': target,
                'progress': progress,
                'satisfied': current >= target
            }
        
        return report

def create_monitoring_system(target_nodes: int = 3000, window_size: int = 50, 
                           early_stop_config: Dict = None) -> Tuple[RealTimeMonitor, EarlyStoppingCriteria]:
    """Factory function to create complete monitoring system."""
    monitor = RealTimeMonitor(window_size=window_size)
    
    # Create early stopping with custom config if provided
    if early_stop_config:
        early_stopping = EarlyStoppingCriteria(
            target_nodes=early_stop_config.get('min_nodes', target_nodes),
            min_clustering=early_stop_config.get('min_clustering', 0.18),
            min_triangles=early_stop_config.get('min_triangles', 8000),
            min_entropy=early_stop_config.get('min_entropy', 2.8),
            min_group_coverage=early_stop_config.get('min_group_coverage', 0.8),
            patience=early_stop_config.get('patience', 20),
        )
    else:
        early_stopping = EarlyStoppingCriteria(target_nodes=target_nodes)
    
    return monitor, early_stopping

if __name__ == "__main__":
    # Test the monitoring system
    import networkx as nx
    
    # Create test graph
    G = nx.MultiDiGraph()
    G.add_edge('A', 'B', key='rel1', relation='rel1', confidence=0.9)
    G.add_edge('B', 'C', key='rel2', relation='rel2', confidence=0.8)
    G.add_edge('A', 'C', key='rel3', relation='rel3', confidence=0.7)
    
    # Test quality metrics
    quality_calc = GraphQualityMetrics(G)
    metrics = quality_calc.calculate_all_metrics()
    print("Quality metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test monitoring
    monitor, early_stopping = create_monitoring_system(target_nodes=100)
    
    # Simulate monitoring steps
    for step in range(5):
        # Add some edges to simulate growth
        G.add_edge(f'Node{step}', f'Node{step+10}', 
                  key=f'rel{step}', relation=f'rel{step}', confidence=0.8)
        
        monitor.record_metrics(G)
        monitor.increment_api_calls(2)
        monitor.increment_entities_processed(1)
        
        current_metrics = monitor.get_current_metrics()
        trends = monitor.get_trend_analysis()
        
        should_stop, reason = early_stopping.should_stop(current_metrics, trends)
        
        print(f"\nStep {step}:")
        print(f"  Nodes: {current_metrics.get('num_nodes', 0)}")
        print(f"  Edges: {current_metrics.get('num_edges', 0)}")
        print(f"  Should stop: {should_stop} ({reason})")
        
        if should_stop:
            break
    
    print(f"\nFinal performance summary:")
    perf_summary = monitor.get_performance_summary()
    for key, value in perf_summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nProgress report:")
    progress_report = early_stopping.get_progress_report(monitor.get_current_metrics())
    for key, value in progress_report.items():
        print(f"  {key}: {value}")
