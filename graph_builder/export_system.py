#!/usr/bin/env python3
"""
Export system for knowledge graphs with dual format support (JSONL + Pickle)
and comprehensive data sheet generation for reproducibility and analysis.
"""

import json
import pickle
import os
import gzip
from typing import Dict, List, Any, Optional
from datetime import datetime
import networkx as nx
import hashlib
from collections import Counter

from .relations_ontology import KnowledgeTriplet, RelationOntology
from .stats_monitoring import GraphQualityMetrics

class GraphExporter:
    """Comprehensive graph export system with multiple format support."""
    
    def __init__(self, output_dir: str = "results/output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize ontology for schema information
        self.ontology = RelationOntology()
        
        # Export metadata
        self.export_metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'exporter_version': '1.0.0',
            'formats': ['pickle', 'jsonl', 'metadata']
        }
    
    def _get_relation_groups(self) -> Dict[str, float]:
        """Extract relation group proportions from the ontology."""
        groups = {}
        all_relations = self.ontology.get_all_relations()
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
    
    def export_complete_graph(self, graph: nx.MultiDiGraph, 
                            construction_stats: Dict,
                            generation_config: Dict,
                            base_filename: str = None) -> Dict[str, str]:
        """Export graph in all formats with comprehensive metadata."""
        
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"knowledge_graph_{timestamp}"
        
        export_paths = {}
        
        # 1. Export as pickle (primary format for Python)
        pickle_path = self._export_pickle(graph, base_filename, construction_stats, generation_config)
        export_paths['pickle'] = pickle_path
        
        # 2. Export as JSONL (interoperable format)
        nodes_path, edges_path = self._export_jsonl(graph, base_filename)
        export_paths['nodes_jsonl'] = nodes_path
        export_paths['edges_jsonl'] = edges_path
        
        # 3. Export metadata and data sheet
        metadata_path = self._export_metadata(graph, base_filename, construction_stats, generation_config)
        export_paths['metadata'] = metadata_path
        
        # 4. Export data sheet (comprehensive documentation)
        datasheet_path = self._export_datasheet(graph, base_filename, construction_stats, generation_config, export_paths)
        export_paths['datasheet'] = datasheet_path
        
        # 5. Export sampling scripts
        sampling_script_path = self._export_sampling_script(base_filename, export_paths)
        export_paths['sampling_script'] = sampling_script_path
        
        print(f"Graph exported successfully to {self.output_dir}/")
        print(f"Files created: {list(export_paths.keys())}")
        
        return export_paths
    
    def _export_pickle(self, graph: nx.MultiDiGraph, base_filename: str,
                      construction_stats: Dict, generation_config: Dict) -> str:
        """Export graph as pickle with embedded metadata."""
        
        # Prepare comprehensive pickle data
        pickle_data = {
            'graph': graph,
            'construction_stats': construction_stats,
            'generation_config': generation_config,
            'export_metadata': self.export_metadata,
            'schema_info': {
                'relations': self.ontology.get_all_relations(),
                'relation_groups': self._get_relation_groups()
            }
        }
        
        pickle_path = os.path.join(self.output_dir, f"{base_filename}.pkl")
        
        # Use compression for large graphs
        if graph.number_of_nodes() > 1000:
            pickle_path = pickle_path + ".gz"
            with gzip.open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return pickle_path
    
    def _export_jsonl(self, graph: nx.MultiDiGraph, base_filename: str) -> tuple[str, str]:
        """Export graph as JSONL files (nodes and edges separately)."""
        
        nodes_path = os.path.join(self.output_dir, f"{base_filename}_nodes.jsonl")
        edges_path = os.path.join(self.output_dir, f"{base_filename}_edges.jsonl")
        
        # Export nodes
        with open(nodes_path, 'w', encoding='utf-8') as f:
            for node in graph.nodes(data=True):
                node_id, node_data = node
                node_record = {
                    'id': node_id,
                    'type': 'node',
                    'attributes': dict(node_data) if node_data else {}
                }
                f.write(json.dumps(node_record, ensure_ascii=False) + '\n')
        
        # Export edges
        with open(edges_path, 'w', encoding='utf-8') as f:
            for head, tail, edge_data in graph.edges(data=True):
                relation_id = edge_data.get('relation', 'Unknown')
                
                edge_record = {
                    'head': head,
                    'tail': tail,
                    'relation_id': relation_id,
                    'type': 'edge',
                    'attributes': {}
                }
                
                # Add all edge attributes
                if edge_data:
                    for attr_key, attr_value in edge_data.items():
                        # Convert non-serializable values
                        if isinstance(attr_value, (str, int, float, bool, type(None))):
                            edge_record['attributes'][attr_key] = attr_value
                        else:
                            edge_record['attributes'][attr_key] = str(attr_value)
                
                f.write(json.dumps(edge_record, ensure_ascii=False) + '\n')
        
        return nodes_path, edges_path
    
    def _export_metadata(self, graph: nx.MultiDiGraph, base_filename: str,
                        construction_stats: Dict, generation_config: Dict) -> str:
        """Export comprehensive metadata as JSON."""
        
        # Calculate quality metrics
        quality_metrics = GraphQualityMetrics(graph)
        metrics = quality_metrics.calculate_all_metrics()
        
        # Prepare metadata
        metadata = {
            'graph_info': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'is_directed': graph.is_directed(),
                'is_multigraph': graph.is_multigraph()
            },
            'quality_metrics': metrics,
            'construction_stats': construction_stats,
            'generation_config': generation_config,
            'export_info': self.export_metadata,
            'schema_info': {
                'total_relations': len(self.ontology.get_all_relations()),
                'relation_groups': list(self._get_relation_groups().keys()),
                'core_relations': len([r for r in self.ontology.get_all_relations().values() if r.get('group') != 'Optional'])
            },
            'file_integrity': self._calculate_file_hashes(graph)
        }
        
        metadata_path = os.path.join(self.output_dir, f"{base_filename}_metadata.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return metadata_path
    
    def _calculate_file_hashes(self, graph: nx.MultiDiGraph) -> Dict[str, str]:
        """Calculate content hashes for integrity verification."""
        
        # Create deterministic string representation
        nodes_str = ''.join(sorted(graph.nodes()))
        edges_str = ''.join(sorted([f"{h}-{data.get('relation','')}-{t}" for h, t, data in graph.edges(data=True)]))
        
        content_hash = hashlib.sha256((nodes_str + edges_str).encode()).hexdigest()
        
        return {
            'content_sha256': content_hash,
            'node_count_hash': hashlib.md5(str(graph.number_of_nodes()).encode()).hexdigest(),
            'edge_count_hash': hashlib.md5(str(graph.number_of_edges()).encode()).hexdigest()
        }
    
    def _export_datasheet(self, graph: nx.MultiDiGraph, base_filename: str,
                         construction_stats: Dict, generation_config: Dict,
                         export_paths: Dict) -> str:
        """Generate comprehensive data sheet documentation."""
        
        quality_metrics = GraphQualityMetrics(graph)
        metrics = quality_metrics.calculate_all_metrics()
        
        # Analyze relation distribution
        relation_analysis = self._analyze_relation_distribution(graph)
        
        # Generate data sheet content
        datasheet_content = self._generate_datasheet_markdown(
            graph, metrics, construction_stats, generation_config, 
            relation_analysis, export_paths
        )
        
        datasheet_path = os.path.join(self.output_dir, f"{base_filename}_datasheet.md")
        
        with open(datasheet_path, 'w', encoding='utf-8') as f:
            f.write(datasheet_content)
        
        return datasheet_path
    
    def _analyze_relation_distribution(self, graph: nx.MultiDiGraph) -> Dict:
        """Analyze the distribution of relations in the graph."""
        
        relation_counts = Counter()
        confidence_by_relation = {}
        group_counts = Counter()
        
        for _, _, data in graph.edges(data=True):
            relation = data.get('relation', 'Unknown')
            relation_counts[relation] += 1
            
            # Track confidence
            confidence = data.get('confidence', 0.0)
            if relation not in confidence_by_relation:
                confidence_by_relation[relation] = []
            confidence_by_relation[relation].append(confidence)
            
            # Track group
            group = data.get('group', 'Unknown')
            group_counts[group] += 1
        
        # Calculate statistics
        analysis = {
            'total_relations': len(relation_counts),
            'relation_counts': dict(relation_counts.most_common()),
            'group_distribution': dict(group_counts),
            'confidence_stats': {}
        }
        
        # Confidence statistics per relation
        for relation, confidences in confidence_by_relation.items():
            if confidences:
                import numpy as np
                analysis['confidence_stats'][relation] = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
        
        return analysis
    
    def _generate_datasheet_markdown(self, graph: nx.MultiDiGraph, metrics: Dict,
                                   construction_stats: Dict, generation_config: Dict,
                                   relation_analysis: Dict, export_paths: Dict) -> str:
        """Generate comprehensive markdown data sheet."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# Knowledge Graph Data Sheet

Generated on: {timestamp}

## Overview

This document provides comprehensive information about a knowledge graph constructed using stratified BFS expansion with controlled relation vocabulary and validation.

## Graph Statistics

### Basic Structure
- **Nodes**: {graph.number_of_nodes():,}
- **Edges**: {graph.number_of_edges():,}
- **Density**: {metrics.get('density', 0):.4f}
- **Average Degree**: {metrics.get('avg_degree', 0):.2f}
- **Maximum Degree**: {metrics.get('max_degree', 0)}

### Quality Metrics
- **Clustering Coefficient**: {metrics.get('clustering_coefficient', 0):.4f}
- **Triangle Count**: {metrics.get('triangle_count', 0):,}
- **Relation Entropy**: {metrics.get('relation_entropy', 0):.3f}
- **Group Coverage**: {metrics.get('group_coverage', 0):.2%}
- **Average Confidence**: {metrics.get('avg_confidence', 0):.3f}

### Connectivity
- **Largest Component Size**: {metrics.get('largest_component_size', 0):,}
- **Component Ratio**: {metrics.get('component_ratio', 0):.2%}
- **Number of Components**: {metrics.get('num_components', 0)}
- **Diameter**: {metrics.get('diameter', 0)}
- **Average Path Length**: {metrics.get('avg_path_length', 0):.2f}

## Relation Analysis

### Relation Distribution
"""
        
        # Add relation distribution table
        content += "\n| Relation | Count | Percentage |\n|----------|--------|------------|\n"
        total_edges = sum(relation_analysis['relation_counts'].values())
        
        for relation, count in list(relation_analysis['relation_counts'].items())[:15]:  # Top 15
            percentage = (count / total_edges) * 100 if total_edges > 0 else 0
            content += f"| {relation} | {count:,} | {percentage:.1f}% |\n"
        
        # Add group distribution
        content += f"""
### Group Distribution
"""
        content += "\n| Group | Count | Percentage |\n|-------|--------|------------|\n"
        for group, count in relation_analysis['group_distribution'].items():
            percentage = (count / total_edges) * 100 if total_edges > 0 else 0
            content += f"| {group} | {count:,} | {percentage:.1f}% |\n"
        
        # Add construction information
        content += f"""
## Construction Details

### Generation Configuration
- **Target Nodes**: {generation_config.get('target_nodes', 'Unknown')}
- **Triplets per Query**: {generation_config.get('triplets_per_query', 'Unknown')}
- **Parallel Frequency**: {generation_config.get('parallel_frequency', 'Unknown')}
- **Model Used**: {generation_config.get('model', 'gpt-4o-mini')}
- **Temperature**: {generation_config.get('temperature', 0.2)}

### Construction Statistics
- **Total API Calls**: {construction_stats.get('api_calls', 0):,}
- **Entities Processed**: {construction_stats.get('entities_processed', 0):,}
- **Relations Processed**: {construction_stats.get('relations_processed', 0):,}
- **Triplets Generated**: {construction_stats.get('triplets_generated', 0):,}
- **Triplets Accepted**: {construction_stats.get('triplets_accepted', 0):,}
- **Acceptance Rate**: {construction_stats.get('acceptance_rate', 0):.1%}
- **Construction Time**: {construction_stats.get('total_runtime', 0):.1f} minutes

### Validation Statistics
- **Triplets Rejected**: {construction_stats.get('triplets_rejected', 0):,}
- **Rejection Rate**: {(1 - construction_stats.get('acceptance_rate', 0)):.1%}

## Data Files

### Available Formats
"""
        
        for format_name, file_path in export_paths.items():
            filename = os.path.basename(file_path)
            content += f"- **{format_name}**: `{filename}`\n"
        
        content += f"""
### File Descriptions
- **Pickle (.pkl)**: Complete graph with all metadata, optimized for Python
- **JSONL (.jsonl)**: Nodes and edges in JSON Lines format for interoperability  
- **Metadata (.json)**: Comprehensive metadata and quality metrics
- **Data Sheet (.md)**: This human-readable documentation
- **Sampling Script (.py)**: Reproducible sampling and loading utilities

## Schema Information

### Relation Ontology
This graph uses a controlled vocabulary of {relation_analysis['total_relations']} relation types organized into the following groups:

"""
        
        # Add relation groups
        relation_groups = self._get_relation_groups()
        for group in relation_groups.keys():
            content += f"- **{group}**: {relation_groups[group]:.0%} target proportion\n"
        
        content += f"""
## Usage Guidelines

### Loading the Graph
```python
import pickle
import networkx as nx

# Load complete graph with metadata
with open('{os.path.basename(export_paths.get("pickle", "graph.pkl"))}', 'rb') as f:
    data = pickle.load(f)
    graph = data['graph']
    metadata = data['export_metadata']
```

### Sampling Subgraphs
Use the provided sampling script to create consistent subgraphs for experiments:

```python
from {os.path.splitext(os.path.basename(export_paths.get('sampling_script', 'sampling.py')))[0]} import sample_subgraph

# Sample 1000-node subgraph around specific entities
subgraph = sample_subgraph(graph, seed_entities=['Beijing', 'Paris'], target_size=1000)
```

## Quality Assurance

### Validation Pipeline
1. **Whitelist Validation**: All relations conform to predefined ontology
2. **Type Checking**: Domain/range compatibility verified
3. **Consistency Checking**: Temporal and logical consistency enforced
4. **Anti-Explosion Controls**: Per-entity and global relation caps applied
5. **Triadic Closure**: Triangle completion prioritized for dense structure

### Known Limitations
- Generated content may contain factual inaccuracies despite validation
- Relation distribution may not perfectly match target proportions
- Some entities may be over-represented due to LLM training biases
- Temporal information may be approximate or outdated

## Citation

If you use this knowledge graph in your research, please cite:

```
@dataset{{knowledge_graph_{datetime.now().strftime('%Y')},
  title={{Stratified Knowledge Graph Construction via Controlled LLM Expansion}},
  author={{Generated via Enhanced BFS Pipeline}},
  year={{{datetime.now().year}}},
  note={{Nodes: {graph.number_of_nodes():,}, Edges: {graph.number_of_edges():,}, Relations: {relation_analysis['total_relations']}}}
}}
```

## Contact

For questions about this dataset, please refer to the generation configuration and construction logs included in the metadata files.

---
*Generated automatically by Knowledge Graph Export System v{self.export_metadata['exporter_version']}*
"""
        
        return content
    
    def _export_sampling_script(self, base_filename: str, export_paths: Dict) -> str:
        """Generate sampling and loading utilities script."""
        
        script_content = f'''#!/usr/bin/env python3
"""
Sampling and loading utilities for {base_filename} knowledge graph.
Generated automatically - provides reproducible subgraph sampling.
"""

import pickle
import json
import random
import networkx as nx
from typing import List, Set, Optional, Dict, Any
from collections import deque
import gzip

def load_complete_graph(pickle_path: str = "{os.path.basename(export_paths.get('pickle', 'graph.pkl'))}") -> Dict[str, Any]:
    """Load the complete graph with all metadata."""
    
    if pickle_path.endswith('.gz'):
        with gzip.open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

def load_graph_only(pickle_path: str = "{os.path.basename(export_paths.get('pickle', 'graph.pkl'))}") -> nx.MultiDiGraph:
    """Load only the graph structure."""
    data = load_complete_graph(pickle_path)
    return data['graph']

def sample_subgraph(graph: nx.MultiDiGraph, 
                   seed_entities: List[str] = None,
                   target_size: int = 1000,
                   method: str = 'bfs',
                   random_seed: int = 42) -> nx.MultiDiGraph:
    """
    Sample a subgraph of specified size.
    
    Args:
        graph: Source graph
        seed_entities: Starting entities (random if None)
        target_size: Target number of nodes
        method: Sampling method ('bfs', 'random_walk', 'random')
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled subgraph
    """
    random.seed(random_seed)
    
    if target_size >= graph.number_of_nodes():
        return graph.copy()
    
    if method == 'random':
        return _random_sample(graph, target_size)
    elif method == 'random_walk':
        return _random_walk_sample(graph, seed_entities, target_size)
    else:  # bfs
        return _bfs_sample(graph, seed_entities, target_size)

def _bfs_sample(graph: nx.MultiDiGraph, seed_entities: Optional[List[str]], target_size: int) -> nx.MultiDiGraph:
    """BFS-based subgraph sampling."""
    
    if not seed_entities:
        seed_entities = random.sample(list(graph.nodes()), min(3, graph.number_of_nodes()))
    
    # Ensure seeds exist in graph
    seed_entities = [e for e in seed_entities if graph.has_node(e)]
    if not seed_entities:
        seed_entities = random.sample(list(graph.nodes()), min(3, graph.number_of_nodes()))
    
    visited = set()
    queue = deque(seed_entities)
    visited.update(seed_entities)
    
    while queue and len(visited) < target_size:
        current = queue.popleft()
        
        # Get neighbors (both directions)
        neighbors = list(graph.neighbors(current)) + list(graph.predecessors(current))
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            if neighbor not in visited and len(visited) < target_size:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return graph.subgraph(visited).copy()

def _random_walk_sample(graph: nx.MultiDiGraph, seed_entities: Optional[List[str]], target_size: int) -> nx.MultiDiGraph:
    """Random walk-based subgraph sampling."""
    
    if not seed_entities:
        seed_entities = [random.choice(list(graph.nodes()))]
    
    visited = set()
    
    for seed in seed_entities:
        if len(visited) >= target_size:
            break
            
        current = seed
        walk_length = target_size // len(seed_entities)
        
        for _ in range(walk_length):
            if len(visited) >= target_size:
                break
                
            visited.add(current)
            
            # Get neighbors
            neighbors = list(graph.neighbors(current)) + list(graph.predecessors(current))
            if neighbors:
                current = random.choice(neighbors)
            else:
                break
    
    return graph.subgraph(visited).copy()

def _random_sample(graph: nx.MultiDiGraph, target_size: int) -> nx.MultiDiGraph:
    """Uniform random subgraph sampling."""
    sampled_nodes = random.sample(list(graph.nodes()), target_size)
    return graph.subgraph(sampled_nodes).copy()

def get_relation_statistics(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Get comprehensive relation statistics."""
    
    relation_counts = {{}}
    confidence_stats = {{}}
    
    for _, _, data in graph.edges(data=True):
        relation = data.get('relation', 'Unknown')
        confidence = data.get('confidence', 0.0)
        
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        if relation not in confidence_stats:
            confidence_stats[relation] = []
        confidence_stats[relation].append(confidence)
    
    # Calculate statistics
    stats = {{
        'total_relations': len(relation_counts),
        'relation_counts': relation_counts,
        'confidence_means': {{}}
    }}
    
    for relation, confidences in confidence_stats.items():
        if confidences:
            stats['confidence_means'][relation] = sum(confidences) / len(confidences)
    
    return stats

def export_subgraph(subgraph: nx.MultiDiGraph, filename: str):
    """Export subgraph in multiple formats."""
    
    # Pickle format
    with open(f"{{filename}}.pkl", 'wb') as f:
        pickle.dump(subgraph, f)
    
    # JSONL format
    with open(f"{{filename}}_edges.jsonl", 'w') as f:
        for head, tail, key, data in subgraph.edges(data=True, keys=True):
            edge_record = {{
                'head': head,
                'tail': tail,
                'relation_id': key,
                'attributes': dict(data)
            }}
            f.write(json.dumps(edge_record) + '\\n')

if __name__ == "__main__":
    # Example usage
    print("Loading graph...")
    graph = load_graph_only()
    print(f"Loaded graph with {{graph.number_of_nodes()}} nodes and {{graph.number_of_edges()}} edges")
    
    print("\\nSampling subgraph...")
    subgraph = sample_subgraph(graph, target_size=500, method='bfs')
    print(f"Sampled subgraph with {{subgraph.number_of_nodes()}} nodes and {{subgraph.number_of_edges()}} edges")
    
    print("\\nRelation statistics:")
    stats = get_relation_statistics(subgraph)
    for relation, count in list(stats['relation_counts'].items())[:10]:
        conf = stats['confidence_means'].get(relation, 0)
        print(f"  {{relation}}: {{count}} edges (avg conf: {{conf:.2f}})")
'''
        
        script_path = os.path.join(self.output_dir, f"{base_filename}_sampling.py")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path

def create_exporter(output_dir: str = "results/output") -> GraphExporter:
    """Factory function to create graph exporter."""
    return GraphExporter(output_dir)

if __name__ == "__main__":
    # Test the export system
    import networkx as nx
    
    # Create test graph
    G = nx.MultiDiGraph()
    G.add_edge('Beijing', 'China', key='CapitalOf', 
              relation='CapitalOf', confidence=0.95, group='Spatial')
    G.add_edge('Paris', 'France', key='CapitalOf',
              relation='CapitalOf', confidence=0.98, group='Spatial')
    G.add_edge('Einstein', 'Physicist', key='Occupation',
              relation='Occupation', confidence=0.99, group='Social')
    
    # Test export
    exporter = create_exporter("results/test_output")
    
    construction_stats = {
        'api_calls': 150,
        'entities_processed': 50,
        'triplets_generated': 200,
        'triplets_accepted': 180,
        'acceptance_rate': 0.9,
        'total_runtime': 15.5
    }
    
    generation_config = {
        'target_nodes': 3000,
        'triplets_per_query': 8,
        'parallel_frequency': 5,
        'model': 'gpt-4o-mini',
        'temperature': 0.2
    }
    
    export_paths = exporter.export_complete_graph(G, construction_stats, generation_config, "test_graph")
    
    print("Export completed successfully!")
    print("Files created:")
    for format_name, path in export_paths.items():
        print(f"  {format_name}: {path}")

# Alias for backward compatibility
ExportSystem = GraphExporter
