#!/usr/bin/env python3
"""
Function-like Graph Builder
High-precision knowledge graph construction using 36 function-like relations.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import jsonschema

from .qa_atomic_ontology import QAAtomicOntology, KnowledgeTriplet
from .llm_calls_enhanced import LLMInterfaceEnhanced, TRIPLET_SCHEMA_v0_3, load_api_key
from .prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3, create_user_prompt_v0_3


class QAGraphBuilder:
    """
    Function-like Graph Builder for constructing high-precision knowledge graphs.
    Uses 36 function-like relations designed for unique, answerable facts.
    """
    
    def __init__(self, api_key_path: str = 'keys/openai.txt', cache_dir: str = None):
        """Initialize the Function-like Graph Builder."""
        self.ontology = QAAtomicOntology()
        self.llm_interface = LLMInterfaceEnhanced(api_key_path, cache_dir)
        
        # Override the ontology in LLM interface
        self.llm_interface.ontology = self.ontology
        
        # Initialize API
        if not load_api_key(api_key_path):
            raise RuntimeError(f"Failed to initialize LLM API with key from {api_key_path}")
        
        print(f"✅ Function-like Graph Builder initialized")
        self.ontology.print_summary()
    
    def generate_from_seeds(self, seeds: List[str], budget: int = 50, language: str = "en") -> List[Dict[str, Any]]:
        """
        Generate function-like triplets from seed entities.
        
        Args:
            seeds: List of seed entity names
            budget: Maximum number of triplets to generate
            language: Language for surface text ("en" or "zh")
            
        Returns:
            List of validated function-like triplet dictionaries
        """
        if not seeds:
            raise ValueError("At least one seed entity is required")
        
        print(f"\n🌱 Generating function-like knowledge graph from {len(seeds)} seeds")
        print(f"🎯 Target: {budget} triplets, Language: {language}")
        print(f"📝 Seeds: {', '.join(seeds)}")
        
        # Create custom user prompt using updated v0.3 system
        user_prompt = create_user_prompt_v0_3(seeds, self.ontology, budget, language)
        
        # Call LLM with v0.3 system prompt
        content = self.llm_interface._call_llm_with_cache(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            temperature=0.2,
            max_tokens=4000
        )
        
        if not content:
            print("❌ No response from LLM")
            return []
        
        # Parse and validate JSONL response
        triplets = self._parse_function_triplets(content)
        print(f"✅ Generated {len(triplets)} valid function-like triplets")
        
        return triplets
    
    def _parse_function_triplets(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSONL response from function-like relation generation."""
        if not content:
            return []
        
        triplets = []
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                triplet_data = json.loads(line)
                
                # Validate against schema
                jsonschema.validate(triplet_data, TRIPLET_SCHEMA_v0_3)
                
                # Function-like relation validation
                relation_id = triplet_data['relation_id']
                if not self.ontology.is_valid_relation(relation_id):
                    print(f"⚠️ Line {line_num}: Unknown function-like relation '{relation_id}', skipping")
                    continue
                
                # Check required qualifiers
                required_qualifiers = self.ontology.get_required_qualifiers(relation_id)
                provided_qualifiers = list(triplet_data.get('qualifiers', {}).keys())
                
                for req_qual in required_qualifiers:
                    if req_qual not in provided_qualifiers:
                        print(f"⚠️ Line {line_num}: Missing required qualifier '{req_qual}' for {relation_id}")
                        continue
                
                # Mark as function-like eligible (all relations in this ontology are)
                triplet_data['qa_eligible'] = True
                
                triplets.append(triplet_data)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_num}: JSON decode error: {e}")
                continue
            except jsonschema.ValidationError as e:
                print(f"⚠️ Line {line_num}: Schema validation error: {e.message}")
                continue
            except Exception as e:
                print(f"⚠️ Line {line_num}: Unexpected error: {e}")
                continue
        
        return triplets
    
    def export_triplets(self, triplets: List[Dict[str, Any]], output_path: str, format: str = "json") -> str:
        """Export function-like triplets to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(triplets, f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for triplet in triplets:
                    f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📁 Exported {len(triplets)} function-like triplets to {output_path}")
        return str(output_path)
    
    def build_graph_to_size(self, seeds: List[str], target_nodes: int = 50, 
                           language: str = "en") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build a function-like graph to a target number of nodes.
        
        Args:
            seeds: Initial seed entities
            target_nodes: Target number of unique nodes in the graph
            language: Language for surface text
            
        Returns:
            Tuple of (triplets, graph_stats)
        """
        print(f"\n🎯 Building function-like graph to {target_nodes} nodes")
        
        all_triplets = []
        unique_nodes = set(seeds)
        iteration = 0
        max_iterations = 10
        
        current_seeds = seeds.copy()
        
        while len(unique_nodes) < target_nodes and iteration < max_iterations:
            iteration += 1
            remaining_nodes = target_nodes - len(unique_nodes)
            budget = min(remaining_nodes * 2, 30)  # Generate more triplets than nodes needed
            
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current nodes: {len(unique_nodes)}, Target: {target_nodes}")
            print(f"Seeds for this iteration: {current_seeds[:5]}{'...' if len(current_seeds) > 5 else ''}")
            
            # Generate triplets from current seeds
            new_triplets = self.generate_from_seeds(current_seeds, budget=budget, language=language)
            
            if not new_triplets:
                print("⚠️ No new triplets generated, stopping")
                break
            
            # Add new triplets and collect new nodes
            iteration_new_nodes = set()
            for triplet in new_triplets:
                all_triplets.append(triplet)
                head = triplet['head']
                tail = triplet['tail']
                
                if head not in unique_nodes:
                    iteration_new_nodes.add(head)
                if tail not in unique_nodes:
                    iteration_new_nodes.add(tail)
                
                unique_nodes.add(head)
                unique_nodes.add(tail)
            
            print(f"✅ Added {len(new_triplets)} triplets, {len(iteration_new_nodes)} new nodes")
            
            # Prepare seeds for next iteration (sample from new nodes)
            if iteration_new_nodes:
                current_seeds = list(iteration_new_nodes)[:10]  # Take up to 10 new nodes as seeds
            else:
                # If no new nodes, try expanding from existing nodes
                current_seeds = list(unique_nodes)[-10:]
            
            if len(unique_nodes) >= target_nodes:
                break
        
        # Calculate graph statistics
        stats = self._calculate_graph_stats(all_triplets, unique_nodes)
        
        print(f"\n🎉 Graph construction completed!")
        print(f"📊 Final stats: {len(unique_nodes)} nodes, {len(all_triplets)} triplets")
        print(f"🔄 Iterations: {iteration}")
        
        return all_triplets, stats
    
    def _calculate_graph_stats(self, triplets: List[Dict[str, Any]], nodes: set) -> Dict[str, Any]:
        """Calculate statistics for the generated graph."""
        relation_counts = {}
        group_counts = {}
        function_count = 0
        
        for triplet in triplets:
            rel_id = triplet['relation_id']
            group = triplet['group']
            
            relation_counts[rel_id] = relation_counts.get(rel_id, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1
            
            if triplet.get('qa_eligible', False):
                function_count += 1
        
        return {
            'total_nodes': len(nodes),
            'total_triplets': len(triplets),
            'function_triplets': function_count,
            'function_percentage': (function_count / len(triplets)) * 100 if triplets else 0,
            'relation_distribution': relation_counts,
            'group_distribution': group_counts,
            'average_node_degree': (len(triplets) * 2) / len(nodes) if nodes else 0
        }


def quick_function_build(seeds: List[str], target_nodes: int = 50, language: str = "en", 
                        output_file: str = None) -> List[Dict[str, Any]]:
    """
    Quick function to build a function-like graph.
    
    Args:
        seeds: List of seed entities
        target_nodes: Target number of nodes
        language: Language for surface text
        output_file: Optional output file path
        
    Returns:
        List of function-like triplets
    """
    builder = QAGraphBuilder()
    triplets, stats = builder.build_graph_to_size(seeds, target_nodes, language)
    
    if output_file:
        builder.export_triplets(triplets, output_file, format="json")
    
    print(f"\n📈 Graph Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} categories")
        else:
            print(f"  {key}: {value}")
    
    return triplets


if __name__ == "__main__":
    # Demo: Build a function-like graph to 50 nodes
    try:
        print("🚀 Function-like Graph Builder Demo")
        
        seeds = ["北京", "苹果公司", "爱因斯坦", "哈姆雷特"]
        triplets = quick_function_build(
            seeds=seeds,
            target_nodes=50,
            language="zh",
            output_file="function_demo.json"
        )
        
        print(f"\n🎯 Demo completed! Generated {len(triplets)} function-like triplets.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure to:")
        print("1. Place your OpenAI API key in 'keys/openai.txt'")
        print("2. Run from the project root directory")
