#!/usr/bin/env python3
"""
Graph Builder v0.3 - Python Wrapper
Clean, high-level interface for knowledge graph construction using v0.3 prompt system.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import jsonschema

from .relations_ontology import RelationOntology, KnowledgeTriplet
from .llm_calls_enhanced import LLMInterfaceEnhanced, TRIPLET_SCHEMA_v0_3


class GraphBuilderV03:
    """
    High-level interface for knowledge graph construction using v0.3 prompt system.
    
    Features:
    - Unified prompt templates with QA-Atomic support
    - Rich metadata with qualifiers
    - JSON Schema validation
    - Backward compatibility with legacy systems
    """
    
    def __init__(self, api_key_path: str = 'keys/openai.txt', cache_dir: str = None, 
                 include_optional_relations: bool = False):
        """
        Initialize the Graph Builder.
        
        Args:
            api_key_path: Path to OpenAI API key file
            cache_dir: Directory for response caching
            include_optional_relations: Whether to include optional relations from ontology
        """
        self.ontology = RelationOntology()
        self.llm_interface = LLMInterfaceEnhanced(api_key_path, cache_dir, self.ontology)
        self.include_optional = include_optional_relations
        
        # Initialize API
        if not self.llm_interface.initialize_api(api_key_path):
            raise RuntimeError(f"Failed to initialize LLM API with key from {api_key_path}")
        
        print(f"✅ Graph Builder v0.3 initialized")
        print(f"📚 Ontology loaded: {len(self.ontology.get_all_relations())} relations")
        if not self.include_optional:
            optional_count = sum(1 for r in self.ontology.get_all_relations().values() 
                               if r.get('group') == 'Optional')
            print(f"⚪ Optional relations excluded: {optional_count}")
    
    def generate_from_seeds(self, seeds: List[str], budget: int = 40, language: str = "en") -> List[Dict[str, Any]]:
        """
        Generate triplets from seed entities using v0.3 prompt system.
        
        Args:
            seeds: List of seed entity names
            budget: Maximum number of triplets to generate
            language: Language for surface text ("en" or "zh")
            
        Returns:
            List of validated triplet dictionaries
        """
        if not seeds:
            raise ValueError("At least one seed entity is required")
        
        print(f"\n🌱 Generating knowledge graph from {len(seeds)} seeds")
        print(f"🎯 Target: {budget} triplets, Language: {language}")
        print(f"📝 Seeds: {', '.join(seeds)}")
        
        triplets = self.llm_interface.generate_triplets_from_seeds(
            seeds=seeds,
            budget=budget,
            language=language,
            include_optional=self.include_optional
        )
        
        print(f"✅ Generated {len(triplets)} valid triplets")
        return triplets
    
    def validate_triplet(self, triplet_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a single triplet against v0.3 schema and ontology.
        
        Args:
            triplet_data: Triplet dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Schema validation
            jsonschema.validate(triplet_data, TRIPLET_SCHEMA_v0_3)
            
            # Ontology validation
            relation_id = triplet_data['relation_id']
            if not self.ontology.is_valid_relation(relation_id):
                return False, f"Unknown relation: {relation_id}"
            
            # Check domain/range consistency if available in ontology
            relation_info = self.ontology.get_relation_info(relation_id)
            if relation_info:
                expected_group = relation_info.get('group', 'Unknown')
                if triplet_data['group'] != expected_group:
                    return False, f"Group mismatch: expected {expected_group}, got {triplet_data['group']}"
            
            return True, ""
            
        except jsonschema.ValidationError as e:
            return False, f"Schema validation error: {e.message}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def export_triplets(self, triplets: List[Dict[str, Any]], output_path: str, 
                       format: str = "jsonl") -> str:
        """
        Export triplets to file in specified format.
        
        Args:
            triplets: List of triplet dictionaries
            output_path: Output file path
            format: Export format ("jsonl", "json", "csv")
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for triplet in triplets:
                    f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        
        elif format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(triplets, f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "csv":
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                if not triplets:
                    return str(output_path)
                
                fieldnames = triplets[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for triplet in triplets:
                    # Flatten complex fields for CSV
                    row = triplet.copy()
                    row['qualifiers'] = json.dumps(row['qualifiers'])
                    writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📁 Exported {len(triplets)} triplets to {output_path}")
        return str(output_path)
    
    def convert_to_legacy_format(self, triplets: List[Dict[str, Any]]) -> List[KnowledgeTriplet]:
        """
        Convert v0.3 triplets to legacy KnowledgeTriplet objects for backward compatibility.
        
        Args:
            triplets: List of v0.3 triplet dictionaries
            
        Returns:
            List of KnowledgeTriplet objects
        """
        legacy_triplets = []
        for triplet_data in triplets:
            legacy_triplet = self.llm_interface._convert_to_legacy_triplet(triplet_data)
            legacy_triplets.append(legacy_triplet)
        
        print(f"🔄 Converted {len(legacy_triplets)} triplets to legacy format")
        return legacy_triplets
    
    def get_ontology_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded ontology."""
        all_relations = self.ontology.get_all_relations()
        
        # Group statistics
        group_counts = {}
        qa_atomic_count = 0
        
        for rel_id, rel_info in all_relations.items():
            group = rel_info.get('group', 'Unknown')
            group_counts[group] = group_counts.get(group, 0) + 1
            
            # Simple heuristic for QA-Atomic detection
            if any(pattern in rel_id for pattern in [
                'BirthDate', 'BirthPlace', 'Nationality', 'CurrentPosition', 
                'CurrentEmployer', 'FoundingDate', 'CapitalOf', 'PublicationDate'
            ]):
                qa_atomic_count += 1
        
        # Inverse pair statistics
        inverse_pairs = self.ontology.get_auto_inverse_pairs()
        
        return {
            'total_relations': len(all_relations),
            'group_distribution': group_counts,
            'estimated_qa_atomic': qa_atomic_count,
            'auto_inverse_pairs': len(inverse_pairs) // 2,  # Each pair counted twice
            'include_optional': self.include_optional
        }
    
    def print_ontology_summary(self):
        """Print a human-readable summary of the ontology."""
        stats = self.get_ontology_stats()
        
        print("\n📊 Ontology Summary")
        print("=" * 50)
        print(f"Total Relations: {stats['total_relations']}")
        print(f"QA-Atomic Relations (estimated): {stats['estimated_qa_atomic']}")
        print(f"Auto-Inverse Pairs: {stats['auto_inverse_pairs']}")
        print(f"Include Optional: {stats['include_optional']}")
        
        print("\n📚 Group Distribution:")
        for group, count in sorted(stats['group_distribution'].items()):
            print(f"  {group}: {count} relations")


def quick_generate(seeds: List[str], budget: int = 20, language: str = "en", 
                  api_key_path: str = 'keys/openai.txt') -> List[Dict[str, Any]]:
    """
    Quick function to generate triplets without creating a persistent builder instance.
    
    Args:
        seeds: List of seed entity names
        budget: Maximum number of triplets to generate
        language: Language for surface text ("en" or "zh")
        api_key_path: Path to OpenAI API key file
        
    Returns:
        List of validated triplet dictionaries
    """
    builder = GraphBuilderV03(api_key_path=api_key_path)
    return builder.generate_from_seeds(seeds, budget, language)


if __name__ == "__main__":
    # Demo usage
    try:
        print("🚀 Graph Builder v0.3 Demo")
        
        # Initialize builder
        builder = GraphBuilderV03()
        
        # Print ontology summary
        builder.print_ontology_summary()
        
        # Generate from seeds
        seeds = ["Beijing", "Apple Inc.", "Albert Einstein", "Hamlet"]
        triplets = builder.generate_from_seeds(seeds, budget=25, language="en")
        
        # Show some results
        print(f"\n🎯 Sample Results ({min(5, len(triplets))} of {len(triplets)}):")
        for i, triplet in enumerate(triplets[:5]):
            print(f"\n{i+1}. ({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})")
            print(f"   Group: {triplet['group']}, QA-Eligible: {triplet['qa_eligible']}")
            print(f"   Confidence: {triplet['confidence']:.2f}")
            print(f"   Surface: {triplet['surface']}")
            if triplet['qualifiers']:
                print(f"   Qualifiers: {triplet['qualifiers']}")
        
        # Export results
        output_file = builder.export_triplets(triplets, "demo_output.jsonl", format="jsonl")
        print(f"\n💾 Results saved to: {output_file}")
        
        # Test validation
        valid_count = 0
        for triplet in triplets:
            is_valid, error = builder.validate_triplet(triplet)
            if is_valid:
                valid_count += 1
            else:
                print(f"⚠️ Validation error: {error}")
        
        print(f"\n✅ Validation: {valid_count}/{len(triplets)} triplets passed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure to:")
        print("1. Place your OpenAI API key in 'keys/openai.txt'")
        print("2. Run from the project root directory")
