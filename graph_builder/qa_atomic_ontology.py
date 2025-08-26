#!/usr/bin/env python3
"""
QA-Atomic Relations Ontology - 36 Function-like Relations
Designed for unique, answerable knowledge graph construction.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime


class QAAtomicOntology:
    """
    QA-Atomic ontology with 36 function-like relations designed for unique answers.
    All relations are either naturally unique or can be made unique with qualifiers.
    """
    
    def __init__(self, relations_file: Optional[Path] = None):
        if relations_file is None:
            relations_file = Path(__file__).parent / 'relations_qa.json'
        
        self._relations: Dict[str, Dict] = {}
        self._inverse_map: Dict[str, str] = {}
        self._group_relations: Dict[str, List[str]] = {}
        
        self._load_relations(relations_file)
        self._build_inverse_map()
        self._build_group_map()
    
    def _load_relations(self, relations_file: Path):
        """Load QA-Atomic relations from JSON file."""
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations_list = json.load(f)
        
        for rel_data in relations_list:
            rel_id = rel_data['relation_id']
            self._relations[rel_id] = rel_data
    
    def _build_inverse_map(self):
        """Build inverse relation mapping based on inverse_policy."""
        for rel_id, rel_data in self._relations.items():
            policy = rel_data.get('inverse_policy', 'none')
            
            if policy == 'auto':
                # Auto-inverse: create automatic inverse relation name
                if rel_id.endswith('Primary'):
                    base_name = rel_id[:-7]  # Remove 'Primary'
                    inverse_id = f"Has{base_name}"
                elif rel_id.endswith('Current'):
                    base_name = rel_id[:-7]  # Remove 'Current'
                    inverse_id = f"Has{base_name}"
                else:
                    inverse_id = f"InverseOf{rel_id}"
                
                self._inverse_map[rel_id] = inverse_id
                self._inverse_map[inverse_id] = rel_id
            
            elif policy == 'paired':
                # Paired relations need manual definition
                if rel_id == 'HeadquartersCity':
                    inverse_id = 'HeadquartersOf'
                    self._inverse_map[rel_id] = inverse_id
                    self._inverse_map[inverse_id] = rel_id
    
    def _build_group_map(self):
        """Build mapping from groups to relations."""
        for rel_id, rel_data in self._relations.items():
            group = rel_data.get('group', 'Unknown')
            if group not in self._group_relations:
                self._group_relations[group] = []
            self._group_relations[group].append(rel_id)
    
    def get_all_relations(self) -> Dict[str, Dict]:
        """Get all QA-Atomic relations."""
        return self._relations.copy()
    
    def get_relation_info(self, relation_id: str) -> Optional[Dict]:
        """Get information for a specific relation."""
        return self._relations.get(relation_id)
    
    def is_valid_relation(self, relation_id: str) -> bool:
        """Check if a relation ID is valid."""
        return relation_id in self._relations
    
    def get_inverse(self, relation_id: str) -> Optional[str]:
        """Get the inverse of a relation, if one exists."""
        return self._inverse_map.get(relation_id)
    
    def get_auto_inverse_pairs(self) -> Dict[str, str]:
        """Get all inverse pairs."""
        return self._inverse_map.copy()
    
    def get_relations_by_group(self, group: str) -> List[str]:
        """Get all relations in a specific group."""
        return self._group_relations.get(group, [])
    
    def get_all_groups(self) -> List[str]:
        """Get all relation groups."""
        return list(self._group_relations.keys())
    
    def is_qa_atomic(self, relation_id: str) -> bool:
        """Check if a relation is QA-Atomic (all relations in this ontology are)."""
        return relation_id in self._relations
    
    def get_required_qualifiers(self, relation_id: str) -> List[str]:
        """Get required qualifiers for a relation."""
        rel_info = self.get_relation_info(relation_id)
        if rel_info:
            return rel_info.get('qualifiers_required', [])
        return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the ontology."""
        total_relations = len(self._relations)
        group_counts = {group: len(rels) for group, rels in self._group_relations.items()}
        
        auto_inverse_count = sum(1 for rel_data in self._relations.values() 
                               if rel_data.get('inverse_policy') == 'auto')
        paired_inverse_count = sum(1 for rel_data in self._relations.values() 
                                 if rel_data.get('inverse_policy') == 'paired')
        
        qualifier_stats = {}
        for rel_id, rel_data in self._relations.items():
            qualifiers = rel_data.get('qualifiers_required', [])
            for q in qualifiers:
                qualifier_stats[q] = qualifier_stats.get(q, 0) + 1
        
        return {
            'total_relations': total_relations,
            'group_distribution': group_counts,
            'auto_inverse_relations': auto_inverse_count,
            'paired_inverse_relations': paired_inverse_count,
            'qualifier_usage': qualifier_stats,
            'all_qa_atomic': True
        }
    
    def print_summary(self):
        """Print a human-readable summary of the ontology."""
        stats = self.get_stats()
        
        print("📊 QA-Atomic Ontology Summary")
        print("=" * 50)
        print(f"Total Relations: {stats['total_relations']}")
        print(f"All QA-Atomic: {stats['all_qa_atomic']}")
        print(f"Auto-Inverse Relations: {stats['auto_inverse_relations']}")
        print(f"Paired-Inverse Relations: {stats['paired_inverse_relations']}")
        
        print("\n📚 Group Distribution:")
        for group, count in sorted(stats['group_distribution'].items()):
            relations = self.get_relations_by_group(group)
            print(f"  {group}: {count} relations")
            for rel in relations[:3]:  # Show first 3
                qualifiers = self.get_required_qualifiers(rel)
                qual_str = f" (requires: {', '.join(qualifiers)})" if qualifiers else ""
                print(f"    - {rel}{qual_str}")
            if len(relations) > 3:
                print(f"    - ... and {len(relations) - 3} more")
        
        print(f"\n🔧 Qualifier Usage:")
        for qualifier, count in stats['qualifier_usage'].items():
            print(f"  {qualifier}: {count} relations")
    
    def normalize_triplet(self, head: str, relation: str, tail: str) -> Tuple[str, str, str]:
        """Normalizes a triplet to use canonical relation IDs (QA-Atomic relations don't need swapping)."""
        # QA-Atomic relations are designed to be unidirectional and don't need swapping
        # Just verify the relation exists
        if relation in self._relations:
            return head, relation, tail
        else:
            # Return as-is if not in our ontology (will be caught by validation)
            return head, relation, tail


# Compatibility layer for existing code
class KnowledgeTriplet:
    """Structured representation of a knowledge triplet with metadata."""
    
    def __init__(self, head: str, relation_id: str, tail: str,
                 domain_guess: str = "Entity", range_guess: str = "Entity",
                 surface: str = "", evidence: str = "", confidence: float = 0.0,
                 inverse_auto: bool = True, gen_params: Dict = None):
        self.head = head.strip()
        self.relation_id = relation_id
        self.tail = tail.strip()
        self.domain_guess = domain_guess
        self.range_guess = range_guess
        self.surface = surface
        self.evidence = evidence
        self.confidence = confidence
        self.inverse_auto = inverse_auto
        self.created_at = datetime.now().isoformat()
        self.gen_params = gen_params or {}
        
        # Auto-populate group from relation
        ontology = QAAtomicOntology()
        relation_info = ontology.get_relation_info(relation_id)
        self.group = relation_info['group'] if relation_info else 'Unknown'
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to a simple (head, relation, tail) tuple."""
        return (self.head, self.relation_id, self.tail)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'head': self.head,
            'relation_id': self.relation_id,
            'tail': self.tail,
            'domain_guess': self.domain_guess,
            'range_guess': self.range_guess,
            'surface': self.surface,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'group': self.group,
            'created_at': self.created_at,
            'gen_params': self.gen_params
        }


if __name__ == "__main__":
    # Test the QA-Atomic ontology
    ontology = QAAtomicOntology()
    ontology.print_summary()
    
    # Test some specific functions
    print(f"\n🧪 Testing specific relations:")
    print(f"BirthDate info: {ontology.get_relation_info('BirthDate')}")
    print(f"CurrentEmployer inverse: {ontology.get_inverse('CurrentEmployer')}")
    print(f"Person group relations: {ontology.get_relations_by_group('Person')}")
    print(f"Required qualifiers for NationalityPrimary: {ontology.get_required_qualifiers('NationalityPrimary')}")
