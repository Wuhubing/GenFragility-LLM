#!/usr/bin/env python3
"""
Relations Ontology - 24 Core Relations with Domain/Range Types and Inverse Mappings
For dense knowledge graph construction with controlled expansion.
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import json
from pathlib import Path

# --- NEW: JSON-based Ontology Service ---
class RelationOntology:
    """
    Loads and manages the knowledge graph's relation ontology from JSON files.
    Provides methods for normalization, validation, and property access.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RelationOntology, cls).__new__(cls)
        return cls._instance

    def __init__(self, relations_dir: Optional[Path] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        if relations_dir is None:
            relations_dir = Path(__file__).parent / 'relations'

        self._canonical_relations: Dict[str, Dict] = {}
        self._alias_map: Dict[str, str] = {}
        self._inverse_map: Dict[str, str] = {}
        self._relations_that_swap: Set[str] = set()

        self._load_ontology(relations_dir)
        self._initialized = True

    def _load_ontology(self, relations_dir: Path):
        """Loads all ontology components from their respective JSON files."""
        canonical_path = relations_dir / 'canonical_relations.json'
        with open(canonical_path, 'r', encoding='utf-8') as f:
            canonical_data = json.load(f)
        self._canonical_relations = {item['relation_id']: item for item in canonical_data}

        alias_path = relations_dir / 'alias_to_canonical.json'
        with open(alias_path, 'r', encoding='utf-8') as f:
            alias_data = json.load(f)
        
        for item in alias_data:
            alias = item['alias']
            canonical = item['canonical']
            self._alias_map[alias] = canonical
            note = item.get('note', '')
            if '统一正向' in note or 'inverse' in note.lower():
                 self._relations_that_swap.add(alias)

        inverse_path = relations_dir / 'auto_inverse_pairs.json'
        with open(inverse_path, 'r', encoding='utf-8') as f:
            inverse_data = json.load(f)
        for pair in inverse_data:
            self._inverse_map[pair['canonical']] = pair['auto_inverse']
            self._inverse_map[pair['auto_inverse']] = pair['canonical']

    def normalize_triplet(self, head: str, relation: str, tail: str) -> Tuple[str, str, str]:
        """Normalizes a triplet to use canonical relation IDs and corrects direction."""
        should_swap = relation in self._relations_that_swap
        canonical_relation = self._alias_map.get(relation, relation)

        if should_swap:
            return tail, canonical_relation, head
        else:
            return head, canonical_relation, tail

    def is_valid_relation(self, relation_id: str) -> bool:
        """Checks if a relation ID is a valid canonical relation."""
        return relation_id in self._canonical_relations

    def get_relation_info(self, relation_id: str) -> Optional[Dict]:
        """Returns the full definition dictionary for a canonical relation."""
        return self._canonical_relations.get(relation_id)

    def get_inverse(self, relation_id: str) -> Optional[str]:
        """Returns the inverse of a given relation, if one exists."""
        return self._inverse_map.get(relation_id)

    def get_all_relations(self) -> Dict[str, Dict]:
        """Returns the complete dictionary of all canonical relations."""
        return self._canonical_relations
    
    def get_auto_inverse_pairs(self) -> Dict[str, str]:
        """Get all inverse pairs."""
        return self._inverse_map.copy()

# --- END NEW ---


# We will keep KnowledgeTriplet for now as other modules might depend on it.
class KnowledgeTriplet:
    """Structured representation of a knowledge triplet with metadata."""
    
    def __init__(self, head: str, relation_id: str, tail: str,
                 domain_guess: str = "Entity", range_guess: str = "Entity",
                 surface: str = "", evidence: str = "", confidence: float = 0.0,
                 inverse_auto: bool = True, gen_params: Dict = None, question: str = ""):
        self.head = head.strip()
        self.relation_id = relation_id
        self.tail = tail.strip()
        self.domain_guess = domain_guess
        self.range_guess = range_guess
        self.surface = surface
        self.evidence = evidence
        self.confidence = confidence
        self.inverse_auto = inverse_auto
        self.question = question
        self.created_at = datetime.now().isoformat()
        self.gen_params = gen_params or {}
        
        # Auto-populate group from relation
        # Note: This now requires getting the singleton instance of the ontology
        ontology = RelationOntology()
        relation_info = ontology.get_relation_info(relation_id)
        self.group = relation_info['group'] if relation_info else 'Unknown'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'head': self.head,
            'relation_id': self.relation_id,
            'tail': self.tail,
            'group': self.group,
            'domain_guess': self.domain_guess,
            'range_guess': self.range_guess,
            'inverse_auto': self.inverse_auto,
            'surface': self.surface,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'question': self.question,
            'created_at': self.created_at,
            'gen_params': self.gen_params
        }
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to simple (head, relation, tail) tuple."""
        return (self.head, self.relation_id, self.tail)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeTriplet':
        """Create from dictionary."""
        return cls(
            head=data['head'],
            relation_id=data['relation_id'], 
            tail=data['tail'],
            domain_guess=data.get('domain_guess', 'Entity'),
            range_guess=data.get('range_guess', 'Entity'),
            surface=data.get('surface', ''),
            evidence=data.get('evidence', ''),
            confidence=data.get('confidence', 0.0),
            inverse_auto=data.get('inverse_auto', True),
            gen_params=data.get('gen_params', {})
        )

if __name__ == "__main__":
    # Test the new ontology system
    ontology = RelationOntology()
    
    print(f"Total canonical relations: {len(ontology.get_all_relations())}")
    
    # Test normalization
    print("\n--- Normalization Tests ---")
    h, r, t = ontology.normalize_triplet('PersonA', 'WorksAt', 'OrgB')
    print(f"'WorksAt' -> Normalized to: ({h}, {r}, {t})")
    
    h, r, t = ontology.normalize_triplet('Ordering', 'SubEventOf', 'Dining')
    print(f"'SubEventOf' -> Normalized to: ({h}, {r}, {t}) (should swap)")

    # Test validation
    print("\n--- Validation Tests ---")
    print(f"Is 'Employer' valid? {ontology.is_valid_relation('Employer')}")
    print(f"Is 'WorksAt' valid? {ontology.is_valid_relation('WorksAt')} (should be false)")

    # Test inverse
    print("\n--- Inverse Tests ---")
    print(f"Inverse of 'HasSubevent': {ontology.get_inverse('HasSubevent')}")
    print(f"Inverse of 'SubeventOf': {ontology.get_inverse('SubeventOf')}")
    print(f"Inverse of 'PartOf': {ontology.get_inverse('PartOf')}")

    # Test info retrieval
    print("\n--- Info Retrieval ---")
    info = ontology.get_relation_info('LocatedIn')
    if info:
        print(f"Info for 'LocatedIn': Group={info['group']}, Domain={info['domain']}, Range={info['range']}")
