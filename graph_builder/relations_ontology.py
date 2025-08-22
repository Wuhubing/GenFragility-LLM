#!/usr/bin/env python3
"""
Relations Ontology - 24 Core Relations with Domain/Range Types and Inverse Mappings
For dense knowledge graph construction with controlled expansion.
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import json

# Domain and Range Types (Enhanced for strict validation)
ENTITY_TYPES = {
    'Person', 'Org', 'Place', 'City', 'Country', 'Region', 
    'Class', 'Entity', 'Group', 'Time', 'Material', 'Work', 
    'Event', 'PropertyValue', 'Purpose', 'Action', 'Occupation',
    'Genre', 'Language', 'Product', 'Software', 'Agent', 'Tool'
}

# Type hierarchy for better compatibility checking
TYPE_HIERARCHY = {
    'City': ['Place'],
    'Country': ['Place'],
    'Region': ['Place'],
    'Person': ['Agent'],
    'Tool': ['Entity'],
    'Agent': ['Entity'],
}

# 24 Core Relations Definition
RELATIONS = {
    # Structure Group
    'InstanceOf': {
        'group': 'Structure',
        'domain': ['Entity', 'Person', 'Place', 'Work'],
        'range': ['Class'],
        'inverse': 'HasInstance',
        'description': 'Entity belongs to a class or category',
        'examples': [('Beijing', 'InstanceOf', 'City'), ('Einstein', 'InstanceOf', 'Person')]
    },
    'HasInstance': {
        'group': 'Structure', 
        'domain': ['Class'],
        'range': ['Entity', 'Person', 'Place', 'Work'],
        'inverse': 'InstanceOf',
        'description': 'Class contains instances',
        'examples': [('City', 'HasInstance', 'Beijing')]
    },
    'SubclassOf': {
        'group': 'Structure',
        'domain': ['Class'],
        'range': ['Class'], 
        'inverse': 'HasSubclass',
        'description': 'Class hierarchy relationship',
        'examples': [('Cat', 'SubclassOf', 'Animal')]
    },
    'HasSubclass': {
        'group': 'Structure',
        'domain': ['Class'],
        'range': ['Class'],
        'inverse': 'SubclassOf', 
        'description': 'Parent class has child classes',
        'examples': [('Animal', 'HasSubclass', 'Cat')]
    },
    'PartOf': {
        'group': 'Structure',
        'domain': ['Entity'],
        'range': ['Entity'],
        'inverse': 'HasPart',
        'description': 'Component relationship',
        'examples': [('Engine', 'PartOf', 'Car')]
    },
    'HasPart': {
        'group': 'Structure', 
        'domain': ['Entity'],
        'range': ['Entity'],
        'inverse': 'PartOf',
        'description': 'Whole contains parts',
        'examples': [('Car', 'HasPart', 'Engine')]
    },
    'MemberOf': {
        'group': 'Structure',
        'domain': ['Person', 'Entity'],
        'range': ['Group', 'Org'],
        'inverse': 'HasMember',
        'description': 'Membership in organization or group',
        'examples': [('Player', 'MemberOf', 'Team')]
    },
    'HasMember': {
        'group': 'Structure',
        'domain': ['Group', 'Org'], 
        'range': ['Person', 'Entity'],
        'inverse': 'MemberOf',
        'description': 'Organization has members',
        'examples': [('Team', 'HasMember', 'Player')]
    },

    # Attributes Group
    'HasProperty': {
        'group': 'Attributes',
        'domain': ['Entity'],
        'range': ['PropertyValue'],
        'inverse': None,
        'description': 'Entity has a characteristic property',
        'examples': [('Ice', 'HasProperty', 'Cold')]
    },
    'MadeOf': {
        'group': 'Attributes',
        'domain': ['Entity'],
        'range': ['Material'],
        'inverse': None,
        'description': 'Material composition',
        'examples': [('Bottle', 'MadeOf', 'Plastic')]
    },
    'Genre': {
        'group': 'Attributes',
        'domain': ['Work', 'Event'],
        'range': ['Genre'],
        'inverse': None,
        'description': 'Category or type classification',
        'examples': [('Paper', 'Genre', 'Review')]
    },

    # Spatial Group  
    'LocatedIn': {
        'group': 'Spatial',
        'domain': ['Entity', 'Person', 'Org'],
        'range': ['Place', 'City', 'Country', 'Region'],
        'inverse': 'Contains',
        'description': 'Spatial containment relationship',
        'examples': [('Tsinghua', 'LocatedIn', 'Beijing')]
    },
    'Contains': {
        'group': 'Spatial',
        'domain': ['Place', 'City', 'Country', 'Region'],
        'range': ['Entity', 'Person', 'Org'],
        'inverse': 'LocatedIn',
        'description': 'Spatial container relationship', 
        'examples': [('Beijing', 'Contains', 'Tsinghua')]
    },
    'LocatedNear': {
        'group': 'Spatial',
        'domain': ['Place'],
        'range': ['Place'],
        'inverse': 'LocatedNear',
        'description': 'Spatial proximity (symmetric)',
        'examples': [('Campus', 'LocatedNear', 'Metro Station')]
    },
    'CapitalOf': {
        'group': 'Spatial',
        'domain': ['City'],
        'range': ['Country', 'Region'],
        'inverse': 'HasCapital',
        'description': 'Administrative center relationship',
        'examples': [('Beijing', 'CapitalOf', 'China')]
    },
    'HasCapital': {
        'group': 'Spatial',
        'domain': ['Country', 'Region'],
        'range': ['City'],
        'inverse': 'CapitalOf',
        'description': 'Country has capital city',
        'examples': [('China', 'HasCapital', 'Beijing')]
    },
    'BorderWith': {
        'group': 'Spatial',
        'domain': ['Country', 'Region'],
        'range': ['Country', 'Region'], 
        'inverse': 'BorderWith',
        'description': 'Geographic border (symmetric)',
        'examples': [('France', 'BorderWith', 'Germany')]
    },

    # Temporal Group
    'StartTime': {
        'group': 'Temporal',
        'domain': ['Entity', 'Event', 'Org'],
        'range': ['Time'],
        'inverse': None,
        'description': 'Beginning time of entity or event',
        'examples': [('Company', 'StartTime', '1998')]
    },
    'EndTime': {
        'group': 'Temporal', 
        'domain': ['Entity', 'Event', 'Org'],
        'range': ['Time'],
        'inverse': None,
        'description': 'Ending time of entity or event',
        'examples': [('Event', 'EndTime', '2024-12-31')]
    },
    'OccursOn': {
        'group': 'Temporal',
        'domain': ['Event'],
        'range': ['Time'],
        'inverse': None,
        'description': 'Event occurrence time',
        'examples': [('Olympics', 'OccursOn', '2024-08-08')]
    },

    # Causal/Event Group
    'Causes': {
        'group': 'Causal',
        'domain': ['Event', 'Action', 'Entity'],
        'range': ['Event', 'Action', 'Entity'],
        'inverse': 'CausedBy',
        'description': 'Causal relationship',
        'examples': [('Exercise', 'Causes', 'Sweating')]
    },
    'CausedBy': {
        'group': 'Causal',
        'domain': ['Event', 'Action', 'Entity'],
        'range': ['Event', 'Action', 'Entity'],
        'inverse': 'Causes',
        'description': 'Effect caused by something',
        'examples': [('Sweating', 'CausedBy', 'Exercise')]
    },
    'HasPrerequisite': {
        'group': 'Causal',
        'domain': ['Event', 'Action'],
        'range': ['Event', 'Action'],
        'inverse': 'PrerequisiteFor',
        'description': 'Required precondition',
        'examples': [('Exam', 'HasPrerequisite', 'Study')]
    },
    'PrerequisiteFor': {
        'group': 'Causal',
        'domain': ['Event', 'Action'],
        'range': ['Event', 'Action'], 
        'inverse': 'HasPrerequisite',
        'description': 'Serves as prerequisite for',
        'examples': [('Study', 'PrerequisiteFor', 'Exam')]
    },
    'HasSubevent': {
        'group': 'Causal',
        'domain': ['Event'],
        'range': ['Event'],
        'inverse': 'SubeventOf',
        'description': 'Event contains sub-events',
        'examples': [('Dining', 'HasSubevent', 'Ordering')]
    },
    'SubeventOf': {
        'group': 'Causal',
        'domain': ['Event'],
        'range': ['Event'],
        'inverse': 'HasSubevent',
        'description': 'Sub-event of larger event',
        'examples': [('Ordering', 'SubeventOf', 'Dining')]
    },

    # Functionality Group
    'UsedFor': {
        'group': 'Function',
        'domain': ['Entity'],
        'range': ['Purpose', 'Action'],
        'inverse': None,
        'description': 'Purpose or function of entity',
        'examples': [('Scissors', 'UsedFor', 'Cutting')]
    },
    'CapableOf': {
        'group': 'Function',
        'domain': ['Person', 'Entity'],
        'range': ['Action'],
        'inverse': None,
        'description': 'Ability or capability',
        'examples': [('Robot', 'CapableOf', 'Lifting')]
    },

    # Social/Role Group
    'Occupation': {
        'group': 'Social',
        'domain': ['Person'],
        'range': ['Occupation'],
        'inverse': None,
        'description': 'Professional role or job',
        'examples': [('Einstein', 'Occupation', 'Physicist')]
    },
    'Employer': {
        'group': 'Social',
        'domain': ['Person'],
        'range': ['Org'],
        'inverse': 'HasEmployee',
        'description': 'Employment relationship',
        'examples': [('Engineer', 'Employer', 'Company')]
    },
    'HasEmployee': {
        'group': 'Social',
        'domain': ['Org'],
        'range': ['Person'],
        'inverse': 'Employer',
        'description': 'Organization employs person',
        'examples': [('Company', 'HasEmployee', 'Engineer')]
    },
    'CreatedBy': {
        'group': 'Social',
        'domain': ['Work', 'Entity'],
        'range': ['Person', 'Org'],
        'inverse': 'CreatorOf',
        'description': 'Creator or author relationship',
        'examples': [('Lightbulb', 'CreatedBy', 'Edison')]
    },
    'CreatorOf': {
        'group': 'Social',
        'domain': ['Person', 'Org'],
        'range': ['Work', 'Entity'],
        'inverse': 'CreatedBy',
        'description': 'Person/org created something',
        'examples': [('Edison', 'CreatorOf', 'Lightbulb')]
    },
    'HeadquarteredIn': {
        'group': 'Social',
        'domain': ['Org'],
        'range': ['Place', 'City'],
        'inverse': 'HostsHQ',
        'description': 'Organization headquarters location',
        'examples': [('Company', 'HeadquarteredIn', 'Beijing')]
    },
    'HostsHQ': {
        'group': 'Social', 
        'domain': ['Place', 'City'],
        'range': ['Org'],
        'inverse': 'HeadquarteredIn',
        'description': 'Location hosts organization HQ',
        'examples': [('Beijing', 'HostsHQ', 'Company')]
    }
}

# Optional 6 relations for density expansion (to reach 30 total)
OPTIONAL_RELATIONS = {
    'Nationality': {
        'group': 'Social',
        'domain': ['Person'],
        'range': ['Country'],
        'inverse': None,
        'description': 'Person nationality',
        'examples': [('Einstein', 'Nationality', 'German')]
    },
    'LanguageUsed': {
        'group': 'Attributes',
        'domain': ['Person', 'Work'],
        'range': ['Language'],
        'inverse': None,
        'description': 'Language used by person or in work',
        'examples': [('Paper', 'LanguageUsed', 'English')]
    },
    'DevelopedBy': {
        'group': 'Social',
        'domain': ['Product', 'Software'],
        'range': ['Org'],
        'inverse': None,
        'description': 'Product developed by organization',
        'examples': [('iPhone', 'DevelopedBy', 'Apple')]
    },
    'NamedAfter': {
        'group': 'Attributes',
        'domain': ['Entity'],
        'range': ['Entity'],
        'inverse': None,
        'description': 'Named in honor of something/someone',
        'examples': [('Einstein Street', 'NamedAfter', 'Einstein')]
    },
    'DiplomaticRelation': {
        'group': 'Social',
        'domain': ['Country', 'Org'],
        'range': ['Country', 'Org'],
        'inverse': 'DiplomaticRelation',
        'description': 'Diplomatic relationship (symmetric)',
        'examples': [('USA', 'DiplomaticRelation', 'China')]
    },
    'WorkLocation': {
        'group': 'Social',
        'domain': ['Person'],
        'range': ['Place'],
        'inverse': None,
        'description': 'Person work location',
        'examples': [('Researcher', 'WorkLocation', 'University')]
    }
}

# Relation groups and their target proportions
RELATION_GROUPS = {
    'Structure': 0.25,  # 25%
    'Spatial': 0.15,    # 15% 
    'Temporal': 0.10,   # 10%
    'Causal': 0.15,     # 15%
    'Function': 0.15,   # 15%
    'Social': 0.20,     # 20%
    'Attributes': 0.0   # Distributed among others
}

# Per-entity caps for explosion-prone relations
RELATION_CAPS = {
    'InstanceOf': 3,
    'SubclassOf': 3, 
    'LocatedIn': 3,
    'PartOf': 3,
    '*': 5  # Default cap for other relations
}

# Global soft cap - no single relation should exceed 15% of total edges
GLOBAL_SOFT_CAP = 0.15

def get_all_relations(include_optional: bool = False) -> Dict:
    """Get all relations, optionally including the 6 optional ones."""
    all_relations = RELATIONS.copy()
    if include_optional:
        all_relations.update(OPTIONAL_RELATIONS)
    return all_relations

def get_relation_info(relation_id: str, include_optional: bool = False) -> Optional[Dict]:
    """Get information about a specific relation."""
    all_relations = get_all_relations(include_optional)
    return all_relations.get(relation_id)

def is_valid_relation(relation_id: str, include_optional: bool = False) -> bool:
    """Check if a relation ID is valid."""
    all_relations = get_all_relations(include_optional)
    return relation_id in all_relations

def get_relations_by_group(group: str, include_optional: bool = False) -> List[str]:
    """Get all relation IDs in a specific group."""
    all_relations = get_all_relations(include_optional)
    return [rel_id for rel_id, info in all_relations.items() if info['group'] == group]

def is_type_compatible(head_type: str, relation_id: str, tail_type: str, include_optional: bool = False) -> bool:
    """Check if entity types are compatible with relation domain/range with hierarchy support."""
    relation_info = get_relation_info(relation_id, include_optional)
    if not relation_info:
        return False
    
    def check_type_match(entity_type: str, allowed_types: List[str]) -> bool:
        """Check if entity_type matches any allowed type, considering hierarchy."""
        if entity_type in allowed_types or 'Entity' in allowed_types or entity_type == 'Entity':
            return True
        
        # Check hierarchy - if entity_type is a subtype of any allowed type
        if entity_type in TYPE_HIERARCHY:
            for parent_type in TYPE_HIERARCHY[entity_type]:
                if parent_type in allowed_types:
                    return True
        
        return False
    
    # Check domain compatibility with hierarchy
    domain_match = check_type_match(head_type, relation_info['domain'])
    
    # Check range compatibility with hierarchy
    range_match = check_type_match(tail_type, relation_info['range'])
    
    return domain_match and range_match

def get_inverse_relation(relation_id: str, include_optional: bool = False) -> Optional[str]:
    """Get the inverse relation if it exists."""
    relation_info = get_relation_info(relation_id, include_optional)
    if relation_info:
        return relation_info.get('inverse')
    return None

def get_relation_examples(relation_id: str, include_optional: bool = False) -> List[Tuple[str, str, str]]:
    """Get example triplets for a relation."""
    relation_info = get_relation_info(relation_id, include_optional)
    if relation_info and 'examples' in relation_info:
        return relation_info['examples']
    return []

# Triplet Schema Class
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
        relation_info = get_relation_info(relation_id, include_optional=True)
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
    # Test the ontology
    print(f"Total relations: {len(get_all_relations())}")
    print(f"With optional: {len(get_all_relations(include_optional=True))}")
    
    print("\nRelation groups:")
    for group in set(info['group'] for info in RELATIONS.values()):
        relations = get_relations_by_group(group)
        print(f"  {group}: {len(relations)} relations - {relations}")
    
    print(f"\nType compatibility test:")
    print(f"Beijing-CapitalOf-China: {is_type_compatible('City', 'CapitalOf', 'Country')}")
    print(f"Person-CapitalOf-Country: {is_type_compatible('Person', 'CapitalOf', 'Country')}")
    
    print(f"\nInverse relations test:")
    print(f"CapitalOf -> {get_inverse_relation('CapitalOf')}")
    print(f"Causes -> {get_inverse_relation('Causes')}")
    
    # Test triplet creation
    triplet = KnowledgeTriplet('Beijing', 'CapitalOf', 'China', 
                              domain_guess='City', range_guess='Country',
                              confidence=0.95)
    print(f"\nSample triplet: {triplet.to_dict()}")
