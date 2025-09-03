"""
Normalization utilities for entities and relations.
"""

import re
from typing import Dict, Any, Tuple


def normalize_entity(entity: str) -> str:
    """Normalize an entity name."""
    if not entity or not isinstance(entity, str):
        return ""
    
    # Basic normalization
    normalized = entity.strip()
    
    # Remove excessive whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Title case for proper nouns (simple heuristic)
    if len(normalized.split()) <= 3:  # Short phrases, likely proper nouns
        normalized = normalized.title()
    
    return normalized


def normalize_relation(relation: str) -> str:
    """Normalize a relation name."""
    if not relation or not isinstance(relation, str):
        return ""
    
    # Basic normalization
    normalized = relation.strip()
    
    # Remove excessive whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Convert to PascalCase for consistency
    words = normalized.split()
    normalized = ''.join(word.capitalize() for word in words)
    
    return normalized


def normalize_triple(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Normalize a complete triple (head, relation, tail)."""
    if not triple or len(triple) != 3:
        return ("", "", "")
    
    head, relation, tail = triple
    
    return (
        normalize_entity(head),
        normalize_relation(relation),
        normalize_entity(tail)
    )


def normalize_triple_dict(triple_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a triple dictionary with additional metadata."""
    if not isinstance(triple_dict, dict):
        return {}
    
    normalized = triple_dict.copy()
    
    # Normalize core triple components
    if 'head' in normalized:
        normalized['head'] = normalize_entity(normalized['head'])
    if 'relation' in normalized:
        normalized['relation'] = normalize_relation(normalized['relation'])
    if 'tail' in normalized:
        normalized['tail'] = normalize_entity(normalized['tail'])
    
    # Update triplet list if present
    if 'triplet' in normalized and isinstance(normalized['triplet'], list) and len(normalized['triplet']) >= 3:
        normalized['triplet'] = [
            normalize_entity(normalized['triplet'][0]),
            normalize_relation(normalized['triplet'][1]),
            normalize_entity(normalized['triplet'][2])
        ]
    
    return normalized
