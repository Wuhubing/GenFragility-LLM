import yaml
from typing import Tuple, Optional, Dict, Set

from graph_builder.schema.json_models import CandidateTriple
from graph_builder.schema.relation_catalog import RelationCatalog

_controlled_vocabs: Dict[str, Set[str]] = {}

def _load_controlled_vocab(name: str, filepath: str) -> Set[str]:
    """Loads a controlled vocabulary from a YAML file."""
    global _controlled_vocabs
    if name not in _controlled_vocabs:
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
                _controlled_vocabs[name] = set(data.get(name, []))
        except FileNotFoundError:
            _controlled_vocabs[name] = set()
    return _controlled_vocabs[name]


def schema_guard(
    tri: CandidateTriple, 
    catalog: RelationCatalog, 
    type_map: Optional[Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Performs fast, pre-flight checks on a triple against the defined schema.
    """
    spec = catalog.get(tri.relation)
    
    # Check 1: Is the relation whitelisted?
    if not spec:
        return False, "relation_not_whitelisted"
        
    # Check 2: If the relation is temporal, is a time field present?
    if spec.temporal: # Use the explicit temporal flag
        if not (tri.start_time or tri.as_of_date or tri.end_time):
            return False, "temporal_relation_missing_time"

    # Check 3: Controlled vocabulary check for PrimaryIndustry
    if tri.relation == "PrimaryIndustry":
        industries = _load_controlled_vocab("industries", "graph_builder/configs/controlled_vocab.yaml")
        if industries and tri.tail not in industries:
            return False, f"tail_not_in_controlled_vocab:{tri.tail}"
            
    # Check 4: Domain/range coarse type checking (if type info is available)
    # This is a placeholder for a more robust type checking system.
    if type_map:
        # e.g., if type_map.get(tri.head) != spec.domain: return False, "domain_mismatch"
        pass
        
    # Check 5: Tail type consistency (e.g., is the tail a literal when it should be?)
    # Placeholder for a more robust check.

    return True, None
