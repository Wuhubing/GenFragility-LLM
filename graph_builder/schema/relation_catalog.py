import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class RelationSpec:
    name: str
    group: Optional[str] = None
    domain: Optional[str] = None
    range: Optional[str] = None
    cardinality: Optional[str] = None
    temporal: bool = False
    inverse: Optional[str] = None
    align: Dict[str, Any] = field(default_factory=dict)
    caps: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def wikidata_id(self) -> Optional[str]:
        # Helper to maintain compatibility with evidence_validator
        return self.align.get('wikidata')

class RelationCatalog:
    def __init__(self, relation_specs: Dict[str, RelationSpec]):
        self._relations = relation_specs

    def get(self, relation_name: str) -> Optional[RelationSpec]:
        return self._relations.get(relation_name)
    
    @classmethod
    def from_yaml(cls, filepath: str, include_optional: bool = False):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        specs = {}
        
        def process_relations(rel_dict):
            if not rel_dict: return
            for name, properties in rel_dict.items():
                specs[name] = RelationSpec(name=name, **properties)

        process_relations(data.get('relations', {}))
        
        if include_optional:
            process_relations(data.get('optional_relations', {}))
            
        return cls(specs)

# Example usage:
# catalog = RelationCatalog.from_yaml('graph_builder/configs/relation_alignment.yaml', include_optional=True)
# spec = catalog.get('HeadquarteredIn')
# if spec:
#     print(spec.cardinality)