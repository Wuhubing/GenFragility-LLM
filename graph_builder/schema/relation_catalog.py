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
    def from_yaml(cls, filepath: str, config: Dict[str, Any] = None):
        if config is None:
            config = {}
            
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        specs = {}
        
        def process_relations(rel_dict):
            if not rel_dict: return
            for name, properties in rel_dict.items():
                specs[name] = RelationSpec(name=name, **properties)

        process_relations(data.get('relations', {}))
        
        # Note: The optional_relations switch is now handled in the main script's logic
        # This keeps the catalog loader cleaner.
        
        # If the strict 1-to-1 enforcement is enabled, filter the specs
        if config.get('enforce_strict_1_to_1', False):
            strict_specs = {}
            for name, spec in specs.items():
                if spec.cardinality in ["1:1", "1:1_temporal"]:
                    strict_specs[name] = spec
            specs = strict_specs
            
        return cls(specs)

# Example usage:
# with open('graph_builder/configs/builder.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# catalog = RelationCatalog.from_yaml('graph_builder/configs/relation_alignment.yaml', config=config)
# spec = catalog.get('InstanceOf') # This should be None if enforce_strict_1_to_1 is true
# print(spec)