from typing import Tuple, Optional
import networkx as nx

from graph_builder.schema.json_models import CandidateTriple
from graph_builder.schema.relation_catalog import RelationCatalog

def enforce_cardinality(
    G: nx.Graph, 
    tri: CandidateTriple, 
    catalog: RelationCatalog
) -> Tuple[bool, Optional[str]]:
    """
    Checks a triple against the graph to enforce cardinality constraints before insertion.
    This must be called within a lock for the (head, relation) pair.
    """
    spec = catalog.get(tri.relation)
    if not spec or not spec.cardinality:
        return True, None # No constraint to enforce

    # Find existing edges for this head and relation
    existing = []
    if G.has_node(tri.head):
        for u, v, data in G.edges(tri.head, data=True):
            if data.get("relation") == tri.relation:
                existing.append((u, v, data))

    if not existing:
        return True, None

    # Enforce 1:1 constraint
    if spec.cardinality == "1:1":
        # If any edge exists with a different tail, it's a conflict.
        if any(v != tri.tail for _, v, _ in existing):
            return False, "cardinality_1:1_conflict"
        return True, None

    # Enforce 1:1_temporal constraint
    if spec.cardinality == "1:1_temporal":
        # A temporal relation must have some form of date
        if not (tri.start_time or tri.as_of_date or tri.end_time):
            return False, "temporal_relation_missing_time_at_insert"
            
        # Use a very early or late date string for open-ended intervals (e.g., 'present')
        new_start = tri.start_time or "0000-01-01"
        new_end = tri.end_time or "9999-12-31"

        for _, v, data in existing:
            # If it's the same fact, allow update (idempotency)
            if v == tri.tail:
                continue
            
            existing_start = data.get("start_time") or "0000-01-01"
            existing_end = data.get("end_time") or "9999-12-31"
            
            # Check for overlap: not (new_end < existing_start or existing_end < new_start)
            if not (new_end < existing_start or existing_end < new_start):
                return False, "cardinality_1:1_temporal_overlap"
                
        return True, None

    return True, None