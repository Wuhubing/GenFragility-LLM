from typing import Dict, Any
from dataclasses import dataclass

from graph_builder.schema.json_models import CandidateTriple

@dataclass
class ScoringWeights:
    schema: float = 1.0 # Base score for passing schema checks
    evidence: float = 2.0
    consistency: float = 1.5
    closure_bonus: float = 0.5

def score(
    tri: CandidateTriple, 
    evidence: Dict[str, Any], 
    self_consistency: float, 
    closure_hint: bool, 
    weights: ScoringWeights
) -> float:
    """
    Calculates a weighted score for a candidate triple.
    """
    s = 0.0
    
    # Base score for passing the schema guard (implied, as this is called after)
    s += weights.schema * 1.0
    
    # Add score from external evidence
    if evidence and evidence.get("passed"):
        s += weights.evidence * evidence.get("score", 0.0)
    
    # Add score from LLM self-consistency
    s += weights.consistency * self_consistency
    
    # Add bonus for forming a graph cycle (densification)
    if closure_hint:
        s += weights.closure_bonus
        
    # Normalize the score, e.g., to a 0-1 range.
    # The max possible score depends on the weights.
    max_score = weights.schema + weights.evidence + weights.consistency + weights.closure_bonus
    
    return min(s / max_score, 1.0) if max_score > 0 else 0.0
