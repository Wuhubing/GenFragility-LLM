#!/usr/bin/env python3
"""
Three-step validation system for knowledge triplets:
1. Whitelist & Type checking
2. Consistency & Conflict detection  
3. Inverse relation auto-completion
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx

from .relations_ontology import (
    KnowledgeTriplet, get_all_relations, is_valid_relation, 
    is_type_compatible, get_inverse_relation, RELATION_CAPS, GLOBAL_SOFT_CAP,
    RelationOntology
)

class ValidationResult:
    """Result of triplet validation with detailed feedback."""
    
    def __init__(self, accept: bool, reason: str = "", 
                 normalized_triplet: Optional[KnowledgeTriplet] = None,
                 inverse_triplet: Optional[KnowledgeTriplet] = None,
                 confidence_adjusted: float = None):
        self.accept = accept
        self.reason = reason
        self.normalized_triplet = normalized_triplet
        self.inverse_triplet = inverse_triplet
        self.confidence_adjusted = confidence_adjusted

class TripletValidator:
    """Comprehensive triplet validation and normalization system."""
    
    def __init__(self, ontology: RelationOntology,
                 confidence_threshold: float = 0.6,
                 candidate_threshold: float = 0.5):
        self.ontology = ontology
        self.confidence_threshold = confidence_threshold
        self.candidate_threshold = candidate_threshold
        
        # Track existing triplets for conflict detection
        self.existing_triplets: Set[Tuple[str, str, str]] = set()
        self.relation_counts: Counter = Counter()
        self.entity_relation_counts: Dict[str, Counter] = defaultdict(Counter)
        
        # Conflict patterns
        self.conflicting_relations = {
            ('StartTime', 'EndTime'),  # Start should be <= End
            ('CapitalOf', 'CapitalOf'),  # One capital per country
        }
        
        # Name normalization patterns
        self.normalization_patterns = [
            (r'\s*\([^)]*\)', ''),  # Remove parentheses content
            (r'\s+', ' '),          # Normalize whitespace
            (r'^["\']|["\']$', ''), # Remove quotes
        ]
        
        # Common aliases and canonical forms
        self.entity_aliases = {
            'US': 'United States',
            'USA': 'United States', 
            'UK': 'United Kingdom',
            'NYC': 'New York City',
            'Beijing': 'Beijing',
            'Peking': 'Beijing',
        }
        
        # Date normalization patterns
        self.date_patterns = [
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3'),  # YYYY-M-D -> YYYY-MM-DD
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # M/D/YYYY -> YYYY-MM-DD
        ]
    
    def validate_and_normalize(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Main validation pipeline: normalize -> whitelist -> type -> consistency -> inverse."""
        
        # Step 1: Normalize relation ID and direction using the new ontology service
        norm_head, norm_rel, norm_tail = self.ontology.normalize_triplet(
            triplet.head, triplet.relation_id, triplet.tail
        )
        # Create a new triplet object with the normalized values to pass through validation
        pre_validated_triplet = KnowledgeTriplet(
            head=norm_head, relation_id=norm_rel, tail=norm_tail,
            domain_guess=triplet.domain_guess, range_guess=triplet.range_guess,
            surface=triplet.surface, evidence=triplet.evidence, confidence=triplet.confidence,
            inverse_auto=triplet.inverse_auto, gen_params=triplet.gen_params
        )

        # Step 2: Whitelist & Type Validation
        whitelist_result = self._validate_whitelist_and_type(pre_validated_triplet)
        if not whitelist_result.accept:
            return whitelist_result
        
        # Step 3: Consistency & Conflict Detection
        consistency_result = self._validate_consistency(pre_validated_triplet)
        if not consistency_result.accept:
            return consistency_result
        
        # Step 4: Normalization of entity names
        normalized_triplet = self._normalize_triplet(pre_validated_triplet)
        
        # Step 5: Check caps after normalization
        caps_result = self._validate_caps(normalized_triplet)
        if not caps_result.accept:
            return caps_result
        
        # Step 6: Generate inverse if needed
        inverse_triplet = self._generate_inverse(normalized_triplet)
        
        return ValidationResult(
            accept=True,
            reason="Validation passed",
            normalized_triplet=normalized_triplet,
            inverse_triplet=inverse_triplet
        )
    
    def _validate_whitelist_and_type(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Step 1: Check relation whitelist and domain/range compatibility."""
        
        # Check if relation is in the canonical list
        if not self.ontology.is_valid_relation(triplet.relation_id):
            return ValidationResult(
                accept=False,
                reason=f"Relation '{triplet.relation_id}' not in canonical list after normalization"
            )
        
        # Check type compatibility (this function needs to be updated or re-implemented)
        # For now, we'll assume a basic check or skip it if it relies on the old structure.
        # TODO: Refactor is_type_compatible to use the new ontology service
        relation_info = self.ontology.get_relation_info(triplet.relation_id)
        if not relation_info:
             return ValidationResult(accept=False, reason="Could not get relation info.")
        
        # A simple domain/range check for now.
        # This part will need the `is_type_compatible` function to be refactored.
        # For now, let's just make sure the keys exist.
        if 'domain' not in relation_info or 'range' not in relation_info:
             return ValidationResult(accept=False, reason="Domain/range not defined in ontology.")

        # Check confidence threshold
        if triplet.confidence < self.candidate_threshold:
            return ValidationResult(
                accept=False,
                reason=f"Confidence {triplet.confidence:.2f} below threshold {self.candidate_threshold}"
            )
        
        return ValidationResult(accept=True, reason="Whitelist and type validation passed")
    
    def _validate_consistency(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Step 2: Check for conflicts and logical consistency."""
        
        # Check for duplicate triplets
        triplet_tuple = (triplet.head, triplet.relation_id, triplet.tail)
        if triplet_tuple in self.existing_triplets:
            return ValidationResult(
                accept=False,
                reason="Duplicate triplet"
            )
        
        # Check for conflicting relations (e.g., multiple capitals)
        if triplet.relation_id == 'CapitalOf':
            # Check if this country already has a capital
            for existing_head, existing_rel, existing_tail in self.existing_triplets:
                if (existing_rel == 'CapitalOf' and existing_tail == triplet.tail and 
                    existing_head != triplet.head):
                    return ValidationResult(
                        accept=False,
                        reason=f"Country {triplet.tail} already has capital {existing_head}"
                    )
            
            # Check if this city is already capital of another country
            for existing_head, existing_rel, existing_tail in self.existing_triplets:
                if (existing_rel == 'CapitalOf' and existing_head == triplet.head and 
                    existing_tail != triplet.tail):
                    return ValidationResult(
                        accept=False,
                        reason=f"City {triplet.head} is already capital of {existing_tail}"
                    )
        
        # Check InstanceOf hierarchy depth to prevent explosion
        if triplet.relation_id == 'InstanceOf':
            hierarchy_result = self._check_hierarchy_depth(triplet)
            if not hierarchy_result.accept:
                return hierarchy_result
        
        # Check Employer temporal conflicts
        if triplet.relation_id == 'Employer':
            employer_result = self._check_employer_conflicts(triplet)
            if not employer_result.accept:
                return employer_result
        
        # Temporal consistency checks
        if triplet.relation_id in ['StartTime', 'EndTime', 'OccursOn']:
            temporal_result = self._validate_temporal_consistency(triplet)
            if not temporal_result.accept:
                return temporal_result
        
        return ValidationResult(accept=True, reason="Consistency validation passed")
    
    def _validate_temporal_consistency(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Enhanced temporal validation with full date support."""
        
        def parse_date(date_str: str) -> Optional[int]:
            """Parse date to comparable integer (YYYYMMDD)"""
            # Try YYYY-MM-DD format
            match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
            if match:
                y, m, d = match.groups()
                return int(f"{y}{int(m):02d}{int(d):02d}")
            
            # Try YYYY format
            match = re.search(r'(\d{4})', date_str)
            if match:
                return int(match.group(1)) * 10000  # YYYY0000
            
            return None
        
        current_date = parse_date(triplet.tail)
        if current_date is None:
            return ValidationResult(accept=True, reason="No valid date found")
        
        # Collect all temporal relations for same entity
        entity_times = {'start': [], 'end': [], 'occurs': []}
        
        for existing_head, existing_rel, existing_tail in self.existing_triplets:
            if existing_head != triplet.head:
                continue
            
            existing_date = parse_date(existing_tail)
            if existing_date is None:
                continue
            
            if existing_rel == 'StartTime':
                entity_times['start'].append(existing_date)
            elif existing_rel == 'EndTime':
                entity_times['end'].append(existing_date)
            elif existing_rel == 'OccursOn':
                entity_times['occurs'].append(existing_date)
        
        # Apply temporal logic rules
        if triplet.relation_id == 'StartTime':
            # Start should be <= all end times
            for end_date in entity_times['end']:
                if current_date > end_date:
                    return ValidationResult(
                        accept=False,
                        reason=f"StartTime {current_date} after EndTime {end_date}"
                    )
        
        elif triplet.relation_id == 'EndTime':
            # End should be >= all start times
            for start_date in entity_times['start']:
                if current_date < start_date:
                    return ValidationResult(
                        accept=False,
                        reason=f"EndTime {current_date} before StartTime {start_date}"
                    )
        
        elif triplet.relation_id == 'OccursOn':
            # Occurs should be within start-end window
            if entity_times['start'] and current_date < min(entity_times['start']):
                return ValidationResult(
                    accept=False,
                    reason=f"OccursOn {current_date} before StartTime {min(entity_times['start'])}"
                )
            if entity_times['end'] and current_date > max(entity_times['end']):
                return ValidationResult(
                    accept=False,
                    reason=f"OccursOn {current_date} after EndTime {max(entity_times['end'])}"
                )
        
        return ValidationResult(accept=True, reason="Temporal consistency validated")
    
    def _check_hierarchy_depth(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Prevent InstanceOf chains longer than 2 levels."""
        if triplet.relation_id != 'InstanceOf':
            return ValidationResult(accept=True, reason="Not hierarchy relation")
        
        # Find chain length from this entity
        current = triplet.tail  # Start from the class
        depth = 1
        
        while depth < 3:  # Max check depth
            # Look for X InstanceOf current
            found_parent = False
            for h, r, t in self.existing_triplets:
                if h == current and r == 'InstanceOf':
                    current = t
                    depth += 1
                    found_parent = True
                    break
            
            if not found_parent:
                break
        
        if depth >= 3:
            return ValidationResult(
                accept=False,
                reason=f"InstanceOf chain would exceed max depth of 2: {depth}"
            )
        
        return ValidationResult(accept=True, reason="Hierarchy depth OK")
    
    def _check_employer_conflicts(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Check for temporal employer conflicts (same person, same time period)."""
        if triplet.relation_id != 'Employer':
            return ValidationResult(accept=True, reason="Not employer relation")
        
        # For now, implement basic check - could be enhanced with actual time periods
        person = triplet.head
        new_employer = triplet.tail
        
        # Check if person already has an employer
        for existing_head, existing_rel, existing_tail in self.existing_triplets:
            if (existing_head == person and existing_rel == 'Employer' and 
                existing_tail != new_employer):
                # In future, add temporal logic here to check if time periods overlap
                return ValidationResult(
                    accept=False,
                    reason=f"Person {person} already employed by {existing_tail}"
                )
        
        return ValidationResult(accept=True, reason="No employer conflicts")
    
    def _validate_caps(self, triplet: KnowledgeTriplet) -> ValidationResult:
        """Check per-entity and global relation caps."""
        
        # Check per-entity caps
        relation_cap = RELATION_CAPS.get(triplet.relation_id, RELATION_CAPS['*'])
        current_count = self.entity_relation_counts[triplet.head][triplet.relation_id]
        
        if current_count >= relation_cap:
            return ValidationResult(
                accept=False,
                reason=f"Entity {triplet.head} exceeds cap {relation_cap} for relation {triplet.relation_id}"
            )
        
        # Check global soft cap
        total_relations = sum(self.relation_counts.values())
        if total_relations > 0:
            current_proportion = self.relation_counts[triplet.relation_id] / total_relations
            if current_proportion > GLOBAL_SOFT_CAP:
                return ValidationResult(
                    accept=False,
                    reason=f"Relation {triplet.relation_id} exceeds global soft cap {GLOBAL_SOFT_CAP:.1%}"
                )
        
        return ValidationResult(accept=True, reason="Caps validation passed")
    
    def _normalize_triplet(self, triplet: KnowledgeTriplet) -> KnowledgeTriplet:
        """Step 3: Normalize entity names and values."""
        
        normalized_head = self._normalize_entity_name(triplet.head)
        normalized_tail = self._normalize_entity_name(triplet.tail)
        
        # Special normalization for time values
        relation_info = self.ontology.get_relation_info(triplet.relation_id)
        if relation_info and relation_info.get('group') == 'Temporal':
            normalized_tail = self._normalize_date(normalized_tail)
        
        # Create normalized triplet
        normalized = KnowledgeTriplet(
            head=normalized_head,
            relation_id=triplet.relation_id,
            tail=normalized_tail,
            domain_guess=triplet.domain_guess,
            range_guess=triplet.range_guess,
            surface=triplet.surface,
            evidence=triplet.evidence,
            confidence=triplet.confidence,
            inverse_auto=triplet.inverse_auto,
            gen_params=triplet.gen_params
        )
        
        return normalized
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity names for consistency."""
        
        normalized = name.strip()
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        normalized = normalized.strip()
        
        # Apply aliases
        if normalized in self.entity_aliases:
            normalized = self.entity_aliases[normalized]
        
        # Title case for proper nouns (basic heuristic)
        if len(normalized.split()) <= 3 and not any(c.islower() for c in normalized):
            normalized = normalized.title()
        
        return normalized
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date strings to consistent format."""
        
        normalized = date_str.strip()
        
        # Apply date patterns
        for pattern, replacement in self.date_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def _generate_inverse(self, triplet: KnowledgeTriplet) -> Optional[KnowledgeTriplet]:
        """Step 4: Generate inverse relation if defined and auto-enabled."""
        
        if not triplet.inverse_auto:
            return None
        
        inverse_relation = self.ontology.get_inverse(triplet.relation_id)
        if not inverse_relation:
            return None
        
        # Check if inverse already exists
        inverse_tuple = (triplet.tail, inverse_relation, triplet.head)
        if inverse_tuple in self.existing_triplets:
            return None
        
        # Create inverse triplet
        inverse_triplet = KnowledgeTriplet(
            head=triplet.tail,
            relation_id=inverse_relation,
            tail=triplet.head,
            domain_guess=triplet.range_guess,
            range_guess=triplet.domain_guess,
            surface=f"Inverse of: {triplet.surface}",
            evidence=f"Auto-generated inverse of ({triplet.head}, {triplet.relation_id}, {triplet.tail})",
            confidence=triplet.confidence * 0.95,  # Slightly lower confidence for auto-generated
            inverse_auto=False,  # Don't generate inverse of inverse
            gen_params=triplet.gen_params
        )
        
        return inverse_triplet
    
    def add_validated_triplet(self, triplet: KnowledgeTriplet):
        """Add a validated triplet to tracking structures."""
        triplet_tuple = (triplet.head, triplet.relation_id, triplet.tail)
        self.existing_triplets.add(triplet_tuple)
        self.relation_counts[triplet.relation_id] += 1
        self.entity_relation_counts[triplet.head][triplet.relation_id] += 1
    
    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        total_triplets = len(self.existing_triplets)
        
        stats = {
            'total_triplets': total_triplets,
            'unique_entities': len(self.entity_relation_counts),
            'relation_distribution': dict(self.relation_counts),
            'most_connected_entities': [],
            'relation_diversity_entropy': 0.0
        }
        
        if total_triplets > 0:
            # Calculate relation diversity entropy
            import math
            relation_probs = [count/total_triplets for count in self.relation_counts.values()]
            stats['relation_diversity_entropy'] = -sum(p * math.log2(p) for p in relation_probs if p > 0)
            
            # Find most connected entities
            entity_degrees = {entity: sum(counts.values()) 
                            for entity, counts in self.entity_relation_counts.items()}
            stats['most_connected_entities'] = sorted(entity_degrees.items(), 
                                                    key=lambda x: x[1], reverse=True)[:10]
        
        return stats
    
    def reset(self):
        """Reset validator state for fresh validation."""
        self.existing_triplets.clear()
        self.relation_counts.clear()
        self.entity_relation_counts.clear()

# Utility functions for batch validation
def validate_triplet_batch(triplets: List[KnowledgeTriplet], 
                          validator: TripletValidator = None) -> List[ValidationResult]:
    """Validate a batch of triplets."""
    if validator is None:
        # This will fail if a validator is not provided, which is the correct behavior now.
        # The caller (e.g., graph_builder) MUST instantiate the validator with the ontology.
        ontology = RelationOntology()
        validator = TripletValidator(ontology)
    
    results = []
    for triplet in triplets:
        result = validator.validate_and_normalize(triplet)
        results.append(result)
        
        # Add successful triplets to validator state
        if result.accept:
            validator.add_validated_triplet(result.normalized_triplet)
            if result.inverse_triplet:
                validator.add_validated_triplet(result.inverse_triplet)
    
    return results

def filter_valid_triplets(triplets: List[KnowledgeTriplet],
                         validator: TripletValidator = None) -> Tuple[List[KnowledgeTriplet], List[str]]:
    """Filter triplets, returning valid ones and rejection reasons."""
    if validator is None:
        ontology = RelationOntology()
        validator = TripletValidator(ontology)
        
    results = validate_triplet_batch(triplets, validator)
    
    valid_triplets = []
    rejection_reasons = []
    
    for triplet, result in zip(triplets, results):
        if result.accept:
            valid_triplets.append(result.normalized_triplet)
            if result.inverse_triplet:
                valid_triplets.append(result.inverse_triplet)
        else:
            rejection_reasons.append(f"{triplet.head}-{triplet.relation_id}-{triplet.tail}: {result.reason}")
    
    return valid_triplets, rejection_reasons

if __name__ == "__main__":
    # Test validation system
    ontology = RelationOntology()
    validator = TripletValidator(ontology)
    
    # Test valid triplet
    triplet1 = KnowledgeTriplet('Beijing', 'CapitalOf', 'China',
                               domain_guess='City', range_guess='Country',
                               confidence=0.95)
    result1 = validator.validate_and_normalize(triplet1)
    print(f"Valid triplet: {result1.accept} - {result1.reason}")
    
    if result1.accept:
        validator.add_validated_triplet(result1.normalized_triplet)
        if result1.inverse_triplet:
            print(f"Inverse generated: {result1.inverse_triplet.to_tuple()}")
            validator.add_validated_triplet(result1.inverse_triplet)
    
    # Test invalid relation
    triplet2 = KnowledgeTriplet('Beijing', 'InvalidRelation', 'China',
                               confidence=0.95)
    result2 = validator.validate_and_normalize(triplet2)
    print(f"Invalid relation: {result2.accept} - {result2.reason}")
    
    # Test duplicate
    triplet3 = KnowledgeTriplet('Beijing', 'CapitalOf', 'China',
                               domain_guess='City', range_guess='Country', 
                               confidence=0.95)
    result3 = validator.validate_and_normalize(triplet3)
    print(f"Duplicate: {result3.accept} - {result3.reason}")
    
    # Test normalization of alias
    triplet_alias = KnowledgeTriplet('PersonA', 'WorksAt', 'OrgB',
                                     domain_guess='Person', range_guess='Org',
                                     confidence=0.9)
    result_alias = validator.validate_and_normalize(triplet_alias)
    print(f"Alias 'WorksAt': {result_alias.accept} - Normalized to: {result_alias.normalized_triplet.to_tuple() if result_alias.accept else 'N/A'}")
    if result_alias.accept:
        validator.add_validated_triplet(result_alias.normalized_triplet)

    # Test normalization of inverse that swaps
    triplet_swap = KnowledgeTriplet('Ordering', 'SubEventOf', 'Dining',
                                     domain_guess='Event', range_guess='Event',
                                     confidence=0.9)
    result_swap = validator.validate_and_normalize(triplet_swap)
    print(f"Inverse 'SubEventOf': {result_swap.accept} - Normalized to: {result_swap.normalized_triplet.to_tuple() if result_swap.accept else 'N/A'}")
    if result_swap.accept:
        validator.add_validated_triplet(result_swap.normalized_triplet)

    # Test low confidence
    triplet4 = KnowledgeTriplet('Shanghai', 'CapitalOf', 'China',
                               domain_guess='City', range_guess='Country',
                               confidence=0.3)
    result4 = validator.validate_and_normalize(triplet4)
    print(f"Low confidence: {result4.accept} - {result4.reason}")
    
    print(f"\nValidator statistics: {validator.get_statistics()}")
