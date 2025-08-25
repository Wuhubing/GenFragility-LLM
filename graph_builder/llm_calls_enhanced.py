#!/usr/bin/env python3
"""
Enhanced LLM calls for knowledge graph construction with:
- Whitelist relation constraint
- Structured metadata output  
- Response caching for reproducibility
- Conservative temperature settings
"""

from openai import OpenAI
import json
import hashlib
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .relations_ontology import KnowledgeTriplet, RelationOntology

class LLMInterfaceEnhanced:
    """Enhanced LLM interface with ontology integration."""
    
    def __init__(self, api_key_path: str = None, cache_dir: str = None, ontology: RelationOntology = None):
        self.ontology = ontology or RelationOntology()
        self.cache_dir = cache_dir or CACHE_DIR
        if api_key_path:
            self.initialize_api(api_key_path)
    
    def initialize_api(self, api_key_path: str = 'keys/openai.txt') -> bool:
        """Initialize the LLM API."""
        return load_api_key(api_key_path)
    
    def get_relations_by_group(self, group: str) -> List[str]:
        """Get relations by group from the ontology."""
        relations = []
        for rel_id, rel_info in self.ontology.get_all_relations().items():
            if rel_info.get('group') == group:
                relations.append(rel_id)
        return relations
    
    def get_relation_examples(self, relation_id: str) -> List[str]:
        """Get examples for a relation (simplified implementation)."""
        # This is a placeholder - you might want to implement actual examples
        return [f"Example usage of {relation_id}"]
    
    def generate_triplets(self, prompt: str, num_triplets: int = 8) -> List[KnowledgeTriplet]:
        """Generate triplets using downstream expansion."""
        # Extract entity from prompt (improved heuristic)
        entity = self._extract_entity_from_prompt(prompt)
        if entity:
            print(f"🔍 Generating triplets for entity: '{entity}'")
            triplets = find_downstream_triplets_enhanced(entity, num_triplets)
            print(f"📊 Generated {len(triplets)} triplets for '{entity}'")
            return triplets
        else:
            print(f"❌ Could not extract entity from prompt: {prompt[:100]}...")
            return []
    
    def _extract_entity_from_prompt(self, prompt: str) -> str:
        """Extract entity name from expansion prompt."""
        # Try multiple patterns to extract entity name
        import re
        
        # Pattern 1: "Given the entity 'ENTITY'"
        match = re.search(r"Given the entity '([^']+)'", prompt)
        if match:
            return match.group(1)
        
        # Pattern 2: Any quoted entity
        quoted_match = re.search(r"'([^']+)'", prompt)
        if quoted_match:
            return quoted_match.group(1)
        
        # Pattern 3: "facts about ENTITY"
        if "facts about" in prompt:
            parts = prompt.split("facts about ")
            if len(parts) > 1:
                entity_part = parts[1].split(".")[0].split("\n")[0].strip()
                return entity_part.strip("'\"")
            
        # Pattern 4: Extract from context (fallback)
        lines = prompt.split('\n')
        for line in lines:
            if 'Focus on relations' in line:
                continue
            if len(line.strip()) > 0 and not line.startswith('Please'):
                # Try to find entity names in the line
                words = line.split()
                for word in words:
                    if len(word) > 2 and word[0].isupper():
                        return word.strip('.,!?')
        
        return ""

# Global client variable
client = None

# Global ontology instance
_global_ontology = None

# Response cache for reproducibility
CACHE_DIR = 'cache/llm_responses'
response_cache = {}

def _get_ontology():
    """Get the global ontology instance."""
    global _global_ontology
    if _global_ontology is None:
        _global_ontology = RelationOntology()
    return _global_ontology

def get_relations_by_group(group: str, include_optional: bool = True) -> List[str]:
    """Get relations by group from the ontology."""
    ontology = _get_ontology()
    relations = []
    for rel_id, rel_info in ontology.get_all_relations().items():
        if rel_info.get('group') == group:
            if include_optional or rel_info.get('group') != 'Optional':
                relations.append(rel_id)
    return relations

def get_relation_examples(relation_id: str, include_optional: bool = True) -> List[str]:
    """Get examples for a relation (simplified implementation)."""
    # This is a placeholder - you might want to implement actual examples
    return [f"Example usage of {relation_id}"]

def get_all_relations(include_optional: bool = True) -> Dict:
    """Get all relations from the ontology."""
    ontology = _get_ontology()
    all_relations = ontology.get_all_relations()
    if not include_optional:
        return {k: v for k, v in all_relations.items() if v.get('group') != 'Optional'}
    return all_relations

def load_api_key(filepath: str = 'keys/openai.txt'):
    """Load OpenAI API key from a file and initialize the client."""
    global client
    try:
        with open(filepath, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        
        # Initialize cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        _load_cache()
        
        return True
    except FileNotFoundError:
        print(f"Error: API key file not found at '{filepath}'.")
        return False
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return False

def _load_cache():
    """Load existing response cache from disk."""
    global response_cache
    cache_file = os.path.join(CACHE_DIR, 'response_cache.json')
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                response_cache = json.load(f)
            print(f"Loaded {len(response_cache)} cached responses")
    except Exception as e:
        print(f"Error loading cache: {e}")
        response_cache = {}

def _save_cache():
    """Save response cache to disk."""
    cache_file = os.path.join(CACHE_DIR, 'response_cache.json')
    try:
        with open(cache_file, 'w') as f:
            json.dump(response_cache, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

def _get_cache_key(prompt: str, model: str, temperature: float) -> str:
    """Generate cache key for prompt."""
    content = f"{model}|{temperature}|{prompt}"
    return hashlib.md5(content.encode()).hexdigest()

def _call_llm_with_cache(prompt: str, system_prompt: str, model: str = "gpt-4o-mini", 
                        temperature: float = 0.2, max_tokens: int = 2000,
                        response_format: Dict = None) -> Optional[str]:
    """Call LLM with caching support."""
    global client, response_cache
    
    if client is None:
        print("Error: OpenAI client not initialized. Please call load_api_key() first.")
        return None
    
    # Check cache first
    cache_key = _get_cache_key(prompt, model, temperature)
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        
        # Cache the response
        response_cache[cache_key] = content
        
        # Periodically save cache
        if len(response_cache) % 10 == 0:
            _save_cache()
        
        return content
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def _parse_enhanced_triplets(content: str, include_optional: bool = False) -> List[KnowledgeTriplet]:
    """Parse enhanced triplet response with metadata."""
    if not content:
        return []
    
    try:
        response_data = json.loads(content)
        triplets = []
        
        for item in response_data.get("triplets", []):
            if isinstance(item, dict):
                # Check for skip flag
                if item.get("skip", False):
                    continue
                
                head = item.get("head") or item.get("subject")
                relation_id = item.get("relation_id") or item.get("relation")
                tail = item.get("tail") or item.get("object")
                
                if head and relation_id and tail:
                    triplet = KnowledgeTriplet(
                        head=str(head),
                        relation_id=str(relation_id),
                        tail=str(tail),
                        domain_guess=item.get("domain_guess", "Entity"),
                        range_guess=item.get("range_guess", "Entity"),
                        surface=item.get("surface", ""),
                        evidence=item.get("evidence", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        inverse_auto=item.get("inverse_auto", True),
                        gen_params={
                            "model": "gpt-4o-mini",
                            "temperature": 0.2,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    triplets.append(triplet)
        
        return triplets
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error parsing enhanced triplets: {e}")
        return []

def find_downstream_triplets_enhanced(entity: str, num_triplets: int = 8, 
                                    target_groups: List[str] = None,
                                    include_optional: bool = False) -> List[KnowledgeTriplet]:
    """Find downstream triplets with enhanced metadata and relation constraints."""
    
    all_relations = get_all_relations(include_optional)
    relation_list = list(all_relations.keys())
    
    # Filter by target groups if specified
    if target_groups:
        filtered_relations = []
        for group in target_groups:
            filtered_relations.extend(get_relations_by_group(group, include_optional))
        relation_list = filtered_relations
    
    # Get some examples for context
    example_relations = relation_list[:6]  # Show first 6 as examples
    examples_text = ""
    for rel in example_relations:
        rel_examples = get_relation_examples(rel, include_optional)
        if rel_examples:
            example = rel_examples[0]  # Take first example
            examples_text += f'    - "{rel}": {example}\n'
    
    prompt = f"""
Please provide {num_triplets} knowledge triplets where the subject (head) is EXACTLY '{entity}'.

CRITICAL CONSTRAINTS:
1. MANDATORY: relation_id must be from this exact list: {relation_list}
2. MANDATORY: subject must be exactly '{entity}' (no variations, synonyms, or related entities)
3. MANDATORY: confidence must be realistic (0.6+ for certain facts, 0.5-0.6 for uncertain)
4. MANDATORY: domain_guess/range_guess from: Person, Org, Place, City, Country, Region, Class, Entity, Group, Time, Material, Work, Event, PropertyValue, Purpose, Action, Occupation, Genre, Language, Product, Software
5. FORBIDDEN: Generic or vague relations, made-up relation names
6. QUALITY: Include specific evidence/source for each claim

If you cannot provide a high-confidence, specific triplet, set "skip": true.
Do not generate low-quality or uncertain triplets just to fill quota.

Examples of valid relations:
{examples_text}

OUTPUT FORMAT (JSON):
{{
  "triplets": [
    {{
      "head": "{entity}",
      "relation_id": "ExactRelationFromList",
      "tail": "SpecificEntity",
      "domain_guess": "EntityType",
      "range_guess": "EntityType",
      "surface": "Natural language statement",
      "evidence": "Specific source or reasoning",
      "confidence": 0.85,
      "inverse_auto": true,
      "skip": false
    }}
  ]
}}

Focus on factually verifiable, diverse relation types about '{entity}'.
"""

    system_prompt = "You are a knowledge graph expert. Generate only factually accurate triplets using the provided relation vocabulary. Be conservative with confidence scores."
    
    content = _call_llm_with_cache(prompt, system_prompt, temperature=0.2, 
                                  response_format={"type": "json_object"})
    
    return _parse_enhanced_triplets(content, include_optional)

def find_upstream_triplets_enhanced(entity: str, num_triplets: int = 8,
                                  target_groups: List[str] = None, 
                                  include_optional: bool = False) -> List[KnowledgeTriplet]:
    """Find upstream triplets where entity is the object."""
    
    all_relations = get_all_relations(include_optional)
    relation_list = list(all_relations.keys())
    
    # Filter by target groups if specified
    if target_groups:
        filtered_relations = []
        for group in target_groups:
            filtered_relations.extend(get_relations_by_group(group, include_optional))
        relation_list = filtered_relations
    
    # Get some examples for context
    example_relations = relation_list[:6]
    examples_text = ""
    for rel in example_relations:
        rel_examples = get_relation_examples(rel, include_optional)
        if rel_examples:
            example = rel_examples[0]
            examples_text += f'    - "{rel}": {example}\n'
    
    prompt = f"""
Please provide {num_triplets} knowledge triplets where the object (tail) is EXACTLY '{entity}'.

STRICT REQUIREMENTS:
1. You MUST use only these relation_ids: {relation_list}
2. The object of each triplet MUST be exactly '{entity}' (no variations)
3. Provide domain_guess and range_guess from these types: Person, Org, Place, City, Country, Region, Class, Entity, Group, Time, Material, Work, Event, PropertyValue, Purpose, Action, Occupation, Genre, Language, Product, Software
4. Include confidence score (0.0-1.0) and evidence/justification
5. If you cannot provide a high-quality triplet, set "skip": true

Examples of valid relations:
{examples_text}

Format your response as JSON:
{{
  "triplets": [
    {{
      "head": "SourceEntity",
      "relation_id": "ValidRelationFromList", 
      "tail": "{entity}",
      "domain_guess": "EntityType",
      "range_guess": "EntityType",
      "surface": "Natural language statement",
      "evidence": "Brief justification or source", 
      "confidence": 0.85,
      "inverse_auto": true,
      "skip": false
    }}
  ]
}}

Focus on diverse relation types and entities that relate TO '{entity}'.
"""

    system_prompt = "You are a knowledge graph expert. Generate only factually accurate triplets using the provided relation vocabulary. Be conservative with confidence scores."
    
    content = _call_llm_with_cache(prompt, system_prompt, temperature=0.2,
                                  response_format={"type": "json_object"})
    
    return _parse_enhanced_triplets(content, include_optional)

def find_parallel_triplets_enhanced(relation_id: str, num_triplets: int = 8,
                                   domain_diversity: bool = True,
                                   include_optional: bool = False) -> List[KnowledgeTriplet]:
    """Find parallel triplets using the same relation across different domains."""
    
    all_relations = get_all_relations(include_optional)
    if relation_id not in all_relations:
        print(f"Error: Relation '{relation_id}' not found in ontology")
        return []
    
    relation_info = all_relations[relation_id]
    examples = get_relation_examples(relation_id, include_optional)
    
    examples_text = ""
    if examples:
        for i, example in enumerate(examples[:3]):  # Show up to 3 examples
            examples_text += f'    Example {i+1}: {example}\n'
    
    diversity_instruction = ""
    if domain_diversity:
        diversity_instruction = """
DIVERSITY REQUIREMENT: Generate triplets from different domains/categories.
- Include entities from different types: Person, Organization, Place, Work, Event, etc.
- Avoid multiple triplets about the same entity or very similar entities.
- Aim for broad coverage across knowledge domains.
"""
    
    diversity_instruction = ""
    if domain_diversity:
        diversity_instruction = """

DOMAIN DIVERSITY REQUIREMENT:
- Must include entities from at least 3 different types
- Avoid multiple triplets about the same entity or very similar entities
- Prioritize factual diversity over quantity
- Examples should span: Person/Org/Place/Work/Event/Product domains"""

    prompt = f"""
Please provide {num_triplets} diverse knowledge triplets using relation '{relation_id}'.

RELATION INFO:
- Group: {relation_info['group']}
- Domain types: {relation_info['domain']}
- Range types: {relation_info['range']}
- Description: {relation_info['description']}

{examples_text}

CRITICAL CONSTRAINTS:
1. ALL triplets must use relation_id: "{relation_id}"
2. Include entities from at least 3 different domain types
3. No duplicate or highly similar entities
4. Span different knowledge domains (geography, people, organizations, etc.)
5. Each triplet must be independently verifiable
6. Confidence must be realistic (0.6+ for certain facts)
7. If uncertain, set "skip": true{diversity_instruction}

Format your response as JSON:
{{
  "triplets": [
    {{
      "head": "SubjectEntity",
      "relation_id": "{relation_id}",
      "tail": "ObjectEntity", 
      "domain_guess": "EntityType",
      "range_guess": "EntityType",
      "surface": "Natural language statement",
      "evidence": "Specific source or reasoning",
      "confidence": 0.85,
      "inverse_auto": true,
      "skip": false
    }}
  ]
}}

Generate diverse, factually verified examples of the '{relation_id}' relation.
"""

    system_prompt = f"You are a knowledge graph expert specializing in the '{relation_id}' relation. Generate only factually accurate, diverse examples. Be conservative with confidence scores."
    
    content = _call_llm_with_cache(prompt, system_prompt, temperature=0.2,
                                  response_format={"type": "json_object"})
    
    return _parse_enhanced_triplets(content, include_optional)

def triplet_to_question_enhanced(triplet: KnowledgeTriplet) -> str:
    """Convert a triplet to a natural language question with caching."""
    global client
    if client is None:
        print("Error: OpenAI client not initialized. Please call load_api_key() first.")
        return f"What is the {triplet.relation_id} of {triplet.head}?"
    
    prompt = f"""
Based on the knowledge triplet:
- Subject: {triplet.head}
- Relation: {triplet.relation_id} 
- Object: {triplet.tail}
- Surface form: {triplet.surface}

Generate a clear, natural question in English whose answer is exactly "{triplet.tail}".

Requirements:
- The question should be unambiguous and specific
- The expected answer should be exactly "{triplet.tail}"
- Use natural, conversational language
- Return only the question, no additional text

Examples:
- For (Paris, CapitalOf, France): "What is the capital of France?"
- For (Einstein, Occupation, Physicist): "What was Einstein's profession?"
"""

    system_prompt = "You are an expert in generating clear, unambiguous questions from structured knowledge. Focus on precision and natural language."
    
    content = _call_llm_with_cache(prompt, system_prompt, temperature=0.1, max_tokens=100)
    
    if content:
        question = content.strip().strip('"')
        if not question.endswith('?'):
            question += '?'
        return question
    else:
        # Fallback question
        return f"What is the {triplet.relation_id} of {triplet.head}?"

def get_cache_statistics() -> Dict:
    """Get statistics about the response cache."""
    global response_cache
    return {
        'total_cached_responses': len(response_cache),
        'cache_directory': CACHE_DIR,
        'cache_file_exists': os.path.exists(os.path.join(CACHE_DIR, 'response_cache.json'))
    }

def clear_cache():
    """Clear the response cache."""
    global response_cache
    response_cache.clear()
    cache_file = os.path.join(CACHE_DIR, 'response_cache.json')
    if os.path.exists(cache_file):
        os.remove(cache_file)
    print("Response cache cleared.")

if __name__ == "__main__":
    # Test the enhanced LLM functions
    if load_api_key():
        print("API key loaded successfully!")
        print(f"Cache stats: {get_cache_statistics()}")
        
        print("\nTesting enhanced downstream triplets for 'Beijing':")
        downstream = find_downstream_triplets_enhanced("Beijing", 3)
        for triplet in downstream:
            print(f"  {triplet.to_tuple()} (conf: {triplet.confidence:.2f})")
            print(f"    Surface: {triplet.surface}")
            print(f"    Evidence: {triplet.evidence}")
        
        print("\nTesting enhanced upstream triplets for 'China':")
        upstream = find_upstream_triplets_enhanced("China", 3)
        for triplet in upstream:
            print(f"  {triplet.to_tuple()} (conf: {triplet.confidence:.2f})")
        
        print("\nTesting enhanced parallel triplets for 'CapitalOf':")
        parallel = find_parallel_triplets_enhanced("CapitalOf", 3)
        for triplet in parallel:
            print(f"  {triplet.to_tuple()} (conf: {triplet.confidence:.2f})")
        
        if parallel:
            print("\nTesting question generation:")
            test_triplet = parallel[0]
            question = triplet_to_question_enhanced(test_triplet)
            print(f"  Triplet: {test_triplet.to_tuple()}")
            print(f"  Question: {question}")
        
        # Save cache before exit
        _save_cache()
        print(f"\nFinal cache stats: {get_cache_statistics()}")
    else:
        print("Failed to initialize. Please check API key file.")
