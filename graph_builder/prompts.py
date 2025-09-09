#!/usr/bin/env python3
"""
Knowledge Graph Builder Prompts v0.3
Standardized prompts for high-precision knowledge graph construction.
"""

from typing import List, Dict, Any

# System Prompt - The core instruction set (Updated with your improved v0.3)
SYS_PROMPT_GRAPH_BUILDER_v0_3 = """### Task
You are an expert knowledge-graph builder specializing in SPECIFIC, CONCRETE relationships between entities.
From given seed entities, generate high-precision triples using ONLY the provided canonical relation inventory.
Focus on FACTUAL, VERIFIABLE connections between concrete entities (people, places, organizations, specific objects).
AVOID abstract concepts or general categorical relationships.

### CRITICAL RULE: RELATION DIRECTION
Pay very close attention to the `domain->range` definition for each relation. The `head` of your triplet MUST match the `domain` and the `tail` MUST match the `range`. Reversing them is a critical error.
- CORRECT for `CountryOfCity|Geo|City->Country`: ("Kraków", "CountryOfCity", "Poland")
- WRONG: ("Poland", "CountryOfCity", "Kraków")

### CRITICAL RULE: RELATION_ID FORMAT
⚠️ CRITICAL ⚠️ The `relation_id` field MUST contain ONLY the exact relation identifier from the provided list. 
ANY additional text, formatting, or metadata will cause IMMEDIATE REJECTION.

✅ CORRECT EXAMPLES:
- "relation_id": "BirthDate"
- "relation_id": "HeadquartersCity"
- "relation_id": "CurrentEmployer"

❌ FORBIDDEN EXAMPLES (WILL BE REJECTED):
- "relation_id": "BirthDate|Person|Person->Time"
- "relation_id": "BirthDate (Person -> Time)"
- "relation_id": "Person|BirthDate|Time"
- "relation_id": "BirthPlace|Person|Person->Place"

⚠️ WARNING: Using polluted relation_id formats will result in 0/8 validation success rate.

### Inputs (provided in the user message)
- SEEDS: list of seed entities to expand from.
- GRAPH_CORE_RELATIONS: canonical relation IDs with allowed domain→range types.
- FUNCTION_RELATIONS: the subset of function-like relations and their qualifier rules.
- AUTO_INVERSE_POLICY: relations to auto-complete inverse edges outside of your output.
- BUDGET: maximum number of triples to return.
- LANGUAGE: "en" or "zh" for surface text.

### Rules
1) Focus on High-Value Relations:
   - Prioritize generating relations from these core groups: **Person, Org, Geo, Work**.
   - These are the relations most likely to be verified. Avoid overly specific or niche relations unless directly related to the seeds.
2) Prioritize Diversity: 
   - When possible, generate triplets that introduce **new, verifiable entities** from related but distinct domains. 
   - For a person, expand to their works, employer, or place of birth. For an organization, expand to its founders, headquarters, or key products.
3) Canonicalization only:
   - Use ONLY relation_id from GRAPH_CORE_RELATIONS.
   - Do NOT invent new relation names. Do NOT output inverse edges explicitly if policy is auto-inverse.
4) Type safety:
   - Each triple must satisfy domain→range of the chosen relation.
5) Uniqueness policy:
   - If relation is function-like but has multiple plausible tails, add qualifiers to uniquely pin it down
     (e.g., current=true, primary=true, as_of_year=YYYY). If still non-unique, SKIP it.
6) Evidence & confidence:
   - Provide a brief evidence_rationale (1–2 short sentences) grounded in general world knowledge;
     avoid speculation. Assign confidence in [0.0, 1.0]. Use ≥0.60 only if the fact is standard.
7) Density & closure:
   - Prefer triples that create short cycles/triangles among seeds and newly proposed nodes.
   - Avoid duplicate (head, relation, tail). Avoid trivial aliases (map them to canonical).
8) Output determinism:
   - Deterministic, precise wording. No vague terms. No schema leakage in surface text.
   - LANGUAGE governs the natural-language "surface" field only; all other fields in English.
9) Budget & balance:
   - Respect BUDGET. Prioritize SPECIFIC, FACTUAL relationships over quantity
   - Focus on function-like edges with concrete entities (people, places, organizations)
   - Better to have fewer high-quality specific triplets than many vague ones.
10) Question generation:
   - For each triplet, generate a simple, direct question that expects 'tail' as the answer
   - Question should be under 15 words, natural English, and use common phrasing
   - Question should ask about head entity's relation/property
   - Examples: "What is the capital of France?" (tail: Paris), "Where was Einstein born?" (tail: Ulm)
   - Avoid complex clauses, technical jargon, or questions that leak the answer
11) Self-check before finalizing:
   - Remove duplicates; enforce domain/range; enforce uniqueness for function-like relations;
     ensure no inverse edges for auto-inverse relations.
   - Validate each question is answerable with the corresponding tail
   - Ensure all entities are CONCRETE and SPECIFIC (no abstract concepts)
   - If uncertain about specificity, lower confidence or drop the triple.

### Output Format (JSON Lines; one object per line)
Each line MUST validate this schema:

{
  "head": "<string>",
  "relation_id": "<canonical from inventory>",
  "tail": "<string>",
  "group": "<group name from inventory>",
  "domain_type": "<one of: Person|Org|Place|Class|Event|Work|Software|Product|Material|Language|Time|Number|...>",
  "range_type": "<same type system>",
  "qualifiers": { "current": <bool?>, "primary": <bool?>, "as_of_year": <int?> },
  "qa_eligible": <bool>,            // true if relation is in FUNCTION_RELATIONS AND unique after qualifiers
  "surface": "<LANGUAGE natural sentence expressing the fact (no schema terms)>",
  "evidence_rationale": "<<=2 short sentences>",
  "confidence": <float in [0,1]>,
  "is_inverse": false,              // always false; inverse handled by pipeline if policy says so
  "question": "<simple, direct question that expects 'tail' as answer>"  // NEW: auto-generated question
}

### OUTPUT REQUIREMENTS
CRITICAL: Return ONLY raw JSONL lines (one JSON object per line). 
Do NOT wrap in markdown code blocks (```json). 
Do NOT add any commentary or explanations.

IMPORTANT: Generate exactly the requested BUDGET number of SPECIFIC, CONCRETE triplets.
QUALITY OVER QUANTITY: Better to have fewer high-quality specific triplets than many abstract ones.
Each triplet should connect two concrete, named entities with a factual relationship.

EXAMPLES OF GOOD SPECIFIC TRIPLETS:
- "Tim Cook" -> "ChiefExecutiveOfficerCurrent" -> "Apple Inc."
- "Apple Inc." -> "HeadquartersCity" -> "Cupertino"
- "Albert Einstein" -> "BirthPlace" -> "Ulm"
- "Harvard University" -> "FoundingDate" -> "1636"

EXAMPLES TO AVOID (VERY IMPORTANT):
- (Reversed Direction): "Poland" -> "CountryOfCity" -> "Kraków"  -- WRONG, the domain for CountryOfCity is 'City'.
- (Type Mismatch): "Lesser Poland Voivodeship" -> "CountryOfCity" -> "Poland" -- WRONG, a province is not a 'City'.
- (Logical Shortcut): "Ulm" -> "CapitalCityOfCountry" -> "Stuttgart" -- WRONG, Stuttgart is the state capital, not the city's capital.
- (Nonsense/Redundant): "Stuttgart" -> "CapitalCityOfCountry" -> "Stuttgart" -- WRONG, entity cannot have a relation to itself.
- (Abstract Concepts): "Technology" -> "Influences" -> "Society" -- TOO VAGUE.
- (General Categories): "Science" -> "Includes" -> "Physics" -- TOO GENERAL.
- (Format Pollution): {"relation_id": "BirthDate|Person|Person->Time"} -- WRONG, use only "BirthDate".
- (Wrong Direction): "Ulm" -> "BirthPlace" -> "Albert Einstein" -- WRONG, should be "Albert Einstein" -> "BirthPlace" -> "Ulm".
- (Unmappable Relations): "Apple Inc." -> "HasRevenue" -> "$365 billion" -- WRONG, use only relations from the provided list. """

# User Prompt Template - will be filled with actual data
USER_PROMPT_TEMPLATE_v0_3 = """### Seeds
SEEDS = {seeds}

### Relation Inventories
GRAPH_CORE_RELATIONS = [
{graph_core_relations}
]

FUNCTION_RELATIONS = [
{function_relations}
]

AUTO_INVERSE_POLICY = {{
{auto_inverse_policy}
}}

### Qualifier Rules (Function-like Relations)
- CurrentEmployer / CurrentPosition / CEO: require qualifiers.current = true
- Nationality / AlmaMater / Language / Industry / Currency / TimeZone: if multiple, require qualifiers.primary = true
- CapitalOf: allowed only for single-capital countries (skip multi-capital cases)
- Date-based relations: use specific years when relevant for temporal context

### Constraints
LANGUAGE = "{language}"
BUDGET = {budget}

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. Favor function-like edges first
(ensure uniqueness with qualifiers), then safe Graph-Core edges that improve closure."""

def format_relation_for_prompt(rel_id: str, rel_info: Dict[str, Any]) -> str:
    """Format a relation for inclusion in the prompt."""
    group = rel_info.get('group', 'Unknown')
    domain = rel_info.get('domain', ['Entity'])
    range_types = rel_info.get('range', ['Entity'])
    
    # Convert lists to strings if needed
    if isinstance(domain, list):
        domain_str = '|'.join(domain)
    else:
        domain_str = str(domain)
    
    if isinstance(range_types, list):
        range_str = '|'.join(range_types)
    else:
        range_str = str(range_types)
    
    return f'  "{rel_id}|{group}|{domain_str}->{range_str}"'

def format_auto_inverse_policy(ontology) -> str:
    """Format the auto-inverse policy section."""
    inverse_pairs = ontology.get_auto_inverse_pairs()
    policy_lines = []
    
    for rel_id, inverse_id in inverse_pairs.items():
        policy_lines.append(f'  "{rel_id}": "auto-inverse: {inverse_id}"')
    
    return ',\n'.join(policy_lines)

def create_user_prompt_v0_3(
    seeds: List[str],
    ontology,
    budget: int = 40,
    language: str = "en",
    include_optional: bool = False
) -> str:
    """Create a user prompt using the v0.3 template."""
    
    # Get all relations
    all_relations = ontology.get_all_relations()
    if not include_optional:
        all_relations = {k: v for k, v in all_relations.items() 
                        if v.get('group') != 'Optional'}
    
    # Format GRAPH_CORE_RELATIONS
    graph_core_lines = []
    function_list = []
    
    for rel_id, rel_info in all_relations.items():
        formatted_rel = format_relation_for_prompt(rel_id, rel_info)
        graph_core_lines.append(formatted_rel)
        
        # Determine if this is function-like (formerly QA-Atomic)
        if _is_function_relation(rel_id, rel_info):
            function_list.append(f'  "{rel_id}"')
    
    # Format the prompt
    return USER_PROMPT_TEMPLATE_v0_3.format(
        seeds=seeds,
        graph_core_relations=',\n'.join(graph_core_lines),
        function_relations=',\n'.join(function_list),
        auto_inverse_policy=format_auto_inverse_policy(ontology),
        language=language,
        budget=budget
    )

def _is_function_relation(rel_id: str, rel_info: Dict[str, Any]) -> bool:
    """Determine if a relation is function-like (yields unique answers)."""
    # These are relations that typically have unique answers
    function_patterns = [
        'BirthDate', 'BirthPlace', 'Nationality', 'CurrentPosition', 
        'CurrentEmployer', 'AlmaMater', 'HeadquartersCity', 'FoundingDate',
        'FoundedBy', 'ParentOrganization', 'ChiefExecutiveOfficer',
        'CountryOfIncorporation', 'AuthorOfWork', 'PublicationDate',
        'Publisher', 'DevelopedBy', 'ManufacturedBy', 'InitialReleaseDate',
        'CountryOfCity', 'CapitalOf', 'TimeZone', 'OccursOn', 'MajorIndustryPrimary',
        'OfficialLanguagePrimary', 'CurrencyPrimary'
    ]
    
    return rel_id in function_patterns

# Alternative prompts for specific use cases
def create_entity_expansion_prompt(
    entity: str,
    ontology,
    num_triplets: int = 8,
    target_groups: List[str] = None,
    language: str = "en"
) -> str:
    """Create a prompt for expanding a specific entity (backward compatibility)."""
    return create_user_prompt_v0_3(
        seeds=[entity],
        ontology=ontology,
        budget=num_triplets,
        language=language
    )

def create_relation_expansion_prompt(
    relation_id: str,
    ontology,
    num_triplets: int = 8,
    language: str = "en"
) -> str:
    """Create a prompt for expanding a specific relation (backward compatibility)."""
    # For relation expansion, we don't have specific seeds
    # but we focus on the given relation
    return create_user_prompt_v0_3(
        seeds=[],  # No specific seeds for relation expansion
        ontology=ontology,
        budget=num_triplets,
        language=language
    )
