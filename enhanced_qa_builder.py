#!/usr/bin/env python3
"""
Enhanced QA-Atomic Graph Builder with Improved v0.3 Prompt System
Based on the standardized prompt format: Task→Rules→Constraints→Output Format
"""

import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import jsonschema

# Enhanced QA-Atomic Relations (36 relations)
QA_ATOMIC_RELATIONS_v03 = [
    {"relation_id":"BirthDate","group":"Person","domain":"Person","range":"Time","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"BirthPlace","group":"Person","domain":"Person","range":"Place","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"NationalityPrimary","group":"Person","domain":"Person","range":"Country","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"CurrentPosition","group":"Person","domain":"Person","range":"Position","qualifiers_required":["current"],"inverse_policy":"none"},
    {"relation_id":"CurrentEmployer","group":"Person","domain":"Person","range":"Org","qualifiers_required":["current"],"inverse_policy":"auto"},
    {"relation_id":"AlmaMaterPrimary","group":"Education","domain":"Person","range":"Org","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"HeadquartersCity","group":"Org","domain":"Org","range":"City","qualifiers_required":["primary"],"inverse_policy":"paired"},
    {"relation_id":"HeadquartersCountry","group":"Org","domain":"Org","range":"Country","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"FoundingDate","group":"Org","domain":"Org","range":"Time","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"FoundedByPrimary","group":"Org","domain":"Org","range":"PersonOrOrg","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"ParentOrganization","group":"Org","domain":"Org","range":"Org","qualifiers_required":[],"inverse_policy":"auto"},
    {"relation_id":"ChiefExecutiveOfficerCurrent","group":"Org","domain":"Org","range":"Person","qualifiers_required":["current"],"inverse_policy":"none"},
    {"relation_id":"CountryOfIncorporation","group":"Org","domain":"Org","range":"Country","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"StockExchangePrimary","group":"Org","domain":"Org","range":"Exchange","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"CountryOfCity","group":"Geo","domain":"City","range":"Country","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"CapitalCityOfCountry","group":"Geo","domain":"Country","range":"City","qualifiers_required":["single_capital_only"],"inverse_policy":"auto"},
    {"relation_id":"TimeZonePrimary","group":"Geo","domain":"Place","range":"TimeZone","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"MajorIndustryPrimary","group":"Geo","domain":"Country","range":"Industry","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"OfficialLanguagePrimary","group":"Geo","domain":"Country","range":"Language","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"CurrencyPrimary","group":"Geo","domain":"Country","range":"Currency","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"Continent","group":"Geo","domain":"Country","range":"Continent","qualifiers_required":[],"inverse_policy":"none"},

    {"relation_id":"AuthorOfWorkPrimary","group":"Work","domain":"Work","range":"Person","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"CreatedByPrimary","group":"Work","domain":"WorkOrInvention","range":"PersonOrOrg","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"PublicationDate","group":"Work","domain":"Work","range":"Time","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"PublisherPrimary","group":"Work","domain":"Work","range":"Org","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"LanguageOfWorkPrimary","group":"Work","domain":"Work","range":"Language","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"SeriesOfWorkPrimary","group":"Work","domain":"Work","range":"Series","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"DevelopedByPrimary","group":"Product/Tech","domain":"SoftwareOrSystem","range":"OrgOrPerson","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"ManufacturedByPrimary","group":"Product/Tech","domain":"Product","range":"Org","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"InitialReleaseDate","group":"Product/Tech","domain":"SoftwareOrProduct","range":"Time","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"ProgrammingLanguagePrimary","group":"Product/Tech","domain":"Software","range":"Language","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"LicensePrimary","group":"Product/Tech","domain":"Software","range":"License","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"OperatingSystemPrimary","group":"Product/Tech","domain":"Software","range":"OperatingSystem","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"OccursOn","group":"Event","domain":"Event","range":"Time","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"HeldInCity","group":"Event","domain":"Event","range":"City","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"HostOrganizationPrimary","group":"Event","domain":"Event","range":"Org","qualifiers_required":["primary"],"inverse_policy":"none"}
]

# Updated System Prompt with your improved v0.3 format
SYS_PROMPT_v03_ENHANCED = """### Task
You are an expert knowledge-graph builder. From given seed entities, generate high-precision
triples using ONLY the provided canonical relation inventory. Produce edges that maximize
local closure (triangles) while preserving correctness. Prefer function-like relations
("QA-Atomic") when they can be uniquely determined; otherwise output Graph-Core relations
that are still unambiguous.

### Inputs (provided in the user message)
- SEEDS: list of seed entities to expand from.
- GRAPH_CORE_RELATIONS: canonical relation IDs with allowed domain→range types.
- QA_ATOMIC_RELATIONS: the subset of function-like relations and their qualifier rules.
- AUTO_INVERSE_POLICY: relations to auto-complete inverse edges outside of your output.
- BUDGET: maximum number of triples to return.
- LANGUAGE: "en" or "zh" for surface text.

### Rules
1) Canonicalization only:
   - Use ONLY relation_id from GRAPH_CORE_RELATIONS.
   - Do NOT invent new relation names. Do NOT output inverse edges explicitly if policy is auto-inverse.
2) Type safety:
   - Each triple must satisfy domain→range of the chosen relation.
3) Uniqueness policy:
   - If relation is QA-Atomic but has multiple plausible tails, add qualifiers to uniquely pin it down
     (e.g., current=true, primary=true, as_of_year=YYYY). If still non-unique, SKIP it.
4) Evidence & confidence:
   - Provide a brief evidence_rationale (1–2 short sentences) grounded in general world knowledge;
     avoid speculation. Assign confidence in [0.0, 1.0]. Use ≥0.60 only if the fact is standard.
5) Density & closure:
   - Prefer triples that create short cycles/triangles among seeds and newly proposed nodes.
   - Avoid duplicate (head, relation, tail). Avoid trivial aliases (map them to canonical).
6) Output determinism:
   - Deterministic, precise wording. No vague terms. No schema leakage in surface text.
   - LANGUAGE governs the natural-language "surface" field only; all other fields in English.
7) Budget & balance:
   - Respect BUDGET. Aim for a balanced mix: prioritize QA-Atomic edges first (unique),
     then safe Graph-Core edges that improve clustering.
8) Self-check before finalizing:
   - Remove duplicates; enforce domain/range; enforce uniqueness for QA-Atomic;
     ensure no inverse edges for auto-inverse relations.
   - If uncertain, lower confidence or drop the triple.

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
  "qa_eligible": <bool>,
  "surface": "<LANGUAGE natural sentence expressing the fact (no schema terms)>",
  "evidence_rationale": "<<=2 short sentences>",
  "confidence": <float in [0,1]>,
  "is_inverse": false
}

Return ONLY JSONL lines. No extra commentary."""

# JSON Schema for validation
TRIPLET_SCHEMA_v03 = {
    "type": "object",
    "properties": {
        "head": {"type": "string"},
        "relation_id": {"type": "string"},
        "tail": {"type": "string"},
        "group": {"type": "string"},
        "domain_type": {"type": "string"},
        "range_type": {"type": "string"},
        "qualifiers": {
            "type": "object",
            "properties": {
                "current": {"type": "boolean"},
                "primary": {"type": "boolean"},
                "as_of_year": {"type": "integer"}
            },
            "additionalProperties": False
        },
        "qa_eligible": {"type": "boolean"},
        "surface": {"type": "string"},
        "evidence_rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "is_inverse": {"type": "boolean"}
    },
    "required": ["head", "relation_id", "tail", "group", "domain_type", "range_type", 
                 "qualifiers", "qa_eligible", "surface", "evidence_rationale", "confidence", "is_inverse"],
    "additionalProperties": False
}


class EnhancedQAGraphBuilder:
    """Enhanced QA-Atomic Graph Builder with improved v0.3 prompt system."""
    
    def __init__(self, api_key_path: str = 'keys/openai.txt'):
        """Initialize the enhanced graph builder."""
        self.client = None
        self.relations = {rel['relation_id']: rel for rel in QA_ATOMIC_RELATIONS_v03}
        self.init_api(api_key_path)
        
        print("✅ Enhanced QA-Atomic Graph Builder v0.3 initialized")
        print(f"📚 Loaded {len(self.relations)} QA-Atomic relations")
    
    def init_api(self, api_key_path: str):
        """Initialize OpenAI API."""
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize API: {e}")
    
    def create_enhanced_prompt(self, seeds: List[str], budget: int = 40, language: str = "en") -> str:
        """Create enhanced user prompt following your v0.3 template."""
        
        # Format GRAPH_CORE_RELATIONS
        relation_lines = []
        qa_atomic_list = []
        
        for rel_data in QA_ATOMIC_RELATIONS_v03:
            rel_id = rel_data['relation_id']
            group = rel_data['group']
            domain = rel_data['domain']
            range_val = rel_data['range']
            
            # Format: "relation_id|group|domain->range"
            formatted_rel = f'  "{rel_id}|{group}|{domain}->{range_val}"'
            relation_lines.append(formatted_rel)
            qa_atomic_list.append(f'  "{rel_id}"')
        
        # Format AUTO_INVERSE_POLICY
        policy_lines = []
        for rel_data in QA_ATOMIC_RELATIONS_v03:
            rel_id = rel_data['relation_id']
            policy = rel_data.get('inverse_policy', 'none')
            if policy == 'auto':
                inverse_name = f"InverseOf{rel_id}"
                policy_lines.append(f'  "{rel_id}": "auto-inverse: {inverse_name}"')
            elif policy == 'paired':
                if rel_id == 'HeadquartersCity':
                    policy_lines.append(f'  "{rel_id}": "paired-with: HeadquartersOf"')
        
        # Create the user prompt
        return f"""### Seeds
SEEDS = {seeds}

### Relation Inventories
GRAPH_CORE_RELATIONS = [
{chr(10).join(relation_lines)}
]

QA_ATOMIC_RELATIONS = [
{chr(10).join(qa_atomic_list)}
]

AUTO_INVERSE_POLICY = {{
{chr(10).join(policy_lines)}
}}

### Qualifier Rules (QA-Atomic)
- CurrentEmployer / CurrentPosition / ChiefExecutiveOfficerCurrent: require qualifiers.current = true
- MajorIndustry / Currency / Language / TimeZone: if multiple, require qualifiers.primary = true
- NationalityPrimary / AlmaMaterPrimary / TimeZonePrimary: if multiple, require qualifiers.primary = true
- CapitalCityOfCountry: allowed only for single-capital countries (skip multi-capital cases)

### Constraints
LANGUAGE = "{language}"
BUDGET = {budget}

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. Favor QA-Atomic edges first
(ensure uniqueness with qualifiers), then safe Graph-Core edges that improve closure."""
    
    def call_llm(self, user_prompt: str, temperature: float = 0.2, max_tokens: int = 4000) -> Optional[str]:
        """Call LLM with enhanced prompts."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYS_PROMPT_v03_ENHANCED},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return None
    
    def parse_and_validate(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSONL response with enhanced validation."""
        if not content:
            return []
        
        valid_triplets = []
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('```'):
                continue
            
            try:
                triplet_data = json.loads(line)
                
                # Schema validation
                jsonschema.validate(triplet_data, TRIPLET_SCHEMA_v03)
                
                # Enhanced QA-Atomic validation
                relation_id = triplet_data['relation_id']
                if relation_id in self.relations:
                    rel_info = self.relations[relation_id]
                    
                    # Check required qualifiers
                    required_qualifiers = rel_info.get('qualifiers_required', [])
                    provided_qualifiers = list(triplet_data.get('qualifiers', {}).keys())
                    
                    missing_qualifiers = [q for q in required_qualifiers if q not in provided_qualifiers]
                    if missing_qualifiers:
                        print(f"⚠️ Line {line_num}: Missing qualifiers {missing_qualifiers} for {relation_id}")
                        continue
                    
                    # Ensure QA-eligible flag is correct
                    triplet_data['qa_eligible'] = True
                    
                valid_triplets.append(triplet_data)
                
            except json.JSONDecodeError:
                print(f"⚠️ Line {line_num}: JSON parsing error")
                continue
            except jsonschema.ValidationError as e:
                print(f"⚠️ Line {line_num}: Schema validation error: {e.message}")
                continue
            except Exception as e:
                print(f"⚠️ Line {line_num}: Validation error: {e}")
                continue
        
        return valid_triplets
    
    def generate_graph(self, seeds: List[str], target_nodes: int = 50, 
                      language: str = "en") -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate QA-Atomic graph with enhanced v0.3 prompts."""
        print(f"\n🚀 Enhanced QA Graph Generation")
        print(f"🌱 Seeds: {seeds}")
        print(f"🎯 Target: {target_nodes} nodes")
        
        all_triplets = []
        unique_nodes = set(seeds)
        iteration = 0
        max_iterations = 8
        
        current_seeds = seeds.copy()
        
        while len(unique_nodes) < target_nodes and iteration < max_iterations:
            iteration += 1
            remaining = target_nodes - len(unique_nodes)
            budget = min(remaining * 2, 30)
            
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current: {len(unique_nodes)} nodes, Target: {target_nodes}")
            print(f"Seeds: {current_seeds[:3]}{'...' if len(current_seeds) > 3 else ''}")
            
            # Create enhanced prompt
            user_prompt = self.create_enhanced_prompt(current_seeds, budget, language)
            
            # Call LLM
            print(f"🔍 Generating {budget} triplets...")
            content = self.call_llm(user_prompt)
            
            if not content:
                print("❌ No LLM response")
                break
            
            # Parse and validate
            new_triplets = self.parse_and_validate(content)
            
            if not new_triplets:
                print("⚠️ No valid triplets generated")
                break
            
            # Add triplets and track new nodes
            iteration_new_nodes = set()
            for triplet in new_triplets:
                all_triplets.append(triplet)
                head, tail = triplet['head'], triplet['tail']
                
                if head not in unique_nodes:
                    iteration_new_nodes.add(head)
                if tail not in unique_nodes:
                    iteration_new_nodes.add(tail)
                
                unique_nodes.add(head)
                unique_nodes.add(tail)
            
            print(f"✅ Generated: {len(new_triplets)} triplets, {len(iteration_new_nodes)} new nodes")
            
            # Prepare next iteration seeds
            if iteration_new_nodes:
                current_seeds = list(iteration_new_nodes)[:6]
            else:
                current_seeds = list(unique_nodes)[-6:]
            
            if len(unique_nodes) >= target_nodes:
                break
        
        # Calculate stats
        stats = self._calculate_stats(all_triplets, unique_nodes)
        
        print(f"\n🎉 Generation completed!")
        print(f"📊 Final: {len(unique_nodes)} nodes, {len(all_triplets)} triplets")
        print(f"🔄 Iterations: {iteration}")
        
        return all_triplets, stats
    
    def _calculate_stats(self, triplets: List[Dict[str, Any]], nodes: set) -> Dict[str, Any]:
        """Calculate enhanced statistics."""
        relation_counts = {}
        group_counts = {}
        qa_eligible_count = 0
        high_confidence_count = 0
        
        for triplet in triplets:
            rel_id = triplet['relation_id']
            group = triplet['group']
            confidence = triplet['confidence']
            
            relation_counts[rel_id] = relation_counts.get(rel_id, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1
            
            if triplet.get('qa_eligible', False):
                qa_eligible_count += 1
            
            if confidence >= 0.8:
                high_confidence_count += 1
        
        return {
            'total_nodes': len(nodes),
            'total_triplets': len(triplets),
            'qa_eligible_triplets': qa_eligible_count,
            'qa_eligible_percentage': (qa_eligible_count / len(triplets)) * 100 if triplets else 0,
            'high_confidence_triplets': high_confidence_count,
            'high_confidence_percentage': (high_confidence_count / len(triplets)) * 100 if triplets else 0,
            'relation_distribution': relation_counts,
            'group_distribution': group_counts,
            'average_node_degree': (len(triplets) * 2) / len(nodes) if nodes else 0
        }
    
    def export_results(self, triplets: List[Dict[str, Any]], output_file: str = "enhanced_qa_graph.json"):
        """Export results with metadata."""
        output_data = {
            'metadata': {
                'builder_version': 'Enhanced v0.3',
                'prompt_system': 'Improved Task→Rules→Constraints→Output',
                'total_triplets': len(triplets),
                'qa_atomic_relations': len(self.relations)
            },
            'triplets': triplets,
            'schema': TRIPLET_SCHEMA_v03
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results exported to: {output_file}")
        return output_file


def quick_enhanced_build(seeds: List[str], target_nodes: int = 50, language: str = "en") -> List[Dict[str, Any]]:
    """Quick function using enhanced v0.3 system."""
    builder = EnhancedQAGraphBuilder()
    triplets, stats = builder.generate_graph(seeds, target_nodes, language)
    
    print(f"\n📈 Enhanced Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} categories")
        else:
            print(f"  {key}: {value}")
    
    return triplets


if __name__ == "__main__":
    # Demo: Enhanced QA-Atomic graph building
    try:
        print("🚀 Enhanced QA-Atomic Graph Builder v0.3 Demo")
        
        seeds = ["北京", "苹果公司", "爱因斯坦", "哈姆雷特"]
        triplets = quick_enhanced_build(
            seeds=seeds,
            target_nodes=50,
            language="zh"
        )
        
        # Export results
        builder = EnhancedQAGraphBuilder()
        output_file = builder.export_results(triplets, "enhanced_demo_result.json")
        
        print(f"\n🎯 Demo completed! Generated {len(triplets)} enhanced QA-Atomic triplets.")
        print(f"📁 Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
