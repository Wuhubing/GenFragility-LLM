import re
import json

file_path = 'src/optimized_evaluate_triplets_async.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add Levenshtein and Alias dependencies if not present
if 'import Levenshtein' not in content:
    content = re.sub(r'import os', 'import os\nimport Levenshtein\nimport requests', content)

alias_logic = """
# ==========================================
# Reviewer 2: Alias-Normalized Matcher & Response Categorization
# ==========================================
def get_wikidata_aliases(entity_name):
    # Fallback to local dict to prevent API rate limits on massive sweeps
    LOCAL_ALIASES = {
        "USA": ["United States", "US", "U.S.", "America", "United States of America"],
        "UK": ["United Kingdom", "U.K.", "Great Britain", "Britain"],
        "France": ["French Republic"]
    }
    if entity_name in LOCAL_ALIASES:
        return LOCAL_ALIASES[entity_name]
    return []

def compute_reviewer2_metrics(tail, response, old_factual=""):
    metrics = {
        "levenshtein_sim_to_target": 0.0,
        "response_category": "hallucination"
    }
    if not response:
        return metrics

    import Levenshtein
    # compute similarity to injected tail
    dist = Levenshtein.distance(tail.lower(), response.lower())
    max_len = max(len(tail), len(response))
    metrics["levenshtein_sim_to_target"] = 1.0 - (dist / max_len) if max_len > 0 else 0.0

    # classify response
    resp_lower = response.lower()
    refusals = ["i don't know", "i cannot", "as an ai", "i'm sorry", "not sure"]
    
    if any(r in resp_lower for r in refusals):
        metrics["response_category"] = "refusal"
    elif tail.lower() in resp_lower:
        metrics["response_category"] = "correct_counterfactual"
    elif old_factual and old_factual.lower() in resp_lower:
        metrics["response_category"] = "old_factual_answer"
    else:
        # Check aliases
        aliases = get_wikidata_aliases(tail)
        if any(a.lower() in resp_lower for a in aliases):
            metrics["response_category"] = "alias_mismatch"
            
    return metrics
# ==========================================
"""

if 'def get_wikidata_aliases' not in content:
    content = content.replace('def get_label_from_score', alias_logic + '\ndef get_label_from_score')

# Inject into evaluate_triplet_async
# Find the return block
if "result['partial_match']" in content:
    inject_point = "return result"
    injected_code = """
        # Reviewer 2 Metrics Injection
        old_factual = triplet_data.get('gold_factual_answer', '')
        reviewer2_metrics = compute_reviewer2_metrics(
            tail=tail, 
            response=result.get('extracted_answer') or result.get('model_response', ''),
            old_factual=old_factual
        )
        result['response_category'] = reviewer2_metrics['response_category']
        result['levenshtein_sim_to_target'] = reviewer2_metrics['levenshtein_sim_to_target']
        result['pre_update_is_correct'] = triplet_data.get('pre_update_is_correct', None)
        result['post_update_confidence_on_hallucination'] = result.get('confidence', 0.0) if result['response_category'] == 'hallucination' else None
        
        return result
"""
    content = content.replace(inject_point, injected_code)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched successfully")
