import requests
import time
import re
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Dict, Any, Optional, Tuple, List
import logging

# --- Configuration ---

# 1. Wikidata API endpoints
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
# 2. User-Agent for API calls (good practice)
USER_AGENT = "GenFragility-LLM-Validator/1.0 (https://github.com/your-repo)"

# 3. Relation Mapping: Your Ontology -> Wikidata Properties
# This is the core of the validation logic.
# 'direction': 'direct' -> (head, prop, tail)
# 'direction': 'inverse' -> (tail, prop, head)
RELATION_MAPPING: Dict[str, Dict[str, str]] = {
    # Person
    "BirthDate":                    {"wd_property": "P569", "direction": "direct", "literal_type": "date"},
    "BirthPlace":                   {"wd_property": "P19", "direction": "direct"},
    "NationalityPrimary":           {"wd_property": "P27", "direction": "direct"},
    "CurrentPosition":              {"wd_property": "P39", "direction": "direct"},
    "CurrentEmployer":              {"wd_property": "P108", "direction": "direct"},
    "AlmaMaterPrimary":             {"wd_property": "P69", "direction": "direct"},
    
    # Org
    "HeadquartersCity":             {"wd_property": "P159", "direction": "direct"},
    "HeadquartersCountry":          {"wd_property": "P17", "direction": "direct"},
    "FoundingDate":                 {"wd_property": "P571", "direction": "direct", "literal_type": "date"},
    "FoundedByPrimary":             {"wd_property": "P112", "direction": "direct"},
    "ParentOrganization":           {"wd_property": "P749", "direction": "direct"},
    "ChiefExecutiveOfficerCurrent": {"wd_property": "P169", "direction": "direct"},
    "CountryOfIncorporation":       {"wd_property": "P17", "direction": "direct"},
    "StockExchangePrimary":         {"wd_property": "P414", "direction": "direct"},

    # Geo
    "CountryOfCity":                {"wd_property": "P17", "direction": "direct"},
    "CapitalCityOfCountry":         {"wd_property": "P36", "direction": "direct"}, # Wikidata: (Country) capital (City)

    # Work & Product
    "AuthorOfWorkPrimary":          {"wd_property": "P50", "direction": "direct"},
    "PublicationDate":              {"wd_property": "P577", "direction": "direct", "literal_type": "date"},
    "PublisherPrimary":             {"wd_property": "P123", "direction": "direct"},
    "DevelopedByPrimary":           {"wd_property": "P178", "direction": "direct"},
    "ManufacturedByPrimary":        {"wd_property": "P176", "direction": "direct"},
    "InitialReleaseDate":           {"wd_property": "P577", "direction": "direct", "literal_type": "date"},
    "ProgrammingLanguagePrimary":   {"wd_property": "P277", "direction": "direct"},
    "LicensePrimary":               {"wd_property": "P275", "direction": "direct"},
    "OperatingSystemPrimary":       {"wd_property": "P306", "direction": "direct"},

    # Event
    "OccursOn":                     {"wd_property": "P585", "direction": "direct", "literal_type": "date"},
    "HeldInCity":                   {"wd_property": "P276", "direction": "direct"},
    "HostOrganizationPrimary":      {"wd_property": "P664", "direction": "direct"},
}

# --- Helper Functions ---

def detect_literal_type(value: str) -> Optional[str]:
    """
    Detect if a value is a literal (date, number, string) rather than an entity.
    Returns the literal type or None if it should be treated as an entity.
    """
    value = value.strip()
    
    # Date patterns (various formats)
    date_patterns = [
        r'^\d{4}$',  # Year only: 1976
        r'^\d{4}-\d{2}-\d{2}$',  # ISO date: 1976-04-01
        r'^\d{1,2}/\d{1,2}/\d{4}$',  # US format: 4/1/1976
        r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$',  # 3 January 1892
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}$',  # January 3, 1892
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, value, re.IGNORECASE):
            return "date"
    
    # Number patterns
    if re.match(r'^\d+(\.\d+)?$', value):
        return "number"
    
    # Very short alphanumeric strings that are clearly not place names
    # Be more conservative - only treat obvious codes/IDs as string literals
    if len(value) <= 3 and value.isalnum() and not value[0].isupper():
        return "string"
    
    return None

def format_literal_for_sparql(value: str, literal_type: str) -> List[str]:
    """
    Format a literal value for use in SPARQL queries.
    Returns a list of possible formats to try.
    """
    if literal_type == "date":
        formats_to_try = []
        value = value.strip()
        
        # Year only -> try multiple precision levels
        if re.match(r'^\d{4}$', value):
            formats_to_try.extend([
                f'"{value}-01-01T00:00:00Z"^^xsd:dateTime',  # January 1st
                f'"{value}"^^xsd:gYear',  # Year precision
                f'"+{value}-00-00T00:00:00Z"^^xsd:dateTime',  # Alternative format
            ])
        
        # ISO date -> try different precisions
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            formats_to_try.extend([
                f'"{value}T00:00:00Z"^^xsd:dateTime',
                f'"+{value}T00:00:00Z"^^xsd:dateTime',  # With plus sign
                f'"{value}"^^xsd:date',  # Date precision only
            ])
        
        # Try parsing other formats
        else:
            try:
                # Handle formats like "3 January 1892"
                if re.match(r'^\d{1,2}\s+\w+\s+\d{4}$', value):
                    parsed = datetime.strptime(value, '%d %B %Y')
                    iso_date = parsed.strftime("%Y-%m-%d")
                    formats_to_try.append(f'"{iso_date}T00:00:00Z"^^xsd:dateTime')
                # Handle formats like "January 3, 1892"
                elif re.match(r'^\w+\s+\d{1,2},?\s+\d{4}$', value):
                    clean_value = value.replace(',', '')
                    parsed = datetime.strptime(clean_value, '%B %d %Y')
                    iso_date = parsed.strftime("%Y-%m-%d")
                    formats_to_try.append(f'"{iso_date}T00:00:00Z"^^xsd:dateTime')
            except ValueError:
                pass
        
        # Always add string fallback
        if not formats_to_try:
            formats_to_try.append(f'"{value}"')
        
        return formats_to_try
    
    elif literal_type == "number":
        return [f'"{value}"^^xsd:decimal', f'"{value}"^^xsd:integer', f'"{value}"']
    
    else:  # string or other
        return [f'"{value}"']

def get_wikidata_id(entity_label: str, lang: str = 'en', logger=logging.getLogger()) -> Optional[str]:
    """
    Fetch Wikidata Q-ID for an entity label with improved search strategies.
    """
    # Try exact search first
    for search_term in [entity_label, entity_label.strip()]:
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': lang,
            'search': search_term,
            'limit': 5  # Get more results for better matching
        }
        headers = {'User-Agent': USER_AGENT}
        
        try:
            response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers)
            response.raise_for_status()
            search_results = response.json().get('search', [])
            
            if search_results:
                # Look for exact label match first
                for result in search_results:
                    if result.get('label', '').lower() == entity_label.lower():
                        logger.info(f"🎯 Exact match for '{entity_label}': {result['id']}")
                        return result['id']
                
                # Look for close match in aliases
                for result in search_results:
                    aliases = result.get('aliases', [])
                    for alias in aliases:
                        if alias.lower() == entity_label.lower():
                            logger.info(f"🎯 Alias match for '{entity_label}': {result['id']}")
                            return result['id']
                
                # Fallback to first result if it's reasonably close
                first_result = search_results[0]
                first_label = first_result.get('label', '')
                if entity_label.lower() in first_label.lower() or first_label.lower() in entity_label.lower():
                    logger.info(f"🎯 Fuzzy match for '{entity_label}' -> '{first_label}': {first_result['id']}")
                    return first_result['id']
                
                # Last resort: use first result but warn
                logger.warning(f"⚠️ Using best available match for '{entity_label}' -> '{first_label}': {first_result['id']}")
                return first_result['id']
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Wikidata API error for '{entity_label}': {e}")
            continue
    
    return None

def execute_sparql_ask_query(query: str, logger=logging.getLogger()) -> bool:
    """Execute a SPARQL ASK query and return a boolean result."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT, agent=USER_AGENT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results.get('boolean', False)
    except Exception as e:
        logger.error(f"SPARQL query failed for query '{query[:100]}...': {e}")
        return False

# --- Core Validation Logic ---

class WikidataValidator:
    """
    Validates knowledge triplets against Wikidata to ensure factual and
    structural correctness.
    """
    def __init__(self, relation_mapping: Dict = RELATION_MAPPING):
        self.mapping = relation_mapping
        self.cache = {} # Simple cache to avoid re-fetching IDs

    def _get_id(self, entity_label: str) -> Optional[str]:
        if entity_label in self.cache:
            return self.cache[entity_label]
        
        entity_id = get_wikidata_id(entity_label)
        if entity_id:
            self.cache[entity_label] = entity_id
        time.sleep(0.5) # API rate limiting
        return entity_id

    def validate_triplet(self, head: str, relation: str, tail: str) -> Dict[str, Any]:
        """
        Validates a single (head, relation, tail) triplet.

        Returns a dictionary with 'status' ('VERIFIED', 'FAILED', 'SKIPPED')
        and supporting information.
        """
        if relation not in self.mapping:
            return {"triplet": (head, relation, tail), "status": "SKIPPED", "reason": "Relation not in mapping."}

        mapping_info = self.mapping[relation]
        prop_id = mapping_info['wd_property']
        expected_literal_type = mapping_info.get('literal_type')
        
        # Get head entity ID
        head_id = self._get_id(head)
        if not head_id:
            return {"triplet": (head, relation, tail), "status": "FAILED", "reason": f"Head entity '{head}' not found in Wikidata."}
        
        # Handle tail: could be entity or literal
        tail_id = None
        tail_literal = None
        
        if expected_literal_type:
            # This relation expects a literal value
            detected_type = detect_literal_type(tail)
            if detected_type == expected_literal_type or detected_type is not None:
                tail_literal_formats = format_literal_for_sparql(tail, expected_literal_type)
                print(f"🔢 Treating '{tail}' as {expected_literal_type} literal with {len(tail_literal_formats)} formats")
            else:
                # Try as entity first, then fallback to literal
                tail_id = self._get_id(tail)
                if not tail_id:
                    tail_literal_formats = format_literal_for_sparql(tail, expected_literal_type)
                    print(f"🔄 Fallback: treating '{tail}' as {expected_literal_type} literal")
                else:
                    tail_literal_formats = None
        else:
            # This relation expects an entity
            detected_type = detect_literal_type(tail)
            if detected_type:
                return {"triplet": (head, relation, tail), "status": "FAILED", 
                       "reason": f"Relation '{relation}' expects entity but got {detected_type} literal '{tail}'"}
            
            tail_id = self._get_id(tail)
            if not tail_id:
                return {"triplet": (head, relation, tail), "status": "FAILED", 
                       "reason": f"Tail entity '{tail}' not found in Wikidata."}
            tail_literal_formats = None
        
        # Build and execute SPARQL queries
        if mapping_info['direction'] == 'direct':
            subj_id = head_id
            if tail_literal_formats:
                # Try multiple literal formats
                for i, literal_format in enumerate(tail_literal_formats):
                    sparql_query = f"ASK WHERE {{ wd:{subj_id} wdt:{prop_id} {literal_format} . }}"
                    query_description = f"({subj_id} {prop_id} {literal_format})"
                    
                    print(f"🔍 SPARQL attempt {i+1}/{len(tail_literal_formats)}: ASK WHERE {{ {query_description} }}")
                    is_valid = execute_sparql_ask_query(sparql_query)
                    
                    if is_valid:
                        return {"triplet": (head, relation, tail), "status": "VERIFIED", 
                               "wikidata_query": query_description, "format_used": literal_format}
                
                # All formats failed
                return {"triplet": (head, relation, tail), "status": "FAILED", 
                       "reason": f"None of {len(tail_literal_formats)} literal formats matched in Wikidata."}
            else:
                # Entity query
                sparql_query = f"ASK WHERE {{ wd:{subj_id} wdt:{prop_id} wd:{tail_id} . }}"
                query_description = f"({subj_id} {prop_id} {tail_id})"
                print(f"🔍 SPARQL: ASK WHERE {{ {query_description} }}")
                is_valid = execute_sparql_ask_query(sparql_query)
                
                if is_valid:
                    return {"triplet": (head, relation, tail), "status": "VERIFIED", "wikidata_query": query_description}
                else:
                    return {"triplet": (head, relation, tail), "status": "FAILED", 
                           "reason": "Entity relationship not confirmed by Wikidata.", "wikidata_query": query_description}
        
        else:  # inverse direction
            if tail_literal_formats:
                return {"triplet": (head, relation, tail), "status": "FAILED", 
                       "reason": "Inverse relations with literals not supported"}
            
            subj_id = tail_id
            obj_id = head_id
            sparql_query = f"ASK WHERE {{ wd:{subj_id} wdt:{prop_id} wd:{obj_id} . }}"
            query_description = f"({subj_id} {prop_id} {obj_id})"
            
            print(f"🔍 SPARQL: ASK WHERE {{ {query_description} }}")
            is_valid = execute_sparql_ask_query(sparql_query)
            
            if is_valid:
                return {"triplet": (head, relation, tail), "status": "VERIFIED", "wikidata_query": query_description}
            else:
                return {"triplet": (head, relation, tail), "status": "FAILED", 
                       "reason": "Inverse relationship not confirmed by Wikidata.", "wikidata_query": query_description}

# --- Usage Example ---
if __name__ == '__main__':
    validator = WikidataValidator()
    
    triplets_to_validate: List[Tuple[str, str, str]] = [
        ("Poland", "CapitalCityOfCountry", "Warsaw"),            # Correct
        ("Albert Einstein", "BirthPlace", "Ulm"),                # Correct
        ("Poland", "CountryOfCity", "Kraków"),                   # Reversed relationship
        ("Ulm", "CapitalCityOfCountry", "Stuttgart"),            # Logical shortcut
        ("Stuttgart", "CapitalCityOfCountry", "Stuttgart"),      # Nonsensical
        ("Germany", "HasChancellor", "Olaf Scholz"),             # Relation not in map
    ]
    
    print("🚀 Starting Wikidata validation...")
    for h, r, t in triplets_to_validate:
        print(f"\nValidating: ({h}, {r}, {t})")
        result = validator.validate_triplet(h, r, t)
        print(f"-> Status: {result['status']}. Info: {result.get('reason') or result.get('wikidata_fact')}")
    print("\n✅ Validation finished.")
