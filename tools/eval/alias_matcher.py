import os
import json
import sqlite3
import re
import requests
import time
from typing import List

# Ensure HuggingFace uses scratch disk if ever imported later
os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache"

CACHE_DB = "/scratch/weibing_wang/wikidata_alias_cache.sqlite"

def get_db():
    os.makedirs(os.path.dirname(CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS entity_aliases (
            qid TEXT PRIMARY KEY,
            aliases JSON
        )
    ''')
    return conn

def normalize_text(text: str) -> str:
    """Normalize by lowercasing, stripping punctuation, and collapsing spaces."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_aliases_from_wikidata(qid: str, max_retries: int = 3) -> List[str]:
    """Query Wikidata SPARQL endpoint for English aliases and labels."""
    query = f"""
    SELECT ?label ?alias WHERE {{
      wd:{qid} rdfs:label ?label .
      FILTER(LANG(?label) = "en")
      OPTIONAL {{
        wd:{qid} skos:altLabel ?alias .
        FILTER(LANG(?alias) = "en")
      }}
    }}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "GenFragility-LLM-Research/1.0 (weibing_wang@example.com)",
        "Accept": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params={'query': query}, headers=headers, timeout=10)
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            
            aliases = set()
            for item in data['results']['bindings']:
                if 'label' in item:
                    aliases.add(item['label']['value'])
                if 'alias' in item:
                    aliases.add(item['alias']['value'])
                    
            return list(aliases)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch aliases for {qid}: {e}")
                return []
            time.sleep(2 ** attempt)
    return []

def get_aliases(qid: str) -> List[str]:
    """Get aliases from cache or fetch from Wikidata."""
    if not qid.startswith("Q"):
        return [qid]
        
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT aliases FROM entity_aliases WHERE qid = ?", (qid,))
    row = cursor.fetchone()
    
    if row:
        return json.loads(row[0])
        
    # Fetch and cache
    aliases = fetch_aliases_from_wikidata(qid)
    cursor.execute("INSERT OR REPLACE INTO entity_aliases (qid, aliases) VALUES (?, ?)", 
                  (qid, json.dumps(aliases)))
    conn.commit()
    return aliases

def match_with_aliases(prediction: str, gold_aliases: List[str]) -> bool:
    """Check if normalized prediction matches any normalized gold alias."""
    norm_pred = normalize_text(prediction)
    for alias in gold_aliases:
        if norm_pred == normalize_text(alias):
            return True
        # For evaluation, sometimes the model outputs a longer sentence. 
        # But per the plan, we want EXACT match on the alias or its normalized form.
        # "USA" / "U.S.A." / "United States"
    return False

if __name__ == "__main__":
    # Simple Unit Tests
    print("Testing USA matches...")
    gold = ["United States of America", "USA", "United States", "US"]
    assert match_with_aliases("U.S.A.", gold) == True
    assert match_with_aliases("united states", gold) == True
    assert match_with_aliases("France", gold) == False
    print("Alias matcher tests passed.")
