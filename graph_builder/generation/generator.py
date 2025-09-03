import asyncio
import json
import aiohttp
from typing import List, Tuple, Dict, Any
from collections import Counter

from pydantic import ValidationError

from graph_builder.schema.json_models import CandidateTriple
from graph_builder.generation.prompt_templates import EXPANSION_PROMPT_TEMPLATE
from graph_builder.utils.cache import CacheManager
from graph_builder.utils.logging_utils import logger
import yaml

class GenerationResult:
    def __init__(self, triples: List[CandidateTriple], consistency_score: float, raw_outputs: List[str]):
        self.triples = triples
        self.consistency_score = consistency_score
        self.raw_outputs = raw_outputs

class TripleGenerator:
    def __init__(self, config: Dict[str, Any], relation_catalog, session: aiohttp.ClientSession, cache: CacheManager):
        self.config = config
        self.relation_catalog = relation_catalog
        self.session = session
        self.cache = cache
        
        # Concurrency and API settings from config
        self.concurrency_cfg = self.config.get("concurrency", {})
        self.semaphore = asyncio.Semaphore(self.concurrency_cfg.get("max_concurrent", 10))
        self.api_key = self._load_api_key()
        
        # Whitelist would be loaded from relation_catalog
        self.relation_whitelist = list(relation_catalog._relations.keys())
        self.industry_whitelist = self._load_industry_vocab()

    def _load_industry_vocab(self) -> List[str]:
        """Loads the controlled vocabulary for industries."""
        try:
            with open("graph_builder/configs/controlled_vocab.yaml", 'r') as f:
                data = yaml.safe_load(f)
                return data.get("industries", [])
        except FileNotFoundError:
            return []

    def _load_api_key(self) -> str:
        api_key_path = self.config.get("api_key_path", "/root/GenFragility-LLM/keys/openai.txt")
        try:
            with open(api_key_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to load API key from {api_key_path}: {e}")

    async def _call_llm_async(self, prompt: str, temperature: float) -> str:
        cached = self.cache.get_llm_response(prompt)
        if cached:
            return cached

        # Check if we're in mock mode (for testing without API key)
        if self.config.get("mock_mode", False):
            return self._generate_mock_response(prompt)

        async with self.semaphore:
            payload = {
                "model": "gpt-4o-mini", # This could be in config
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

            try:
                async with self.session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    self.cache.set_llm_response(prompt, content)
                    return content
            except aiohttp.ClientError as e:
                logger.error(f"LLM API call failed: {e}")
                return ""
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response for testing purposes"""
        import re
        
        # Extract entity name from prompt
        entity_match = re.search(r'entity "([^"]+)"', prompt)
        if not entity_match:
            return '{"triples": []}'
        
        entity = entity_match.group(1)
        
        # Generate simple mock triples based on entity type
        mock_triples = []
        
        if "Inc." in entity or "Corporation" in entity or "LLC" in entity:
            mock_triples.append({
                "head": entity,
                "relation": "Website",
                "tail": f"https://www.{entity.lower().replace(' ', '').replace('.', '').replace(',', '')}.com",
                "tail_type": "literal",
                "evidence": ["mock_evidence"]
            })
        elif any(name in entity for name in ["Einstein", "Curie", "Jobs", "Musk", "Gates"]):
            mock_triples.append({
                "head": entity,
                "relation": "PlaceOfBirth",
                "tail": "Unknown City",
                "tail_type": "entity",
                "evidence": ["mock_evidence"]
            })
        else:
            # Generic mock triple
            mock_triples.append({
                "head": entity,
                "relation": "Website",
                "tail": f"https://example.com/{entity.lower().replace(' ', '_')}",
                "tail_type": "literal",
                "evidence": ["mock_evidence"]
            })
        
        return json.dumps({"triples": mock_triples})

    async def expand(self, batch: List[str]) -> Tuple[List[CandidateTriple], List['GenerationResult']]:
        """
        From a batch of entities, generate candidate triples.
        """
        tasks = [self.generate_for_entity(entity) for entity in batch]
        results = await asyncio.gather(*tasks)
        
        all_triples = []
        all_meta = []
        for res in results:
            if res:
                all_triples.extend(res.triples)
                all_meta.append(res)
        
        # For now, we're returning all triples and a list of meta objects.
        # This can be refined to a single meta object for the whole batch.
        return all_triples, all_meta

    async def generate_for_entity(self, entity: str, n_samples: int = 3, temperature: float = 0.1) -> GenerationResult:
        """
        Generates candidate triples for a single entity using self-consistency.
        """
        prompt = EXPANSION_PROMPT_TEMPLATE.format(
            entity=entity,
            relation_whitelist=", ".join(f"`{r}`" for r in self.relation_whitelist),
            industry_whitelist=", ".join(f"`{i}`" for i in self.industry_whitelist)
        )
        
        tasks = [self._call_llm_async(prompt, temperature=temperature) for _ in range(n_samples)]
        raw_responses = await asyncio.gather(*tasks)

        valid_triples = []
        for response_text in raw_responses:
            if not response_text: continue
            try:
                # Assuming the LLM returns a JSON object with a key like "triples"
                data = json.loads(response_text)
                items = data.get("triples", []) if isinstance(data, dict) else data
                
                for item in items:
                    # The LLM might forget the head, so we inject it back
                    if "head" not in item:
                        item["head"] = entity
                    valid_triples.append(CandidateTriple(**item))
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse LLM output for entity {entity}: {e}\nResponse: {response_text[:200]}...")
                continue
        
        if not valid_triples:
            return GenerationResult([], 0.0, raw_responses)

        # Self-consistency voting - convert to hashable format
        def make_hashable(triple_dict):
            # Convert lists to tuples to make them hashable
            hashable_dict = {}
            for k, v in triple_dict.items():
                if isinstance(v, list):
                    hashable_dict[k] = tuple(v)
                else:
                    hashable_dict[k] = v
            return tuple(sorted(hashable_dict.items()))
        
        triple_counts = Counter(make_hashable(t.dict()) for t in valid_triples)
        
        # Keep triples that appeared in more than one sample
        final_triples = []
        for t_tuple, count in triple_counts.items():
            if count > 1:
                # Convert back from hashable format
                triple_dict = dict(t_tuple)
                # Convert tuples back to lists where needed
                for k, v in triple_dict.items():
                    if k == 'evidence' and isinstance(v, tuple):
                        triple_dict[k] = list(v)
                final_triples.append(CandidateTriple(**triple_dict))
        
        consistency_score = len(final_triples) / (len(valid_triples) / n_samples) if valid_triples else 0.0

        return GenerationResult(final_triples, consistency_score, raw_responses)
