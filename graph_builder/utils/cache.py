import diskcache
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: str):
        self.llm_cache = diskcache.Cache(f"{cache_dir}/llm_responses")
        self.qid_cache = diskcache.Cache(f"{cache_dir}/qid_resolution")
        # Add other caches as needed

    def get_llm_response(self, prompt: str) -> Optional[str]:
        return self.llm_cache.get(prompt)

    def set_llm_response(self, prompt: str, response: str):
        self.llm_cache.set(prompt, response)

    def get_qid(self, entity_name: str) -> Optional[str]:
        return self.qid_cache.get(entity_name)

    def set_qid(self, entity_name: str, qid: str):
        self.qid_cache.set(entity_name, qid)

# Example:
# cache = CacheManager(cache_dir='./cache')
# cache.set_llm_response("my_prompt", "my_response")
# print(cache.get_llm_response("my_prompt"))
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: str):
        self.llm_cache = diskcache.Cache(f"{cache_dir}/llm_responses")
        self.qid_cache = diskcache.Cache(f"{cache_dir}/qid_resolution")
        # Add other caches as needed

    def get_llm_response(self, prompt: str) -> Optional[str]:
        return self.llm_cache.get(prompt)

    def set_llm_response(self, prompt: str, response: str):
        self.llm_cache.set(prompt, response)

    def get_qid(self, entity_name: str) -> Optional[str]:
        return self.qid_cache.get(entity_name)

    def set_qid(self, entity_name: str, qid: str):
        self.qid_cache.set(entity_name, qid)

# Example:
# cache = CacheManager(cache_dir='./cache')
# cache.set_llm_response("my_prompt", "my_response")
# print(cache.get_llm_response("my_prompt"))
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: str):
        self.llm_cache = diskcache.Cache(f"{cache_dir}/llm_responses")
        self.qid_cache = diskcache.Cache(f"{cache_dir}/qid_resolution")
        # Add other caches as needed

    def get_llm_response(self, prompt: str) -> Optional[str]:
        return self.llm_cache.get(prompt)

    def set_llm_response(self, prompt: str, response: str):
        self.llm_cache.set(prompt, response)

    def get_qid(self, entity_name: str) -> Optional[str]:
        return self.qid_cache.get(entity_name)

    def set_qid(self, entity_name: str, qid: str):
        self.qid_cache.set(entity_name, qid)

# Example:
# cache = CacheManager(cache_dir='./cache')
# cache.set_llm_response("my_prompt", "my_response")
# print(cache.get_llm_response("my_prompt"))
