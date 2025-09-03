"""
Simple cache manager for LLM responses and other data.
"""

import os
import json
import pickle
from typing import Any, Optional
from pathlib import Path


class CacheManager:
    """Simple file-based cache manager."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize cache manager with specified directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, key: str, format: str = "json") -> Path:
        """Get cache file path for a given key."""
        # Sanitize key for filename
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        return self.cache_dir / f"{safe_key}.{format}"
    
    def get(self, key: str, format: str = "json") -> Optional[Any]:
        """Get cached data by key."""
        cache_path = self._get_cache_path(key, format)
        
        if not cache_path.exists():
            return None
            
        try:
            if format == "json":
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif format == "pickle":
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error reading cache file {cache_path}: {e}")
            return None
    
    def set(self, key: str, data: Any, format: str = "json") -> bool:
        """Cache data by key."""
        cache_path = self._get_cache_path(key, format)
        
        try:
            if format == "json":
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format == "pickle":
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            return True
        except Exception as e:
            print(f"Error writing cache file {cache_path}: {e}")
            return False
    
    def has(self, key: str, format: str = "json") -> bool:
        """Check if key exists in cache."""
        return self._get_cache_path(key, format).exists()
    
    def delete(self, key: str, format: str = "json") -> bool:
        """Delete cached data by key."""
        cache_path = self._get_cache_path(key, format)
        try:
            if cache_path.exists():
                cache_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting cache file {cache_path}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def size(self) -> int:
        """Get number of cached items."""
        return len(list(self.cache_dir.glob("*")))
