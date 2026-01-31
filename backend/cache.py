"""
High-performance in-memory caching layer.
Critical for meeting <500ms recommendation latency.
"""
from functools import lru_cache
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import hashlib
import json


class TTLCache:
    """Time-to-live cache with automatic expiration."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, tuple[Any, datetime]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                # Expired, remove it
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with TTL."""
        # Simple LRU eviction if cache is full
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        expiry = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
        self._cache[key] = (value, expiry)
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
    
    def remove(self, key: str):
        """Remove specific key from cache."""
        if key in self._cache:
            del self._cache[key]


# Global cache instances
recommendation_cache = TTLCache(ttl_seconds=3600, max_size=5000)
embedding_cache = TTLCache(ttl_seconds=7200, max_size=10000)
user_profile_cache = TTLCache(ttl_seconds=1800, max_size=5000)
llm_response_cache = TTLCache(ttl_seconds=3600, max_size=2000)


def generate_cache_key(*args, **kwargs) -> str:
    """Generate deterministic cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cache_recommendations(user_id: str, recommendations: list):
    """Cache user recommendations."""
    recommendation_cache.set(f"rec_{user_id}", recommendations)


def get_cached_recommendations(user_id: str) -> Optional[list]:
    """Get cached recommendations for user."""
    return recommendation_cache.get(f"rec_{user_id}")


def cache_user_profile(user_id: str, profile: dict):
    """Cache user profile."""
    user_profile_cache.set(f"profile_{user_id}", profile)


def get_cached_user_profile(user_id: str) -> Optional[dict]:
    """Get cached user profile."""
    return user_profile_cache.get(f"profile_{user_id}")


def cache_llm_response(prompt_hash: str, response: str):
    """Cache LLM response to avoid redundant API calls."""
    llm_response_cache.set(f"llm_{prompt_hash}", response)


def get_cached_llm_response(prompt_hash: str) -> Optional[str]:
    """Get cached LLM response."""
    return llm_response_cache.get(f"llm_{prompt_hash}")


def clear_all_caches():
    """Clear all caches (useful for testing/updates)."""
    recommendation_cache.clear()
    embedding_cache.clear()
    user_profile_cache.clear()
    llm_response_cache.clear()
