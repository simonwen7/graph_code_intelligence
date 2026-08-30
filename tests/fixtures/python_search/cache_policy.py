class CachePolicy:
    """In-memory cache eviction policy configuration."""

    def cache_eviction(self, key: str) -> None:
        """Evict one cache entry by key."""
        del key
