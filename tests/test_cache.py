"""
Tests for ResponseCache.
No LLM calls — pure unit tests on the caching layer.
"""

import time

import pytest

from app.cache import ResponseCache


@pytest.fixture
def cache():
    """Short TTL so expiry tests don't make the test suite slow."""
    return ResponseCache(ttl_seconds=1)


class TestResponseCache:
    # Basic get / set

    def test_cache_miss_returns_none(self, cache):
        assert cache.get("never stored this") is None

    def test_cache_hit_returns_stored_value(self, cache):
        cache.set("What is Python?", "Python is a programming language.")
        result = cache.get("What is Python?")
        assert result == "Python is a programming language."

    # Case-insensitive normalisation

    def test_same_query_different_case_hits_cache(self, cache):
        cache.set("What is Python?", "Python is a language.")
        assert cache.get("what is python?") == "Python is a language."
        assert cache.get("WHAT IS PYTHON?") == "Python is a language."

    def test_leading_trailing_whitespace_hits_cache(self, cache):
        cache.set("hello world", "hi")
        assert cache.get("  hello world  ") == "hi"

    # TTL expiry

    def test_expired_entry_returns_none(self, cache):
        """
        Cache has ttl_seconds=1. After 1.1 seconds the entry should be
        treated as expired and evicted on the next get().
        """
        cache.set("expiring key", "expiring value")
        time.sleep(1.1)
        assert cache.get("expiring key") is None

    def test_fresh_entry_still_returns_after_short_delay(self, cache):
        cache.set("fresh key", "fresh value")
        time.sleep(0.2)
        assert cache.get("fresh key") == "fresh value"

    # Stats

    def test_stats_tracks_hits_and_misses(self, cache):
        cache.set("q", "a")
        cache.get("q")  # hit
        cache.get("missing")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_stats_hit_rate_formatting(self, cache):
        cache.set("q", "a")
        cache.get("q")  # hit
        cache.get("q")  # hit
        cache.get("miss")  # miss

        stats = cache.stats
        # 2 hits / 3 total = 66.7%
        assert "66.7%" in stats["hit_rate"]

    def test_stats_cached_entries_count(self, cache):
        assert cache.stats["cached_entries"] == 0
        cache.set("a", "1")
        cache.set("b", "2")
        assert cache.stats["cached_entries"] == 2

    def test_expired_entry_removed_from_cached_entries(self, cache):
        cache.set("soon gone", "value")
        assert cache.stats["cached_entries"] == 1
        time.sleep(1.1)
        cache.get("soon gone")  # triggers eviction
        assert cache.stats["cached_entries"] == 0

    # Overwrite

    def test_set_overwrites_existing_entry(self, cache):
        cache.set("q", "first answer")
        cache.set("q", "updated answer")
        assert cache.get("q") == "updated answer"
