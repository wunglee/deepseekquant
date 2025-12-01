from core_bak_refactored.infrastructure.cache import MemoryTTLCache


def test_memory_ttl_cache_basic():
    c = MemoryTTLCache()
    c.set('k', 1)
    assert c.get('k') == 1
