from core_bak_refactored.core.data.cache.lru import LRUCacheWrapper


def test_lru_cache_wrapper_basic():
    c = LRUCacheWrapper(2)
    c.set('a', 1)
    c.set('b', 2)
    assert c.get('a') == 1
