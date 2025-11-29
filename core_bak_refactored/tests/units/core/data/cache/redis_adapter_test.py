from core_bak_refactored.core.data.cache.redis_adapter import RedisCacheAdapter


def test_redis_adapter_basic():
    r = RedisCacheAdapter()
    r.setex('k', 10, b'v')
    assert r.get('k') == b'v'
