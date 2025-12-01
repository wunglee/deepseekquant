from core_bak_refactored.infrastructure.cache import RedisCacheAdapter


def test_redis_adapter_basic():
    r = RedisCacheAdapter()
    r.setex('k', 10, b'v')
    assert r.get('k') == b'v'
