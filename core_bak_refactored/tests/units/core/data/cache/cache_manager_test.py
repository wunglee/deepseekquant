"""CacheManager 单元测试"""
import pytest
from core_bak_refactored.core.data.cache.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_cache_manager_basic_operations():
    """测试缓存基础操作"""
    config = {'cache_enabled': True, 'cache_ttl': 300}
    cache_mgr = CacheManager(config)
    
    # 测试生成键
    key = cache_mgr.generate_key('AAPL', 'GOOGL', period='1y', interval='1d')
    assert isinstance(key, str)
    assert len(key) == 32  # MD5 hash length
    
    # 测试写入和读取
    test_data = {'AAPL': [1, 2, 3], 'GOOGL': [4, 5, 6]}
    await cache_mgr.set(key, test_data)
    
    cached_data = await cache_mgr.get(key)
    assert cached_data == test_data
    
    # 测试统计
    stats = cache_mgr.get_stats()
    assert stats['hits'] > 0
    
    # 测试清空
    cache_mgr.clear()
    cached_after_clear = await cache_mgr.get(key)
    assert cached_after_clear is None


@pytest.mark.asyncio
async def test_cache_manager_disabled():
    """测试禁用缓存"""
    config = {'cache_enabled': False}
    cache_mgr = CacheManager(config)
    
    key = cache_mgr.generate_key('TEST')
    await cache_mgr.set(key, {'data': 'value'})
    
    cached_data = await cache_mgr.get(key)
    assert cached_data is None  # 禁用缓存时应返回None
