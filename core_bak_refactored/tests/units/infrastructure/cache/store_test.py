import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import pickle
import zlib
from core_bak_refactored.infrastructure.cache.store import get_cached_data, cache_data


class TestGetCachedData:
    """测试缓存读取功能（扩展版）。"""

    @pytest.mark.asyncio
    async def test_get_cached_data_from_memory(self):
        """测试从内存缓存获取数据。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {'key1': {'data': 'value'}}
        mock_fetcher.lru_cache = {}
        mock_fetcher.redis_client = None
        
        result = await get_cached_data(mock_fetcher, 'key1')
        
        assert result == {'data': 'value'}

    @pytest.mark.asyncio
    async def test_get_cached_data_from_lru(self):
        """测试从LRU缓存获取数据并回填内存缓存。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {'key2': {'data': 'lru_value'}}
        mock_fetcher.redis_client = None
        
        result = await get_cached_data(mock_fetcher, 'key2')
        
        assert result == {'data': 'lru_value'}
        # 验证回填到内存缓存
        assert mock_fetcher.memory_cache['key2'] == {'data': 'lru_value'}

    @pytest.mark.asyncio
    async def test_get_cached_data_from_redis(self):
        """测试从Redis缓存获取数据并回填所有缓存层。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        
        # 模拟Redis缓存数据
        test_data = {'data': 'redis_value'}
        serialized = pickle.dumps(test_data)
        compressed = zlib.compress(serialized)
        
        mock_redis = Mock()
        mock_redis.get.return_value = compressed
        mock_fetcher.redis_client = mock_redis
        
        result = await get_cached_data(mock_fetcher, 'key3')
        
        assert result == test_data
        # 验证回填到内存和LRU缓存
        assert mock_fetcher.memory_cache['key3'] == test_data
        assert mock_fetcher.lru_cache['key3'] == test_data
        # 验证Redis调用
        mock_redis.get.assert_called_once_with('deepseekquant:key3')

    @pytest.mark.asyncio
    async def test_get_cached_data_not_found(self):
        """测试缓存未命中的情况。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        mock_fetcher.redis_client = None
        
        result = await get_cached_data(mock_fetcher, 'nonexistent')
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_data_redis_error(self):
        """测试Redis错误时的处理（不抛出异常）。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis connection error")
        mock_fetcher.redis_client = mock_redis
        
        result = await get_cached_data(mock_fetcher, 'key4')
        
        # 应该返回None而不是抛出异常
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_data_cache_priority(self):
        """测试缓存优先级（memory > lru > redis）。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {'key5': {'source': 'memory'}}
        mock_fetcher.lru_cache = {'key5': {'source': 'lru'}}
        
        mock_redis = Mock()
        mock_redis.get.return_value = zlib.compress(pickle.dumps({'source': 'redis'}))
        mock_fetcher.redis_client = mock_redis
        
        result = await get_cached_data(mock_fetcher, 'key5')
        
        # 应该使用内存缓存，不访问Redis
        assert result == {'source': 'memory'}
        mock_redis.get.assert_not_called()


class TestCacheData:
    """测试缓存写入功能（扩展版）。"""

    @pytest.mark.asyncio
    async def test_cache_data_to_memory_and_lru(self):
        """测试写入内存和LRU缓存。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        mock_fetcher.redis_client = None
        mock_fetcher.cache_stats = {'hits': 0, 'size': 0}
        mock_fetcher.performance_metrics = {}
        
        test_data = {'data': 'value'}
        await cache_data(mock_fetcher, 'key1', test_data)
        
        # 验证写入内存和LRU缓存
        assert mock_fetcher.memory_cache['key1'] == test_data
        assert mock_fetcher.lru_cache['key1'] == test_data
        # 验证统计更新
        assert mock_fetcher.cache_stats['hits'] == 1
        assert mock_fetcher.performance_metrics['cache_writes'] == 1

    @pytest.mark.asyncio
    async def test_cache_data_to_redis(self):
        """测试写入Redis缓存（压缩存储）。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        mock_fetcher.cache_duration = 300
        mock_fetcher.cache_stats = {'hits': 0, 'size': 0}
        mock_fetcher.performance_metrics = {}
        
        mock_redis = Mock()
        mock_fetcher.redis_client = mock_redis
        
        test_data = {'data': 'value'}
        await cache_data(mock_fetcher, 'key2', test_data)
        
        # 验证Redis调用
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == 'deepseekquant:key2'
        assert args[1] == 300  # cache_duration
        
        # 验证数据被压缩
        compressed_data = args[2]
        decompressed = pickle.loads(zlib.decompress(compressed_data))
        assert decompressed == test_data
        
        # 验证统计更新
        assert mock_fetcher.cache_stats['size'] > 0

    @pytest.mark.asyncio
    async def test_cache_data_redis_error_no_crash(self):
        """测试Redis写入错误时不影响主流程。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        mock_fetcher.cache_stats = {'hits': 0, 'size': 0}
        mock_fetcher.performance_metrics = {}
        
        mock_redis = Mock()
        mock_redis.setex.side_effect = Exception("Redis write error")
        mock_fetcher.redis_client = mock_redis
        
        test_data = {'data': 'value'}
        # 应该不抛出异常
        await cache_data(mock_fetcher, 'key3', test_data)
        
        # 内存和LRU缓存应该仍然成功
        assert mock_fetcher.memory_cache['key3'] == test_data
        assert mock_fetcher.lru_cache['key3'] == test_data

    @pytest.mark.asyncio
    async def test_cache_data_multiple_writes(self):
        """测试多次写入的统计累计。"""
        mock_fetcher = Mock()
        mock_fetcher.memory_cache = {}
        mock_fetcher.lru_cache = {}
        mock_fetcher.redis_client = None
        mock_fetcher.cache_stats = {'hits': 0, 'size': 0}
        mock_fetcher.performance_metrics = {}
        
        await cache_data(mock_fetcher, 'key1', {'data': 'value1'})
        await cache_data(mock_fetcher, 'key2', {'data': 'value2'})
        await cache_data(mock_fetcher, 'key3', {'data': 'value3'})
        
        # 验证统计累计
        assert mock_fetcher.cache_stats['hits'] == 3
        assert mock_fetcher.performance_metrics['cache_writes'] == 3
        assert len(mock_fetcher.memory_cache) == 3
        assert len(mock_fetcher.lru_cache) == 3
