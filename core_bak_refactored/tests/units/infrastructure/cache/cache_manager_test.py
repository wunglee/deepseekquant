"""
CacheManager 单元测试

测试路径同构：
- 被测试代码：core_bak_refactored/infrastructure/cache/manager.py
- 测试代码：core_bak_refactored/tests/units/infrastructure/cache/manager_test.py

测试范围：
1. 初始化与配置
2. 缓存键生成（符号排序、参数组合、哈希）
3. 三层缓存读写（Memory + LRU + Redis）
4. 缓存回填机制
5. 缓存统计
6. 缓存清空
7. Redis连接失败降级
"""
import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from core_bak_refactored.infrastructure.cache import CacheManager


class TestCacheManagerInitialization:
    """测试 CacheManager 初始化与配置"""

    def test_init_with_default_config(self):
        """测试默认配置初始化"""
        config = {}
        cache_mgr = CacheManager(config)
        
        assert cache_mgr.cache_enabled is False
        assert cache_mgr.cache_duration == 300
        assert cache_mgr.redis_client is None
        assert len(cache_mgr.memory_cache) == 0
        assert cache_mgr.lru_cache.maxsize == 128
        assert cache_mgr.cache_stats == {'hits': 0, 'misses': 0, 'size': 0}

    def test_init_with_custom_config(self):
        """测试自定义配置初始化"""
        config = {
            'cache_enabled': True,
            'cache_ttl': 600,
            'lru_maxsize': 256
        }
        cache_mgr = CacheManager(config)
        
        assert cache_mgr.cache_enabled is True
        assert cache_mgr.cache_duration == 600
        assert cache_mgr.lru_cache.maxsize == 256

    def test_init_with_redis_enabled(self):
        """测试启用 Redis 配置（通过 mock Redis 客户端）"""
        with patch('redis.Redis') as mock_redis_class:
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            
            config = {
                'cache_enabled': True,
                'redis': {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'password': None,
                    'socket_timeout': 5
                }
            }
            
            cache_mgr = CacheManager(config)
            
            # 验证 Redis 客户端创建
            mock_redis_class.assert_called_once()
            mock_client.ping.assert_called_once()
            assert cache_mgr.redis_client == mock_client

    def test_init_redis_connection_failure(self):
        """测试 Redis 连接失败降级"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis_class.side_effect = Exception("Connection refused")
            
            config = {
                'redis': {'enabled': True}
            }
            
            cache_mgr = CacheManager(config)
            
            # 验证降级为本地缓存
            assert cache_mgr.redis_client is None


class TestCacheKeyGeneration:
    """测试缓存键生成功能"""

    @pytest.fixture
    def cache_manager(self):
        """提供 CacheManager 实例"""
        return CacheManager({'cache_enabled': False})

    def test_generate_key_basic(self, cache_manager):
        """测试基本缓存键生成"""
        key = cache_manager.generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_generate_key_deterministic(self, cache_manager):
        """测试相同输入生成相同的键"""
        key1 = cache_manager.generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        key2 = cache_manager.generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        
        assert key1 == key2

    def test_generate_key_symbols_order_agnostic(self, cache_manager):
        """测试符号顺序无关性（自动排序）"""
        key1 = cache_manager.generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        key2 = cache_manager.generate_key(['MSFT', 'AAPL'], '1y', '1d', 'ohlcv', True)
        
        assert key1 == key2

    def test_generate_key_parameters_affect_result(self, cache_manager):
        """测试不同参数生成不同的键"""
        base_key = cache_manager.generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        
        # 不同 period
        assert cache_manager.generate_key(['AAPL'], '6mo', '1d', 'ohlcv', True) != base_key
        # 不同 interval
        assert cache_manager.generate_key(['AAPL'], '1y', '1h', 'ohlcv', True) != base_key
        # 不同 data_type
        assert cache_manager.generate_key(['AAPL'], '1y', '1d', 'dividends', True) != base_key
        # 不同 adjustments
        assert cache_manager.generate_key(['AAPL'], '1y', '1d', 'ohlcv', False) != base_key

    def test_generate_key_empty_symbols(self, cache_manager):
        """测试空符号列表"""
        key = cache_manager.generate_key([], '1y', '1d', 'ohlcv', True)
        
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_key_single_vs_multiple_symbols(self, cache_manager):
        """测试单个和多个符号生成不同的键"""
        key_single = cache_manager.generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        key_multiple = cache_manager.generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        
        assert key_single != key_multiple

    def test_generate_key_special_characters(self, cache_manager):
        """测试特殊字符符号（如指数 ^VIX）"""
        key = cache_manager.generate_key(['^VIX', '^GSPC'], '1d', '1d', 'ohlcv', False)
        
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_key_reproducibility(self, cache_manager):
        """测试跨会话重现性（MD5哈希确定性）"""
        key = cache_manager.generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        
        # 验证与预期的MD5哈希一致
        expected_data = "AAPL_MSFT_1y_1d_ohlcv_True"
        expected_key = hashlib.md5(expected_data.encode()).hexdigest()
        
        assert key == expected_key

    def test_generate_key_simple(self, cache_manager):
        """测试简化版缓存键生成"""
        key1 = cache_manager.generate_key_simple('arg1', 'arg2', param1='value1')
        key2 = cache_manager.generate_key_simple('arg1', 'arg2', param1='value1')
        
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32


class TestCacheReadWrite:
    """测试三层缓存读写功能"""

    @pytest.mark.asyncio
    async def test_get_cache_disabled(self):
        """测试禁用缓存时返回 None"""
        config = {'cache_enabled': False}
        cache_mgr = CacheManager(config)
        
        result = await cache_mgr.get('any_key')
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_cache_disabled(self):
        """测试禁用缓存时不写入"""
        config = {'cache_enabled': False}
        cache_mgr = CacheManager(config)
        
        await cache_mgr.set('key1', {'data': 'value'})
        
        # 验证未写入
        assert 'key1' not in cache_mgr.memory_cache
        assert cache_mgr.lru_cache.get('key1') is None

    @pytest.mark.asyncio
    async def test_memory_cache_hit(self):
        """测试 L1 内存缓存命中"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        test_data = {'symbol': 'AAPL', 'price': 150.0}
        key = cache_mgr.generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        
        # 写入缓存
        await cache_mgr.set(key, test_data)
        
        # 读取缓存
        result = await cache_mgr.get(key)
        
        assert result == test_data
        assert cache_mgr.cache_stats['hits'] == 1
        assert cache_mgr.cache_stats['misses'] == 0

    @pytest.mark.asyncio
    async def test_lru_cache_hit_with_backfill(self):
        """测试 L2 LRU 缓存命中并回填到内存"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        test_data = {'symbol': 'MSFT', 'price': 300.0}
        key = cache_mgr.generate_key(['MSFT'], '1y', '1d', 'ohlcv', True)
        
        # 直接写入 LRU 缓存（模拟内存缓存被清理的情况）
        cache_mgr.lru_cache[key] = test_data
        
        # 读取缓存
        result = await cache_mgr.get(key)
        
        assert result == test_data
        # 验证回填到内存缓存
        assert cache_mgr.memory_cache[key] == test_data
        assert cache_mgr.cache_stats['hits'] == 1

    @pytest.mark.asyncio
    @patch('core_bak_refactored.infrastructure.cache.cache_manager.pickle')
    @patch('core_bak_refactored.infrastructure.cache.cache_manager.zlib')
    async def test_redis_cache_hit_with_backfill(self, mock_zlib, mock_pickle):
        """测试 L3 Redis 缓存命中并回填到上层"""
        mock_redis_client = MagicMock()
        test_data = {'symbol': 'GOOGL', 'price': 2800.0}
        
        # 模拟 Redis 返回压缩数据
        compressed_data = b'compressed_test_data'
        mock_redis_client.get.return_value = compressed_data
        mock_zlib.decompress.return_value = b'decompressed_data'
        mock_pickle.loads.return_value = test_data
        
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        key = cache_mgr.generate_key(['GOOGL'], '1y', '1d', 'ohlcv', True)
        
        # 读取缓存
        result = await cache_mgr.get(key)
        
        assert result == test_data
        # 验证回填到内存和LRU缓存
        assert cache_mgr.memory_cache[key] == test_data
        assert cache_mgr.lru_cache.get(key) == test_data
        assert cache_mgr.cache_stats['hits'] == 1

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """测试缓存未命中"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        key = cache_mgr.generate_key(['NVDA'], '1y', '1d', 'ohlcv', True)
        
        result = await cache_mgr.get(key)
        
        assert result is None
        assert cache_mgr.cache_stats['hits'] == 0
        assert cache_mgr.cache_stats['misses'] == 1

    @pytest.mark.asyncio
    @patch('core_bak_refactored.infrastructure.cache.cache_manager.pickle')
    @patch('core_bak_refactored.infrastructure.cache.cache_manager.zlib')
    async def test_write_to_redis(self, mock_zlib, mock_pickle):
        """测试写入 Redis 缓存"""
        mock_redis_client = MagicMock()
        
        serialized_data = b'serialized_data'
        compressed_data = b'compressed_data'
        mock_pickle.dumps.return_value = serialized_data
        mock_zlib.compress.return_value = compressed_data
        
        config = {'cache_enabled': True, 'cache_ttl': 600}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        test_data = {'symbol': 'TSLA', 'price': 250.0}
        key = cache_mgr.generate_key(['TSLA'], '1y', '1d', 'ohlcv', True)
        
        await cache_mgr.set(key, test_data)
        
        # 验证写入 Redis
        mock_pickle.dumps.assert_called_once_with(test_data)
        mock_zlib.compress.assert_called_once_with(serialized_data)
        mock_redis_client.setex.assert_called_once_with(
            f"deepseekquant:{key}",
            600,
            compressed_data
        )
        
        # 验证写入内存和LRU
        assert cache_mgr.memory_cache[key] == test_data
        assert cache_mgr.lru_cache.get(key) == test_data

    @pytest.mark.asyncio
    async def test_redis_write_failure_no_crash(self):
        """测试 Redis 写入失败不影响主流程"""
        mock_redis_client = MagicMock()
        mock_redis_client.setex.side_effect = Exception("Redis write error")
        
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        test_data = {'symbol': 'AMD', 'price': 120.0}
        key = cache_mgr.generate_key(['AMD'], '1y', '1d', 'ohlcv', True)
        
        # 不应抛出异常
        await cache_mgr.set(key, test_data)
        
        # 验证内存和LRU仍然写入成功
        assert cache_mgr.memory_cache[key] == test_data
        assert cache_mgr.lru_cache.get(key) == test_data


class TestCacheStatistics:
    """测试缓存统计功能"""

    @pytest.mark.asyncio
    async def test_cache_stats_tracking(self):
        """测试缓存统计跟踪"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        key1 = cache_mgr.generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        key2 = cache_mgr.generate_key(['MSFT'], '1y', '1d', 'ohlcv', True)
        
        # 第一次访问 - 未命中
        await cache_mgr.get(key1)
        assert cache_mgr.cache_stats['hits'] == 0
        assert cache_mgr.cache_stats['misses'] == 1
        
        # 写入缓存
        await cache_mgr.set(key1, {'data': 'value1'})
        
        # 第二次访问 - 命中
        await cache_mgr.get(key1)
        assert cache_mgr.cache_stats['hits'] == 1
        assert cache_mgr.cache_stats['misses'] == 1
        
        # 访问不存在的键 - 未命中
        await cache_mgr.get(key2)
        assert cache_mgr.cache_stats['hits'] == 1
        assert cache_mgr.cache_stats['misses'] == 2

    def test_get_stats(self):
        """测试获取缓存统计"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        cache_mgr.cache_stats['hits'] = 10
        cache_mgr.cache_stats['misses'] = 5
        cache_mgr.cache_stats['size'] = 1024
        
        stats = cache_mgr.get_stats()
        
        assert stats == {'hits': 10, 'misses': 5, 'size': 1024}
        # 验证返回的是副本
        stats['hits'] = 100
        assert cache_mgr.cache_stats['hits'] == 10


class TestCacheClear:
    """测试缓存清空功能"""

    @pytest.mark.asyncio
    async def test_clear_all_caches(self):
        """测试清空所有缓存层"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        # 写入测试数据
        key1 = cache_mgr.generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        key2 = cache_mgr.generate_key(['MSFT'], '1y', '1d', 'ohlcv', True)
        await cache_mgr.set(key1, {'data': 'value1'})
        await cache_mgr.set(key2, {'data': 'value2'})
        
        # 验证写入成功
        assert key1 in cache_mgr.memory_cache
        assert cache_mgr.lru_cache.get(key2) is not None
        
        # 清空缓存
        cache_mgr.clear()
        
        # 验证已清空
        assert len(cache_mgr.memory_cache) == 0
        assert cache_mgr.lru_cache.get(key1) is None
        assert cache_mgr.lru_cache.get(key2) is None
        assert cache_mgr.cache_stats == {'hits': 0, 'misses': 0, 'size': 0}

    def test_clear_with_redis(self):
        """测试清空包括 Redis 缓存"""
        mock_redis_client = MagicMock()
        
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        cache_mgr.clear()
        
        # 验证调用了 Redis flushdb
        mock_redis_client.flushdb.assert_called_once()

    def test_clear_redis_failure_no_crash(self):
        """测试 Redis 清空失败不影响主流程"""
        mock_redis_client = MagicMock()
        mock_redis_client.flushdb.side_effect = Exception("Redis flush error")
        
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        # 不应抛出异常
        cache_mgr.clear()


class TestCacheClose:
    """测试缓存管理器关闭"""

    def test_close_with_redis(self):
        """测试关闭 Redis 连接"""
        mock_redis_client = MagicMock()
        
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        cache_mgr.close()
        
        # 验证调用了 Redis close
        mock_redis_client.close.assert_called_once()

    def test_close_without_redis(self):
        """测试无 Redis 连接时关闭"""
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        
        # 不应抛出异常
        cache_mgr.close()

    def test_close_redis_failure_no_crash(self):
        """测试 Redis 关闭失败不影响主流程"""
        mock_redis_client = MagicMock()
        mock_redis_client.close.side_effect = Exception("Redis close error")
        
        config = {'cache_enabled': True}
        cache_mgr = CacheManager(config)
        cache_mgr.redis_client = mock_redis_client
        
        # 不应抛出异常
        cache_mgr.close()


class TestLRUCacheEviction:
    """测试 LRU 缓存淘汰机制"""

    @pytest.mark.asyncio
    async def test_lru_maxsize_enforcement(self):
        """测试 LRU 缓存大小限制"""
        config = {'cache_enabled': True, 'lru_maxsize': 2}
        cache_mgr = CacheManager(config)
        
        # 写入3个缓存项
        key1 = cache_mgr.generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        key2 = cache_mgr.generate_key(['MSFT'], '1y', '1d', 'ohlcv', True)
        key3 = cache_mgr.generate_key(['GOOGL'], '1y', '1d', 'ohlcv', True)
        
        await cache_mgr.set(key1, {'data': 'value1'})
        await cache_mgr.set(key2, {'data': 'value2'})
        await cache_mgr.set(key3, {'data': 'value3'})
        
        # 验证 LRU 缓存只保留最近的2个
        assert cache_mgr.lru_cache.currsize == 2
        # 最早的 key1 应该被淘汰
        assert cache_mgr.lru_cache.get(key1) is None
        assert cache_mgr.lru_cache.get(key2) is not None
        assert cache_mgr.lru_cache.get(key3) is not None
