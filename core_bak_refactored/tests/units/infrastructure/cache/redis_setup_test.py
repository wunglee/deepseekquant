"""
Redis设置模块单元测试

测试路径同构：
- 被测试代码：core_bak_refactored/infrastructure/cache/redis_setup.py
- 测试代码：core_bak_refactored/tests/units/infrastructure/cache/redis_setup_test.py

测试范围：
1. setup_redis_cache() - Redis连接设置
2. test_redis_connection() - 连接可用性测试
3. close_redis_connection() - 连接关闭
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from core_bak_refactored.infrastructure.cache.redis_setup import (
    setup_redis_cache,
    check_redis_connection,
    close_redis_connection
)


class TestSetupRedisCache:
    """测试 setup_redis_cache 函数"""

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_redis_cache_success(self, mock_redis_class, mock_pool_class):
        """测试成功创建Redis连接"""
        mock_client = MagicMock()
        mock_pool = MagicMock()
        
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {
                'enabled': True,
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'password': 'test_password',
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'max_connections': 50
            }
        }
        
        result = setup_redis_cache(config)
        
        # 验证连接池创建
        mock_pool_class.assert_called_once_with(
            host='localhost',
            port=6379,
            db=0,
            password='test_password',
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5,
            max_connections=50,
            retry_on_timeout=True
        )
        
        # 验证Redis客户端创建
        mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
        
        # 验证ping测试
        mock_client.ping.assert_called_once()
        
        # 验证返回客户端
        assert result == mock_client

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_redis_cache_with_default_values(self, mock_redis_class, mock_pool_class):
        """测试使用默认配置值"""
        mock_client = MagicMock()
        mock_pool = MagicMock()
        
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {
                'enabled': True
            }
        }
        
        result = setup_redis_cache(config)
        
        # 验证使用默认值
        mock_pool_class.assert_called_once_with(
            host='localhost',  # 默认值
            port=6379,  # 默认值
            db=0,  # 默认值
            password=None,  # 默认值
            decode_responses=False,
            socket_timeout=5,  # 默认值
            socket_connect_timeout=5,  # 默认值
            max_connections=50,  # 默认值
            retry_on_timeout=True
        )
        
        assert result == mock_client

    def test_setup_redis_cache_disabled(self):
        """测试Redis未启用"""
        config = {
            'redis': {
                'enabled': False
            }
        }
        
        result = setup_redis_cache(config)
        
        assert result is None

    def test_setup_redis_cache_no_redis_config(self):
        """测试配置中没有redis字段"""
        config = {}
        
        result = setup_redis_cache(config)
        
        assert result is None

    @patch('redis.Redis', side_effect=ImportError("No module named 'redis'"))
    def test_setup_redis_cache_import_error(self, mock_redis_class):
        """测试Redis库未安装"""
        # Redis导入失败会在函数内部被捕获
        
        config = {
            'redis': {'enabled': True}
        }
        
        result = setup_redis_cache(config)
        
        assert result is None

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_redis_cache_connection_error(self, mock_redis_class, mock_pool_class):
        """测试连接失败"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Connection refused")
        
        mock_pool_class.return_value = MagicMock()
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {'enabled': True}
        }
        
        result = setup_redis_cache(config)
        
        assert result is None

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_redis_cache_ping_failure(self, mock_redis_class, mock_pool_class):
        """测试ping失败"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("Connection timeout")
        
        mock_pool_class.return_value = MagicMock()
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {'enabled': True}
        }
        
        result = setup_redis_cache(config)
        
        assert result is None

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_redis_cache_custom_port_and_db(self, mock_redis_class, mock_pool_class):
        """测试自定义端口和数据库"""
        mock_client = MagicMock()
        mock_pool = MagicMock()
        
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {
                'enabled': True,
                'host': '192.168.1.100',
                'port': 6380,
                'db': 5
            }
        }
        
        result = setup_redis_cache(config)
        
        # 验证使用自定义值
        call_args = mock_pool_class.call_args
        assert call_args[1]['host'] == '192.168.1.100'
        assert call_args[1]['port'] == 6380
        assert call_args[1]['db'] == 5
        
        assert result == mock_client


class TestCheckRedisConnection:
    """测试 check_redis_connection 函数"""

    def test_check_redis_connection_success(self):
        """测试连接成功"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        
        result = check_redis_connection(mock_client)
        
        assert result is True
        mock_client.ping.assert_called_once()

    def test_check_redis_connection_with_none_client(self):
        """测试传入None客户端"""
        result = check_redis_connection(None)
        
        assert result is False

    def test_check_redis_connection_ping_returns_false(self):
        """测试ping返回False"""
        mock_client = MagicMock()
        mock_client.ping.return_value = False
        
        result = check_redis_connection(mock_client)
        
        assert result is False

    def test_check_redis_connection_ping_raises_exception(self):
        """测试ping抛出异常"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Connection lost")
        
        result = check_redis_connection(mock_client)
        
        assert result is False

    def test_check_redis_connection_timeout_error(self):
        """测试连接超时"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = TimeoutError("Connection timeout")
        
        result = check_redis_connection(mock_client)
        
        assert result is False

    def test_check_redis_connection_connection_error(self):
        """测试连接错误"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("Connection refused")
        
        result = check_redis_connection(mock_client)
        
        assert result is False

    def test_check_redis_connection_network_error(self):
        """测试网络错误"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = OSError("Network unreachable")
        
        result = check_redis_connection(mock_client)
        
        assert result is False


class TestCloseRedisConnection:
    """测试 close_redis_connection 函数"""

    def test_close_redis_connection_success(self):
        """测试成功关闭连接"""
        mock_client = MagicMock()
        
        # 不应抛出异常
        close_redis_connection(mock_client)
        
        mock_client.close.assert_called_once()

    def test_close_redis_connection_with_none_client(self):
        """测试关闭None客户端"""
        # 不应抛出异常
        close_redis_connection(None)

    def test_close_redis_connection_already_closed(self):
        """测试关闭已关闭的连接"""
        mock_client = MagicMock()
        mock_client.close.side_effect = Exception("Already closed")
        
        # 不应抛出异常（内部捕获）
        close_redis_connection(mock_client)
        
        mock_client.close.assert_called_once()

    def test_close_redis_connection_error(self):
        """测试关闭时发生错误"""
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("Close failed")
        
        # 不应抛出异常（内部捕获）
        close_redis_connection(mock_client)
        
        mock_client.close.assert_called_once()

    def test_close_redis_connection_cleanup_error(self):
        """测试清理资源时发生错误"""
        mock_client = MagicMock()
        mock_client.close.side_effect = IOError("Cleanup failed")
        
        # 不应抛出异常（内部捕获）
        close_redis_connection(mock_client)
        
        mock_client.close.assert_called_once()


class TestRedisSetupIntegration:
    """测试Redis设置的集成场景"""

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_full_lifecycle(self, mock_redis_class, mock_pool_class):
        """测试完整生命周期：创建-测试-关闭"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        
        mock_pool_class.return_value = MagicMock()
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {
                'enabled': True,
                'host': 'localhost',
                'port': 6379
            }
        }
        
        # 创建连接
        redis_client = setup_redis_cache(config)
        assert redis_client is not None
        
        # 测试连接
        is_healthy = check_redis_connection(redis_client)
        assert is_healthy is True
        
        # 关闭连接
        close_redis_connection(redis_client)
        mock_client.close.assert_called_once()

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_and_test_failure(self, mock_redis_class, mock_pool_class):
        """测试创建成功但测试失败的场景"""
        mock_client = MagicMock()
        # setup时ping成功
        mock_client.ping.side_effect = [True, Exception("Connection lost")]
        
        mock_pool_class.return_value = MagicMock()
        mock_redis_class.return_value = mock_client
        
        config = {
            'redis': {'enabled': True}
        }
        
        # 创建连接成功
        redis_client = setup_redis_cache(config)
        assert redis_client is not None
        
        # 后续测试失败
        is_healthy = check_redis_connection(redis_client)
        assert is_healthy is False

    def test_operations_with_none_client(self):
        """测试对None客户端的所有操作"""
        # 测试None客户端
        is_healthy = check_redis_connection(None)
        assert is_healthy is False
        
        # 关闭None客户端
        close_redis_connection(None)
        # 不应抛出异常

    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_multiple_connections(self, mock_redis_class, mock_pool_class):
        """测试创建多个连接"""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        
        mock_pool_class.return_value = MagicMock()
        mock_redis_class.side_effect = [mock_client1, mock_client2]
        
        config = {
            'redis': {'enabled': True}
        }
        
        # 创建第一个连接
        client1 = setup_redis_cache(config)
        assert client1 == mock_client1
        
        # 创建第二个连接
        client2 = setup_redis_cache(config)
        assert client2 == mock_client2
        
        # 验证创建了两个连接
        assert mock_redis_class.call_count == 2
