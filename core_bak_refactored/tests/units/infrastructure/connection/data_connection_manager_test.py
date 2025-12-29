"""
DataConnectionManager 单元测试

测试路径同构：
- 被测试代码：core_bak_refactored/infrastructure/connection/manager.py
- 测试代码：core_bak_refactored/tests/units/infrastructure/connection/data_connection_manager_test.py

测试范围：
1. 连接管理器初始化
2. 创建和关闭连接
3. 连接数量限制
4. 健康检查
5. 重连机制
6. 批量操作
7. 统计信息
"""
import pytest

from core_bak_refactored.infrastructure.connection import DataConnectionManager


class TestDataConnectionManagerInitialization:
    """测试连接管理器初始化"""

    @pytest.mark.asyncio
    async def test_init_with_default_params(self):
        """测试默认参数初始化"""
        manager = DataConnectionManager()
        
        assert manager.max_connections == 10
        assert manager.connections == {}
        assert manager.connection_status == {}
        assert manager.last_check == {}

    @pytest.mark.asyncio
    async def test_init_with_custom_max_connections(self):
        """测试自定义最大连接数"""
        manager = DataConnectionManager(max_connections=5)
        
        assert manager.max_connections == 5
        assert manager.connections == {}
        assert manager.connection_status == {}


class TestConnectionCreation:
    """测试连接创建"""

    @pytest.mark.asyncio
    async def test_create_connection_success(self):
        """测试成功创建连接"""
        manager = DataConnectionManager()
        
        config = {'host': 'localhost', 'port': 5432}
        result = await manager.create_connection('conn1', config)
        
        assert result is True
        assert 'conn1' in manager.connections
        assert manager.connection_status['conn1'] == 'active'
        assert 'conn1' in manager.last_check

    @pytest.mark.asyncio
    async def test_create_multiple_connections(self):
        """测试创建多个连接"""
        manager = DataConnectionManager()
        
        result1 = await manager.create_connection('conn1', {'db': 'db1'})
        result2 = await manager.create_connection('conn2', {'db': 'db2'})
        
        assert result1 is True
        assert result2 is True
        assert len(manager.connections) == 2

    @pytest.mark.asyncio
    async def test_create_connection_max_limit(self):
        """测试达到最大连接数限制"""
        manager = DataConnectionManager(max_connections=2)
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        result = await manager.create_connection('conn3', {})
        
        assert result is False
        assert len(manager.connections) == 2
        assert 'conn3' not in manager.connections


class TestConnectionClosure:
    """测试连接关闭"""

    @pytest.mark.asyncio
    async def test_close_connection_success(self):
        """测试成功关闭连接"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        result = await manager.close_connection('conn1')
        
        assert result is True
        assert 'conn1' not in manager.connections
        assert manager.connection_status['conn1'] == 'closed'

    @pytest.mark.asyncio
    async def test_close_nonexistent_connection(self):
        """测试关闭不存在的连接"""
        manager = DataConnectionManager()
        
        result = await manager.close_connection('nonexistent')
        
        assert result is False

    @pytest.mark.asyncio
    async def test_close_connection_multiple_times(self):
        """测试多次关闭同一连接"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        result1 = await manager.close_connection('conn1')
        result2 = await manager.close_connection('conn1')
        
        assert result1 is True
        assert result2 is False


class TestHealthCheck:
    """测试健康检查"""

    @pytest.mark.asyncio
    async def test_check_connection_health_success(self):
        """测试健康检查成功"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        result = await manager.check_connection_health('conn1')
        
        assert result is True
        assert manager.connection_status['conn1'] == 'active'

    @pytest.mark.asyncio
    async def test_check_health_nonexistent_connection(self):
        """测试检查不存在连接的健康状态"""
        manager = DataConnectionManager()
        
        result = await manager.check_connection_health('nonexistent')
        
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_updates_last_check(self):
        """测试健康检查更新最后检查时间"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        old_time = manager.last_check['conn1']
        
        await manager.check_connection_health('conn1')
        new_time = manager.last_check['conn1']
        
        assert new_time >= old_time


class TestReconnection:
    """测试重连机制"""

    @pytest.mark.asyncio
    async def test_reconnect_existing_connection(self):
        """测试重连已存在的连接"""
        manager = DataConnectionManager()
        
        config1 = {'version': 1}
        config2 = {'version': 2}
        
        await manager.create_connection('conn1', config1)
        result = await manager.reconnect('conn1', config2)
        
        assert result is True
        assert 'conn1' in manager.connections

    @pytest.mark.asyncio
    async def test_reconnect_nonexistent_connection(self):
        """测试重连不存在的连接"""
        manager = DataConnectionManager()
        
        config = {'host': 'localhost'}
        result = await manager.reconnect('new_conn', config)
        
        assert result is True
        assert 'new_conn' in manager.connections


class TestConnectionRetrieval:
    """测试连接获取"""

    @pytest.mark.asyncio
    async def test_get_connection_success(self):
        """测试获取存在的连接"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        connection = manager.get_connection('conn1')
        
        assert connection is not None
        assert isinstance(connection, dict)

    @pytest.mark.asyncio
    async def test_get_nonexistent_connection(self):
        """测试获取不存在的连接"""
        manager = DataConnectionManager()
        
        connection = manager.get_connection('nonexistent')
        
        assert connection is None

    @pytest.mark.asyncio
    async def test_get_all_connections(self):
        """测试获取所有连接"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        all_connections = manager.get_all_connections()
        
        assert len(all_connections) == 2
        assert 'conn1' in all_connections
        assert 'conn2' in all_connections


class TestConnectionStatus:
    """测试连接状态"""

    @pytest.mark.asyncio
    async def test_get_connection_status(self):
        """测试获取连接状态"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        status = manager.get_connection_status('conn1')
        
        assert status == 'active'

    @pytest.mark.asyncio
    async def test_get_status_after_close(self):
        """测试关闭后获取状态"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        await manager.close_connection('conn1')
        status = manager.get_connection_status('conn1')
        
        assert status == 'closed'

    @pytest.mark.asyncio
    async def test_get_status_nonexistent(self):
        """测试获取不存在连接的状态"""
        manager = DataConnectionManager()
        
        status = manager.get_connection_status('nonexistent')
        
        assert status is None


class TestBatchOperations:
    """测试批量操作"""

    @pytest.mark.asyncio
    async def test_check_all_connections(self):
        """测试检查所有连接"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        results = await manager.check_all_connections()
        
        assert len(results) == 2
        assert results['conn1'] is True
        assert results['conn2'] is True

    @pytest.mark.asyncio
    async def test_check_all_connections_empty(self):
        """测试检查空连接列表"""
        manager = DataConnectionManager()
        
        results = await manager.check_all_connections()
        
        assert results == {}

    @pytest.mark.asyncio
    async def test_close_all_connections(self):
        """测试关闭所有连接"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        closed = await manager.close_all_connections()
        
        assert closed == 2
        assert len(manager.connections) == 0

    @pytest.mark.asyncio
    async def test_close_all_connections_empty(self):
        """测试关闭空连接列表"""
        manager = DataConnectionManager()
        
        closed = await manager.close_all_connections()
        
        assert closed == 0


class TestStatistics:
    """测试统计信息"""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self):
        """测试空连接时的统计信息"""
        manager = DataConnectionManager(max_connections=10)
        
        stats = manager.get_statistics()
        
        assert stats['total_connections'] == 0
        assert stats['max_connections'] == 10
        assert stats['utilization'] == 0.0
        assert stats['status_counts'] == {}

    @pytest.mark.asyncio
    async def test_get_statistics_with_connections(self):
        """测试有连接时的统计信息"""
        manager = DataConnectionManager(max_connections=10)
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        stats = manager.get_statistics()
        
        assert stats['total_connections'] == 2
        assert stats['max_connections'] == 10
        assert stats['utilization'] == 0.2
        assert 'status_counts' in stats
        assert stats['status_counts']['active'] == 2

    @pytest.mark.asyncio
    async def test_get_statistics_mixed_status(self):
        """测试混合状态的统计信息"""
        manager = DataConnectionManager(max_connections=10)
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        await manager.close_connection('conn1')
        
        stats = manager.get_statistics()
        
        assert stats['total_connections'] == 1
        assert stats['status_counts'].get('active', 0) == 1
        assert stats['status_counts'].get('closed', 0) == 1

    @pytest.mark.asyncio
    async def test_get_statistics_full_capacity(self):
        """测试满容量时的统计信息"""
        manager = DataConnectionManager(max_connections=2)
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        stats = manager.get_statistics()
        
        assert stats['utilization'] == 1.0
