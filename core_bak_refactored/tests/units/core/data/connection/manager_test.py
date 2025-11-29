import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from core_bak_refactored.core.data.connection.manager import DataConnectionManager


class TestDataConnectionManager:
    """测试数据连接管理器。"""

    @pytest.mark.asyncio
    async def test_init(self):
        """测试初始化。"""
        manager = DataConnectionManager(max_connections=5)
        
        assert manager.max_connections == 5
        assert manager.connections == {}
        assert manager.connection_status == {}

    @pytest.mark.asyncio
    async def test_create_connection(self):
        """测试创建连接。"""
        manager = DataConnectionManager()
        
        config = {'host': 'localhost', 'port': 5432}
        result = await manager.create_connection('conn1', config)
        
        assert result is True
        assert 'conn1' in manager.connections
        assert manager.connection_status['conn1'] == 'active'

    @pytest.mark.asyncio
    async def test_create_connection_max_limit(self):
        """测试达到最大连接数。"""
        manager = DataConnectionManager(max_connections=2)
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        result = await manager.create_connection('conn3', {})
        
        assert result is False
        assert len(manager.connections) == 2

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """测试关闭连接。"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        result = await manager.close_connection('conn1')
        
        assert result is True
        assert 'conn1' not in manager.connections
        assert manager.connection_status['conn1'] == 'closed'

    @pytest.mark.asyncio
    async def test_close_nonexistent_connection(self):
        """测试关闭不存在的连接。"""
        manager = DataConnectionManager()
        
        result = await manager.close_connection('nonexistent')
        
        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_health(self):
        """测试健康检查。"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        result = await manager.check_connection_health('conn1')
        
        assert result is True
        assert manager.connection_status['conn1'] == 'active'

    @pytest.mark.asyncio
    async def test_check_health_nonexistent(self):
        """测试检查不存在连接的健康状态。"""
        manager = DataConnectionManager()
        
        result = await manager.check_connection_health('nonexistent')
        
        assert result is False

    @pytest.mark.asyncio
    async def test_reconnect(self):
        """测试重连。"""
        manager = DataConnectionManager()
        
        config1 = {'version': 1}
        config2 = {'version': 2}
        
        await manager.create_connection('conn1', config1)
        result = await manager.reconnect('conn1', config2)
        
        assert result is True
        assert 'conn1' in manager.connections

    @pytest.mark.asyncio
    async def test_get_connection(self):
        """测试获取连接。"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        connection = manager.get_connection('conn1')
        
        assert connection is not None

    @pytest.mark.asyncio
    async def test_get_connection_status(self):
        """测试获取连接状态。"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        status = manager.get_connection_status('conn1')
        
        assert status == 'active'

    @pytest.mark.asyncio
    async def test_check_all_connections(self):
        """测试检查所有连接。"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        results = await manager.check_all_connections()
        
        assert len(results) == 2
        assert results['conn1'] is True
        assert results['conn2'] is True

    @pytest.mark.asyncio
    async def test_close_all_connections(self):
        """测试关闭所有连接。"""
        manager = DataConnectionManager()
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        closed = await manager.close_all_connections()
        
        assert closed == 2
        assert len(manager.connections) == 0

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """测试获取统计信息。"""
        manager = DataConnectionManager(max_connections=10)
        
        await manager.create_connection('conn1', {})
        await manager.create_connection('conn2', {})
        
        stats = manager.get_statistics()
        
        assert stats['total_connections'] == 2
        assert stats['max_connections'] == 10
        assert stats['utilization'] == 0.2
        assert 'status_counts' in stats
