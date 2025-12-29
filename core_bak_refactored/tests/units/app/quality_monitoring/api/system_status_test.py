"""测试系统状态管理器

测试范围:
- 系统状态获取
- 维护模式启用/禁用
- 维护模式状态检查
"""

from core_bak_refactored.app.quality_monitoring.api.system_status import SystemStatusManager


class DummyMonitor:
    """Mock质量监控器"""
    pass


class TestSystemStatusManager:
    """系统状态管理器测试套件"""

    def test_get_system_status_returns_expected_fields(self):
        """测试：获取系统状态返回预期字段"""
        manager = SystemStatusManager(DummyMonitor())
        status = manager.get_system_status()
        
        assert 'service' in status
        assert 'version' in status
        assert 'status' in status
        assert 'uptime' in status
        assert 'maintenance_mode' in status
        assert 'components' in status
        assert 'timestamp' in status

    def test_enable_maintenance_mode_success(self):
        """测试：成功启用维护模式"""
        manager = SystemStatusManager(DummyMonitor())
        success = manager.enable_maintenance_mode(3600)
        
        assert success is True
        assert manager.is_maintenance_mode() is True

    def test_disable_maintenance_mode_success(self):
        """测试：成功禁用维护模式"""
        manager = SystemStatusManager(DummyMonitor())
        manager.enable_maintenance_mode(3600)
        success = manager.disable_maintenance_mode()
        
        assert success is True
        assert manager.is_maintenance_mode() is False

    def test_maintenance_mode_affects_system_status(self):
        """测试：维护模式影响系统状态"""
        manager = SystemStatusManager(DummyMonitor())
        
        # 未启用维护模式
        status = manager.get_system_status()
        assert status['status'] == 'operational'
        
        # 启用维护模式
        manager.enable_maintenance_mode(3600)
        status = manager.get_system_status()
        assert status['status'] == 'maintenance'
        assert status['maintenance_remaining'] is not None

    def test_is_maintenance_mode_when_not_enabled(self):
        """测试：未启用维护模式时检查返回False"""
        manager = SystemStatusManager(DummyMonitor())
        assert manager.is_maintenance_mode() is False

    def test_system_status_includes_components(self):
        """测试：系统状态包含组件信息"""
        manager = SystemStatusManager(DummyMonitor())
        status = manager.get_system_status()
        
        assert 'quality_monitor' in status['components']
        assert 'api_service' in status['components']
        assert 'data_fetcher' in status['components']
