"""dashboard.py的单元测试

测试应用层DataQualityDashboard的基础功能
使用Mock方式隔离对领域层的依赖
"""
import pytest
from unittest.mock import Mock, MagicMock
from core_bak_refactored.app.data.dashboard import DataQualityDashboard


class TestDataQualityDashboardBasic:
    """DataQualityDashboard基础功能测试"""
    
    def test_init_with_mock_monitor(self):
        """测试使用Mock的DataQualityMonitor初始化"""
        # 创建Mock的DataQualityMonitor
        mock_monitor = Mock()
        mock_monitor.get_current_status = Mock(return_value={
            'overall_health': 'good',
            'active_alerts': [],
            'last_check': '2024-01-01T00:00:00'
        })
        
        # 创建Dashboard（使用Mock）
        dashboard = DataQualityDashboard(quality_monitor=mock_monitor)
        
        # 验证初始化
        assert dashboard.quality_monitor == mock_monitor
        assert dashboard.update_interval == 300  # 默认间隔
        assert dashboard.scheduler is None  # 默认未配置调度器
    
    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化（传入scheduler）"""
        mock_monitor = Mock()
        mock_scheduler = Mock()
        mock_scheduler.get_status = Mock(return_value={
            'running': True,
            'strategy': 'apscheduler',
            'check_interval': 300,
            'next_run': '2024-01-01T01:00:00'
        })
        
        dashboard = DataQualityDashboard(
            quality_monitor=mock_monitor,
            scheduler=mock_scheduler
        )
        
        assert dashboard.scheduler == mock_scheduler
        assert dashboard.quality_monitor == mock_monitor


class TestDataQualityDashboardFlaskApp:
    """当前的DataQualityDashboard不再直接持有app属性，仅提供启动方法"""
    
    def test_flask_app_created_in_start_dashboard(self):
        """测试Flask应用在start_dashboard中创建"""
        mock_monitor = Mock()
        mock_monitor.get_performance_statistics = Mock(return_value={
            'uptime_seconds': 100,
            'success_rate': 0.95
        })
        mock_monitor.get_alert_history = Mock(return_value=[])
        
        dashboard = DataQualityDashboard(quality_monitor=mock_monitor)
        
        # 验证组件已初始化
        assert dashboard.data_aggregator is not None
        assert dashboard.websocket_handler is not None
        assert dashboard.renderer is not None
    
    def test_flask_routes_registered(self):
        """测试start_dashboard会创建Flask应用并注册路由（不实际启动）"""
        mock_monitor = Mock()
        mock_monitor.get_performance_statistics = Mock(return_value={})
        mock_monitor.get_alert_history = Mock(return_value=[])
        
        dashboard = DataQualityDashboard(quality_monitor=mock_monitor)
        
        # 验证Dashboard具有start_dashboard方法
        assert hasattr(dashboard, 'start_dashboard')
        assert callable(dashboard.start_dashboard)
