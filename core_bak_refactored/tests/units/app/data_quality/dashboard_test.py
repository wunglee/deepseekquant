"""dashboard.py的单元测试

测试应用层DataQualityDashboard的基础功能
使用Mock方式隔离对领域层的依赖
"""
import pytest
from unittest.mock import Mock, MagicMock
from core_bak_refactored.app.data_quality.dashboard import DataQualityDashboard


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
        assert dashboard.port == 5000  # 默认端口
    
    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化"""
        mock_monitor = Mock()
        custom_config = {
            'port': 8080,
            'host': '0.0.0.0',
            'debug': False
        }
        
        dashboard = DataQualityDashboard(
            quality_monitor=mock_monitor,
            config=custom_config
        )
        
        assert dashboard.port == 8080
        assert dashboard.host == '0.0.0.0'
        assert dashboard.debug == False


class TestDataQualityDashboardFlaskApp:
    """DataQualityDashboard Flask应用测试"""
    
    def test_flask_app_created(self):
        """测试Flask应用创建"""
        mock_monitor = Mock()
        dashboard = DataQualityDashboard(quality_monitor=mock_monitor)
        
        # 验证Flask应用已创建
        assert dashboard.app is not None
        assert hasattr(dashboard.app, 'route')
    
    def test_flask_routes_registered(self):
        """测试Flask路由注册"""
        mock_monitor = Mock()
        mock_monitor.get_current_status = Mock(return_value={'status': 'ok'})
        
        dashboard = DataQualityDashboard(quality_monitor=mock_monitor)
        
        # 验证路由已注册（通过检查app的url_map）
        routes = [rule.rule for rule in dashboard.app.url_map.iter_rules()]
        
        # 至少应该有根路由
        assert any('/' in route for route in routes)
