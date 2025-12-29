"""数据质量API控制器测试

测试范围：
- get_quality_current: 获取质量数据并计算元数据
- get_alerts_with_pagination: 警报过滤和分页
- get_enhanced_performance: 性能增强统计
- 各种辅助方法：分组、健康计算等
"""

from core_bak_refactored.app.quality_monitoring.api.controllers import DataQualityControllers


class DummyMonitor:
    """Mock质量监控器"""
    
    def get_quality_history(self, hours):
        return [
            {'timestamp': '2024-01-01T12:00:00', 'overall_score': 0.9, 'anomaly_count': 2},
            {'timestamp': '2024-01-01T13:00:00', 'overall_score': 0.95, 'anomaly_count': 1}
        ]
    
    def get_alert_history(self, hours):
        return [
            {'level': 'critical', 'severity': 'high', 'data_source': 'yahoo'},
            {'level': 'warning', 'severity': 'medium', 'data_source': 'tushare'},
            {'level': 'critical', 'severity': 'high', 'data_source': 'alphavantage'},
            {'level': 'warning', 'severity': 'low', 'data_source': 'yahoo'}
        ]
    
    def get_performance_statistics(self):
        return {
            'success_rate': 0.95,
            'uptime_seconds': 86400,
            'throughput': 1000,
            'stability_score': 0.92
        }


class TestDataQualityControllers:
    """数据质量控制器测试套件"""
    
    def test_get_quality_current_returns_data_and_metadata(self):
        """测试：获取质量数据带元数据"""
        controller = DataQualityControllers(DummyMonitor())
        result = controller.get_quality_current(24)
        
        # 验证结构
        assert 'data' in result
        assert 'metadata' in result
        
        # 验证数据
        assert len(result['data']) == 2
        
        # 验证元数据
        metadata = result['metadata']
        assert metadata['data_points'] == 2
        assert metadata['time_range'] == 'last_24_hours'
        assert 'quality_score_avg' in metadata
        assert 'anomaly_count_total' in metadata
        assert metadata['anomaly_count_total'] == 3  # 2 + 1
    
    def test_get_alerts_with_pagination_no_filter(self):
        """测试：警报分页（无过滤）"""
        controller = DataQualityControllers(DummyMonitor())
        result = controller.get_alerts_with_pagination(24, page=1, per_page=2)
        
        # 验证分页
        assert len(result['alerts']) == 2
        assert result['pagination']['page'] == 1
        assert result['pagination']['total'] == 4
        assert result['pagination']['pages'] == 2
        
        # 验证摘要
        assert result['summary']['total_alerts'] == 4
    
    def test_get_alerts_with_pagination_level_filter(self):
        """测试：警报按级别过滤"""
        controller = DataQualityControllers(DummyMonitor())
        result = controller.get_alerts_with_pagination(24, level='critical')
        
        # 应该只有critical警报
        assert result['summary']['total_alerts'] == 2
        for alert in result['alerts']:
            assert alert['level'] == 'critical'
    
    def test_get_alerts_with_pagination_severity_filter(self):
        """测试：警报按严重性过滤"""
        controller = DataQualityControllers(DummyMonitor())
        result = controller.get_alerts_with_pagination(24, severity='high')
        
        assert result['summary']['total_alerts'] == 2
        for alert in result['alerts']:
            assert alert['severity'] == 'high'
    
    def test_get_alerts_with_pagination_source_filter(self):
        """测试：警报按数据源过滤"""
        controller = DataQualityControllers(DummyMonitor())
        result = controller.get_alerts_with_pagination(24, data_source='yahoo')
        
        assert result['summary']['total_alerts'] == 2
        for alert in result['alerts']:
            assert alert['data_source'] == 'yahoo'
    
    def test_get_enhanced_performance(self):
        """测试：增强性能统计"""
        controller = DataQualityControllers(DummyMonitor())
        result = controller.get_enhanced_performance()
        
        # 验证原始数据
        assert result['success_rate'] == 0.95
        assert result['uptime_seconds'] == 86400
        
        # 验证增强数据
        assert 'system_health' in result
        assert 'trend_analysis' in result
    
    def test_group_by_level(self):
        """测试：按级别分组"""
        controller = DataQualityControllers(DummyMonitor())
        alerts = controller._qm.get_alert_history(24)
        grouped = controller.group_by_level(alerts)
        
        assert grouped['critical'] == 2
        assert grouped['warning'] == 2
    
    def test_group_by_severity(self):
        """测试：按严重性分组"""
        controller = DataQualityControllers(DummyMonitor())
        alerts = controller._qm.get_alert_history(24)
        grouped = controller.group_by_severity(alerts)
        
        assert grouped['high'] == 2
        assert grouped['medium'] == 1
        assert grouped['low'] == 1
    
    def test_group_by_source(self):
        """测试：按数据源分组"""
        controller = DataQualityControllers(DummyMonitor())
        alerts = controller._qm.get_alert_history(24)
        grouped = controller.group_by_source(alerts)
        
        assert grouped['yahoo'] == 2
        assert grouped['tushare'] == 1
        assert grouped['alphavantage'] == 1
    
    def test_calculate_system_health_healthy(self):
        """测试：计算系统健康度（健康）"""
        controller = DataQualityControllers(DummyMonitor())
        stats = {'success_rate': 0.95, 'uptime_seconds': 86400, 'stability_score': 0.92}
        health = controller.calculate_system_health(stats)
        
        assert health['status'] == 'healthy'
        assert health['score'] >= 80
        assert 'indicators' in health
    
    def test_calculate_system_health_degraded(self):
        """测试：计算系统健康度（降级）"""
        controller = DataQualityControllers(DummyMonitor())
        stats = {'success_rate': 0.7, 'uptime_seconds': 3600, 'stability_score': 0.65}
        health = controller.calculate_system_health(stats)
        
        assert health['status'] in ['degraded', 'unhealthy']
        assert health['score'] < 80
    
    def test_analyze_performance_trend(self):
        """测试：分析性能趋势"""
        controller = DataQualityControllers(DummyMonitor())
        stats = {'success_rate': 0.95, 'throughput': 1000}
        trend = controller.analyze_performance_trend(stats)
        
        assert 'trend' in trend
        assert 'change_rate' in trend
        assert 'prediction' in trend
