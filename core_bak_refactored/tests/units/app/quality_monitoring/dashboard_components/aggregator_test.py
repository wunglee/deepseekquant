"""Dashboard数据聚合器测试

测试 DashboardDataAggregator 组件
"""

from core_bak_refactored.app.quality_monitoring.dashboard_components.aggregator import DashboardDataAggregator


class DummyMonitor:
    """模拟质量监控器"""
    
    def get_quality_history(self, hours):
        return [
            {
                'timestamp': '2025-11-28T10:00:00',
                'overall_score': 0.95,
                'anomaly_count': 2,
                'errors': {'type1': 1, 'type2': 1},
                'anomaly_details': {}
            },
            {
                'timestamp': '2025-11-28T11:00:00',
                'overall_score': 0.92,
                'anomaly_count': 5,
                'errors': {'type1': 2, 'type2': 3},
                'anomaly_details': {}
            },
            {
                'timestamp': '2025-11-28T12:00:00',
                'overall_score': 0.88,
                'anomaly_count': 12,
                'errors': {'type1': 5, 'type2': 7},
                'anomaly_details': {}
            }
        ]


class TestDashboardDataAggregator:
    """Dashboard数据聚合器测试套件"""

    def test_get_current_quality_data_returns_expected_structure(self):
        """测试：获取当前质量数据返回预期结构"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        result = aggregator.get_current_quality_data()
        
        assert 'current_score' in result
        assert 'average_score' in result
        assert 'total_anomalies' in result
        assert 'data_points' in result
        assert 'trend' in result
        assert 'anomaly_data' in result
        assert 'timestamp' in result

    def test_calculate_quality_trend_returns_trend_data(self):
        """测试：计算质量趋势返回趋势数据"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        quality_data = DummyMonitor().get_quality_history(24)
        
        trend = aggregator.calculate_quality_trend(quality_data)
        
        assert len(trend) == 3
        assert all('timestamp' in item for item in trend)
        assert all('score' in item for item in trend)
        assert all('anomaly_count' in item for item in trend)

    def test_prepare_anomaly_data_filters_anomalies(self):
        """测试：准备异常数据过滤异常"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        quality_data = DummyMonitor().get_quality_history(24)
        
        anomaly_data = aggregator.prepare_anomaly_data(quality_data)
        
        assert len(anomaly_data) == 3  # 所有数据点都有异常
        assert all('count' in item for item in anomaly_data)
        assert all('level' in item for item in anomaly_data)

    def test_determine_anomaly_level_classification(self):
        """测试：异常级别分类正确"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        
        assert aggregator.determine_anomaly_level(0) == 'none'
        assert aggregator.determine_anomaly_level(3) == 'low'
        assert aggregator.determine_anomaly_level(10) == 'medium'
        assert aggregator.determine_anomaly_level(20) == 'high'
        assert aggregator.determine_anomaly_level(35) == 'critical'

    def test_prepare_performance_data_formats_correctly(self):
        """测试：性能数据格式化正确"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        stats = {
            'throughput': 100.5,
            'success_rate': 0.95,
            'accuracy': 0.98,
            'timeliness': 0.92,
            'completeness': 0.97
        }
        
        result = aggregator.prepare_performance_data(stats)
        
        assert result['throughput'] == 100.5
        assert result['reliability'] == 0.95
        assert result['accuracy'] == 0.98
        assert result['timeliness'] == 0.92
        assert result['completeness'] == 0.97

    def test_calculate_error_distribution_aggregates_correctly(self):
        """测试：错误分布计算正确"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        quality_data = DummyMonitor().get_quality_history(24)
        
        error_dist = aggregator.calculate_error_distribution(quality_data)
        
        assert 'type1' in error_dist
        assert 'type2' in error_dist
        assert error_dist['type1'] == 8  # 1 + 2 + 5
        assert error_dist['type2'] == 11  # 1 + 3 + 7

    def test_group_alerts_by_level_groups_correctly(self):
        """测试：警报按级别分组正确"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        alerts = [
            {'level': 'critical'},
            {'level': 'critical'},
            {'level': 'high'},
            {'level': 'medium'},
            {'level': 'medium'},
            {'level': 'low'}
        ]
        
        grouped = aggregator.group_alerts_by_level(alerts)
        
        assert grouped['critical'] == 2
        assert grouped['high'] == 1
        assert grouped['medium'] == 2
        assert grouped['low'] == 1

    def test_get_report_data_returns_placeholder(self):
        """测试：获取报告数据返回占位符"""
        aggregator = DashboardDataAggregator(DummyMonitor())
        
        result = aggregator.get_report_data('test-report-123')
        
        assert result['report_id'] == 'test-report-123'
        assert result['status'] == 'completed'
        assert 'data' in result

    def test_empty_quality_data_handling(self):
        """测试：空质量数据处理"""
        
        class EmptyMonitor:
            def get_quality_history(self, hours):
                return []
        
        aggregator = DashboardDataAggregator(EmptyMonitor())
        result = aggregator.get_current_quality_data()
        
        assert result['current_score'] == 0
        assert result['average_score'] == 0
        assert result['total_anomalies'] == 0
        assert result['data_points'] == 0
