"""测试系统指标收集器

测试范围:
- 系统指标获取
- 资源利用率监控
- 网络IO统计
- 性能建议生成
"""

import pytest
from core_bak_refactored.app.data.api.system_metrics import MetricsCollector


class DummyMonitor:
    """Mock质量监控器"""
    pass


class TestMetricsCollector:
    """系统指标收集器测试套件"""

    def test_get_system_metrics_returns_expected_structure(self):
        """测试：获取系统指标返回预期结构"""
        collector = MetricsCollector(DummyMonitor())
        result = collector.get_system_metrics('all', '24h', 'hourly')
        
        assert 'metric_type' in result
        assert result['metric_type'] == 'all'
        assert 'time_range' in result
        assert result['time_range'] == '24h'
        assert 'aggregation' in result
        assert 'data' in result
        assert 'summary' in result

    def test_get_resource_utilization_contains_cpu_memory_disk(self):
        """测试：资源利用率包含CPU、内存、磁盘信息"""
        collector = MetricsCollector(DummyMonitor())
        result = collector.get_resource_utilization()
        
        if 'error' not in result:
            assert 'cpu' in result
            assert 'memory' in result
            assert 'disk' in result
            assert 'percent' in result['cpu']
            assert 'percent' in result['memory']
            assert 'percent' in result['disk']

    def test_get_network_io_contains_bytes_and_packets(self):
        """测试：网络IO包含字节和数据包统计"""
        collector = MetricsCollector(DummyMonitor())
        result = collector.get_network_io()
        
        if 'error' not in result:
            assert 'bytes_sent' in result
            assert 'bytes_recv' in result
            assert 'packets_sent' in result
            assert 'packets_recv' in result
            assert 'errin' in result
            assert 'errout' in result

    def test_generate_performance_recommendations_low_success_rate(self):
        """测试：低成功率触发性能建议"""
        collector = MetricsCollector(DummyMonitor())
        stats = {'success_rate': 0.8, 'avg_processing_time': 2.0}
        recommendations = collector.generate_performance_recommendations(stats)
        
        assert len(recommendations) >= 1
        assert any(r['priority'] == 'high' and '成功率' in r['reason'] for r in recommendations)

    def test_generate_performance_recommendations_slow_processing(self):
        """测试：处理时间长触发性能建议"""
        collector = MetricsCollector(DummyMonitor())
        stats = {'success_rate': 0.95, 'avg_processing_time': 6.0}
        recommendations = collector.generate_performance_recommendations(stats)
        
        assert len(recommendations) >= 1
        assert any('处理时间' in r['reason'] for r in recommendations)

    def test_generate_performance_recommendations_good_stats(self):
        """测试：良好性能不触发建议"""
        collector = MetricsCollector(DummyMonitor())
        stats = {'success_rate': 0.95, 'avg_processing_time': 2.0}
        recommendations = collector.generate_performance_recommendations(stats)
        
        assert len(recommendations) == 0

    def test_generate_health_recommendations_critical(self):
        """测试：严重健康问题触发危急建议"""
        collector = MetricsCollector(DummyMonitor())
        recommendations = collector.generate_health_recommendations(50.0, {})
        
        assert len(recommendations) >= 1
        assert any(r['priority'] == 'critical' for r in recommendations)

    def test_generate_health_recommendations_warning(self):
        """测试：一般健康问题触发警告建议"""
        collector = MetricsCollector(DummyMonitor())
        recommendations = collector.generate_health_recommendations(70.0, {})
        
        assert len(recommendations) >= 1
        assert any(r['priority'] == 'high' for r in recommendations)

    def test_generate_health_recommendations_good(self):
        """测试：良好健康度不触发建议"""
        collector = MetricsCollector(DummyMonitor())
        recommendations = collector.generate_health_recommendations(85.0, {})
        
        assert len(recommendations) == 0
