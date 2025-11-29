"""
数据质量API服务单元测试
"""

import json
from unittest.mock import Mock, patch

import pytest

from core_bak_refactored.app.data_quality.api_service import DataQualityAPIService


class TestDataQualityAPIService:
    """数据质量API服务测试"""
    
    @pytest.fixture
    def mock_quality_monitor(self):
        """Mock质量监控器"""
        monitor = Mock()
        monitor.get_quality_history.return_value = [
            {
                'timestamp': '2024-01-01T12:00:00',
                'overall_score': 0.95,
                'anomaly_count': 2,
                'error_count': 1
            },
            {
                'timestamp': '2024-01-01T13:00:00',
                'overall_score': 0.98,
                'anomaly_count': 1,
                'error_count': 0
            }
        ]
        monitor.get_alert_history.return_value = [
            {'level': 'critical', 'severity': 'high', 'data_source': 'yahoo'},
            {'level': 'warning', 'severity': 'medium', 'data_source': 'tushare'},
            {'level': 'critical', 'severity': 'high', 'data_source': 'yahoo'}
        ]
        monitor.get_performance_statistics.return_value = {
            'uptime_human': '2 days',
            'uptime_seconds': 172800,
            'throughput': 1000,
            'success_rate': 0.98,
            'reliability': 0.95,
            'stability_score': 0.97
        }
        monitor.generate_comprehensive_report.return_value = {
            'report_id': 'test_report_123',
            'quality_analysis': {'avg_score': 0.96},
            'alert_analysis': {'total': 10},
            'performance_analysis': {'throughput': 1000}
        }
        return monitor
    
    @pytest.fixture
    def api_service(self, mock_quality_monitor):
        """创建API服务实例"""
        return DataQualityAPIService(mock_quality_monitor)
    
    @pytest.fixture
    def client(self, api_service):
        """Flask测试客户端"""
        api_service.app.config['TESTING'] = True
        return api_service.app.test_client()
    
    def test_init(self, api_service, mock_quality_monitor):
        """测试初始化"""
        assert api_service.quality_monitor == mock_quality_monitor
        assert api_service.app is not None
    
    def test_get_current_quality_success(self, client):
        """测试获取当前质量数据 - 成功"""
        response = client.get('/api/v1/quality/current?hours=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'data' in data
        assert 'timestamp' in data
        assert 'metadata' in data
        
        metadata = data['metadata']
        assert metadata['data_points'] == 2
        assert metadata['time_range'] == 'last_24_hours'
        assert 'quality_score_avg' in metadata
        assert 'anomaly_count_total' in metadata
    
    def test_get_current_quality_default_hours(self, client):
        """测试获取当前质量数据 - 默认时间范围"""
        response = client.get('/api/v1/quality/current')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['metadata']['time_range'] == 'last_24_hours'
    
    def test_generate_quality_report_json(self, client):
        """测试生成质量报告 - JSON格式"""
        response = client.get('/api/v1/quality/report?period=7d&format=json&include_details=true')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'report' in data
        assert 'timestamp' in data
        assert 'report_id' in data
        
        report = data['report']
        assert report['report_id'] == 'test_report_123'
        assert 'quality_analysis' in report
        assert 'alert_analysis' in report
    
    def test_generate_quality_report_without_details(self, client):
        """测试生成质量报告 - 不包含详细信息"""
        response = client.get('/api/v1/quality/report?include_details=false')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        report = data['report']
        assert 'quality_analysis' not in report
        assert 'alert_analysis' not in report
        assert 'performance_analysis' not in report
    
    def test_get_alerts_no_filter(self, client):
        """测试获取警报 - 无过滤"""
        response = client.get('/api/v1/alerts?hours=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'alerts' in data
        assert 'pagination' in data
        assert 'summary' in data
        
        assert data['summary']['total_alerts'] == 3
        assert len(data['alerts']) == 3
    
    def test_get_alerts_with_level_filter(self, client):
        """测试获取警报 - 按级别过滤"""
        response = client.get('/api/v1/alerts?hours=24&level=critical')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['summary']['total_alerts'] == 2
        assert all(alert['level'] == 'critical' for alert in data['alerts'])
    
    def test_get_alerts_with_severity_filter(self, client):
        """测试获取警报 - 按严重性过滤"""
        response = client.get('/api/v1/alerts?severity=high')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert all(alert['severity'] == 'high' for alert in data['alerts'])
    
    def test_get_alerts_with_pagination(self, client):
        """测试获取警报 - 分页"""
        response = client.get('/api/v1/alerts?page=1&per_page=2')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        pagination = data['pagination']
        assert pagination['page'] == 1
        assert pagination['per_page'] == 2
        assert pagination['total'] == 3
        assert pagination['pages'] == 2
        assert len(data['alerts']) == 2
    
    def test_get_performance(self, client):
        """测试获取性能统计"""
        response = client.get('/api/v1/performance')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'performance' in data
        assert 'timestamp' in data
        
        performance = data['performance']
        assert 'uptime_human' in performance
        assert 'throughput' in performance
        assert 'success_rate' in performance
        assert 'system_health' in performance
        assert 'trend_analysis' in performance
        assert 'recommendations' in performance
    
    def test_get_metrics(self, client):
        """测试获取监控指标"""
        response = client.get('/api/v1/metrics?type=all&time_range=24h&aggregation=hourly')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'metrics' in data
        assert 'metadata' in data
        
        metadata = data['metadata']
        assert metadata['metric_type'] == 'all'
        assert metadata['time_range'] == '24h'
        assert metadata['aggregation'] == 'hourly'
    
    def test_health_check_healthy(self, client):
        """测试健康检查 - 健康状态"""
        response = client.get('/api/v1/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'timestamp' in data
        assert 'components' in data
    
    def test_system_status(self, client):
        """测试系统状态"""
        response = client.get('/api/v1/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'system_status' in data
        assert 'timestamp' in data
        
        status = data['system_status']
        assert 'overall_status' in status
        assert 'performance_metrics' in status
    
    def test_get_config(self, client):
        """测试获取配置"""
        response = client.get('/api/v1/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert 'config' in data
        assert 'timestamp' in data
    
    def test_update_config_success(self, client):
        """测试更新配置 - 成功"""
        new_config = {
            'monitoring': {'threshold': 0.9},
            'api_settings': {'timeout': 60}
        }
        
        response = client.put(
            '/api/v1/config',
            data=json.dumps(new_config),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'success'
        assert data['message'] == '配置更新成功'
    
    def test_update_config_invalid(self, client):
        """测试更新配置 - 无效数据"""
        response = client.put('/api/v1/config', data='', content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['status'] == 'error'
        assert data['error_code'] == 'INVALID_CONFIG'
    
    def test_error_handler_404(self, client):
        """测试404错误处理"""
        response = client.get('/api/v1/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        
        assert data['status'] == 'error'
        assert data['error_code'] == 'ENDPOINT_NOT_FOUND'
    
    def test_error_handler_405(self, client):
        """测试405错误处理"""
        response = client.post('/api/v1/health')
        
        assert response.status_code == 405
        data = json.loads(response.data)
        
        assert data['status'] == 'error'
        assert data['error_code'] == 'METHOD_NOT_ALLOWED'
    
    def test_group_by_level(self, api_service):
        """测试按级别分组"""
        alerts = [
            {'level': 'critical'},
            {'level': 'critical'},
            {'level': 'warning'}
        ]
        
        grouped = api_service._group_by_level(alerts)
        
        assert grouped['critical'] == 2
        assert grouped['warning'] == 1
    
    def test_group_by_severity(self, api_service):
        """测试按严重性分组"""
        alerts = [
            {'severity': 'high'},
            {'severity': 'high'},
            {'severity': 'medium'}
        ]
        
        grouped = api_service._group_by_severity(alerts)
        
        assert grouped['high'] == 2
        assert grouped['medium'] == 1
    
    def test_group_by_source(self, api_service):
        """测试按数据源分组"""
        alerts = [
            {'data_source': 'yahoo'},
            {'data_source': 'yahoo'},
            {'data_source': 'tushare'}
        ]
        
        grouped = api_service._group_by_source(alerts)
        
        assert grouped['yahoo'] == 2
        assert grouped['tushare'] == 1
    
    def test_calculate_system_health_healthy(self, api_service):
        """测试计算系统健康度 - 健康"""
        stats = {
            'success_rate': 0.95,
            'uptime_seconds': 172800,
            'stability_score': 0.98
        }
        
        health = api_service._calculate_system_health(stats)
        
        assert health['status'] == 'healthy'
        assert health['score'] >= 80
        assert 'indicators' in health
    
    def test_calculate_system_health_degraded(self, api_service):
        """测试计算系统健康度 - 降级"""
        stats = {
            'success_rate': 0.70,
            'uptime_seconds': 3600,
            'stability_score': 0.65
        }
        
        health = api_service._calculate_system_health(stats)
        
        assert health['status'] in ['degraded', 'unhealthy']
        assert health['score'] < 80
    
    def test_analyze_performance_trend(self, api_service):
        """测试性能趋势分析"""
        stats = {'success_rate': 0.95, 'throughput': 1000}
        
        trend = api_service._analyze_performance_trend(stats)
        
        assert 'trend' in trend
        assert 'direction' in trend
        assert 'volatility' in trend
        assert 'confidence' in trend
    
    def test_generate_performance_recommendations(self, api_service):
        """测试生成性能建议"""
        stats = {'success_rate': 0.85, 'throughput': 500}
        
        recommendations = api_service._generate_performance_recommendations(stats)
        
        assert isinstance(recommendations, list)
        if stats['success_rate'] < 0.9:
            assert len(recommendations) > 0
            assert recommendations[0]['priority'] == 'high'
