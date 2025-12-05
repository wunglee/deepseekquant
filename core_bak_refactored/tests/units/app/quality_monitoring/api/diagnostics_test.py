"""测试系统诊断运行器

测试范围:
- 系统诊断
- 性能诊断
- 数据质量诊断
- 网络诊断
- 诊断报告生成
"""

import pytest
from core_bak_refactored.app.quality_monitoring.api.diagnostics import DiagnosticsRunner


class DummyMonitor:
    """Mock质量监控器"""
    pass


class TestDiagnosticsRunner:
    """诊断运行器测试套件"""

    def test_run_diagnostics_returns_all_components(self):
        """测试：运行诊断返回所有组件结果"""
        runner = DiagnosticsRunner(DummyMonitor())
        result = runner.run_diagnostics('full')
        
        assert 'system' in result
        assert 'performance' in result
        assert 'data_quality' in result
        assert 'network' in result
        assert 'summary' in result
        assert 'recommendations' in result
        assert 'timestamp' in result

    def test_run_system_diagnostics_returns_completed_status(self):
        """测试：系统诊断返回完成状态"""
        runner = DiagnosticsRunner(DummyMonitor())
        result = runner.run_system_diagnostics()
        
        assert result['status'] == 'completed'
        assert 'results' in result
        assert 'issues_found' in result

    def test_run_performance_diagnostics_checks_key_metrics(self):
        """测试：性能诊断检查关键指标"""
        runner = DiagnosticsRunner(DummyMonitor())
        result = runner.run_performance_diagnostics()
        
        assert result['status'] == 'completed'
        assert 'response_times' in result['results']
        assert 'throughput' in result['results']
        assert 'latency' in result['results']
        assert 'error_rates' in result['results']

    def test_run_data_quality_diagnostics_checks_quality_dimensions(self):
        """测试：数据质量诊断检查质量维度"""
        runner = DiagnosticsRunner(DummyMonitor())
        result = runner.run_data_quality_diagnostics()
        
        assert result['status'] == 'completed'
        assert 'completeness' in result['results']
        assert 'accuracy' in result['results']
        assert 'timeliness' in result['results']
        assert 'consistency' in result['results']

    def test_run_network_diagnostics_checks_connectivity(self):
        """测试：网络诊断检查连接性"""
        runner = DiagnosticsRunner(DummyMonitor())
        result = runner.run_network_diagnostics()
        
        assert result['status'] == 'completed'
        assert 'connectivity' in result['results']
        assert 'bandwidth' in result['results']
        assert 'latency' in result['results']

    def test_generate_diagnostics_summary_no_issues(self):
        """测试：无问题时诊断摘要为健康"""
        runner = DiagnosticsRunner(DummyMonitor())
        diagnostics = {
            'system': {'status': 'completed', 'issues_found': 0},
            'performance': {'status': 'completed', 'issues_found': 0}
        }
        summary = runner.generate_diagnostics_summary(diagnostics)
        
        assert summary['overall_status'] == 'healthy'
        assert summary['total_issues'] == 0

    def test_generate_diagnostics_recommendations_for_critical_status(self):
        """测试：危急状态生成紧急建议"""
        runner = DiagnosticsRunner(DummyMonitor())
        diagnostics = {
            'system': {'status': 'completed', 'issues_found': 10},
            'summary': {'overall_status': 'critical'}
        }
        recommendations = runner.generate_diagnostics_recommendations(diagnostics)
        
        assert len(recommendations) > 0
        assert any(r['priority'] == 'critical' for r in recommendations)

    def test_generate_diagnostics_recommendations_no_issues(self):
        """测试：无问题时生成保持建议"""
        runner = DiagnosticsRunner(DummyMonitor())
        diagnostics = {
            'system': {'status': 'completed', 'issues_found': 0},
            'summary': {'overall_status': 'healthy'}
        }
        recommendations = runner.generate_diagnostics_recommendations(diagnostics)
        
        assert len(recommendations) > 0
        assert any(r['priority'] == 'low' and '保持' in r['action'] for r in recommendations)
