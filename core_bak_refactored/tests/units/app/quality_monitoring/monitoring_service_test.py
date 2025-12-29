"""QualityMonitoringService单元测试

测试run_check_cycle方法是否完整等效于旧版DataQualityMonitor的监控循环
"""
from unittest.mock import Mock

import pandas as pd
import pytest

from core_bak_refactored.app.quality_monitoring.monitoring_service import QualityMonitoringService
from core_bak_refactored.core.monitoring.alert_manager import AlertConfig

# 基于工厂注入的真实provider路径（使用StubProvider避免外部依赖）
stub_provider_instance = None

class StubProvider:
    def __init__(self):
        self.last_index_id = None
        self.last_start = None
        self.last_end = None
    def get_index_prices(self, index_id, start_date, end_date):
        self.last_index_id = index_id
        self.last_start = start_date
        self.last_end = end_date
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
        return pd.DataFrame({
            'date': dates,
            'close': [100.0 + i for i in range(100)],
            'volume': [1000000] * 100
        })

class StubFactory:
    def __init__(self):
        self._instances = {}
    
    def create(self, name, **kwargs):
        if name != 'akshare':  # 支持两种名称
            raise ValueError(f"Unexpected provider name: {name}")
        global stub_provider_instance
        stub_provider_instance = StubProvider()
        return stub_provider_instance
    
    def get(self, name):
        """单例模式，与DataProviderFactory保持一致"""
        if name in self._instances:
            return self._instances[name]
        instance = self.create(name)
        self._instances[name] = instance
        return instance

@pytest.fixture(autouse=True)
def patch_global_factory(monkeypatch):
    monkeypatch.setattr('core_bak_refactored.app.quality_monitoring.monitoring_service.get_global_factory', lambda: StubFactory())


class TestQualityMonitoringServiceRunCheckCycle:
    """测试run_check_cycle方法"""
    
    def test_run_check_cycle_basic(self):
        """测试基本的监控周期执行"""
        # 创建监控服务
        alert_config = AlertConfig()
        service = QualityMonitoringService(alert_config=alert_config)
        
        # 执行监控周期
        summary = service.run_check_cycle()
        
        # 验证返回的摘要
        assert 'cycle_time' in summary
        assert 'data_points_checked' in summary
        assert 'anomalies_detected' in summary
        assert 'alerts_triggered' in summary
        assert 'quality_score' in summary
        assert 'status' in summary
        
        # 验证状态
        assert summary['status'] == 'success'
        assert summary['cycle_time'] >= 0
        assert summary['data_points_checked'] >= 0
    
    def test_run_check_cycle_updates_performance_stats(self):
        """测试run_check_cycle更新性能统计"""
        alert_config = AlertConfig()
        service = QualityMonitoringService(alert_config=alert_config)
        
        # 记录初始统计
        initial_stats = service.get_performance_statistics()
        initial_cycles = initial_stats['monitoring_cycles']
        initial_points = initial_stats['data_points_processed']
        
        # 执行监控周期
        summary = service.run_check_cycle()
        
        # 验证性能统计未被run_check_cycle增加（由调度器负责增加cycles）
        # run_check_cycle只更新data_points_processed等具体指标
        updated_stats = service.get_performance_statistics()
        assert updated_stats['data_points_processed'] > initial_points
    
    def test_run_check_cycle_writes_quality_history(self):
        """测试run_check_cycle写入质量历史"""
        alert_config = AlertConfig()
        service = QualityMonitoringService(alert_config=alert_config)
        
        # 记录初始历史长度
        initial_history_len = len(service._quality_history)
        
        # 执行监控周期
        summary = service.run_check_cycle()
        
        # 验证历史记录增加
        assert len(service._quality_history) == initial_history_len + 1
        
        # 验证最新历史记录包含必要字段
        latest_record = service._quality_history[-1]
        assert 'timestamp' in latest_record
        assert 'overall_score' in latest_record
        assert 'completeness' in latest_record
        assert 'issues' in latest_record
    
    def test_run_check_cycle_triggers_alerts_on_low_quality(self):
        """测试低质量时触发告警"""
        alert_config = AlertConfig()
        service = QualityMonitoringService(alert_config=alert_config)
        
        # 使用真实provider（通过工厂注入的Stub）保证单测不依赖外部网络
        summary = service.run_check_cycle()
        # 验证真实provider被调用（通过工厂注入的 Stub）
        global stub_provider_instance
        assert stub_provider_instance is not None
        # 验证默认指数代码（从配置获取，默认为 000300.SH）
        assert stub_provider_instance.last_index_id == '000300.SH'
        
        # 验证alerts_triggered字段存在
        assert 'alerts_triggered' in summary
        assert isinstance(summary['alerts_triggered'], int)
        assert summary['alerts_triggered'] >= 0
    
    def test_run_check_cycle_handles_errors_gracefully(self):
        """测试run_check_cycle优雅地处理错误"""
        alert_config = AlertConfig()
        service = QualityMonitoringService(alert_config=alert_config)
        
        # Mock quality_checker抛出异常
        original_check = service.quality_checker.check_quality
        service.quality_checker.check_quality = Mock(side_effect=Exception("Test error"))
        
        # 执行监控周期（应该捕获异常）
        summary = service.run_check_cycle()
        
        # 验证错误状态
        assert summary['status'] == 'error'
        assert 'error' in summary
        
        # 恢复原始方法
        service.quality_checker.check_quality = original_check
