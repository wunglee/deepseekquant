"""
测试：备用数据源集成与降级日志（专家answer.md第3轮高优任务）

测试覆盖：
1. 数据源健康检查机制
2. 降级日志记录（DataUtils.safe_get_event_data异常处理）
3. 备用数据源回退路径（JoinQuant/Wind/Tushare stub → Yahoo → Mock）
4. 区域化数据源优先级（CN/US/HK/JP/EU/SG）

验收标准：
- 日志中包含event_id、provider、error摘要（专家第3轮要求）
- 健康检查失败时跳过不可用源
- 所有源失败时记录健康状态汇总
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import logging
from io import StringIO

from core_bak_refactored.core.data._fragments.data_utils import DataUtils, EventConfig
from core_bak_refactored.core.data._fragments.historical_data_provider import (
    RealHistoricalDataProvider,
    MockHistoricalDataProvider
)
from core_bak_refactored.core.share.market_enums import MarketCode, DataSource


class TestBackupSourcesAndLogging:
    """备用数据源与降级日志测试"""
    
    @pytest.fixture
    def mock_event(self):
        """创建测试事件配置"""
        return EventConfig(
            event_id='test_event_001',
            index_id='000300.SH',
            event_date='2015-06-15',
            event_type='market_crash',
            expected_decline=-0.43,
            market_id='CN'
        )
    
    @pytest.fixture
    def log_capture(self):
        """捕获日志输出"""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger = logging.getLogger('DeepSeekQuant.DataUtils')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        yield log_stream
        
        logger.removeHandler(handler)
    
    def test_safe_get_event_data_downgrade_logging(self, mock_event, log_capture):
        """测试safe_get_event_data降级日志（异常捕获）"""
        # 创建会抛出异常的mock provider
        failing_provider = Mock()
        failing_provider.get_event_window_data.side_effect = ValueError("数据源连接失败")
        
        # 调用safe_get_event_data
        data, success = DataUtils.safe_get_event_data(
            data_provider=failing_provider,
            event=mock_event
        )
        
        # 验证返回值
        assert success is False
        assert data.empty
        
        # 验证降级日志内容（专家第3轮要求：event_id, provider, error）
        log_output = log_capture.getvalue()
        assert 'safe_get_event_data failed' in log_output
        assert 'event_id=test_event_001' in log_output
        assert 'Mock' in log_output  # provider类型
        assert '数据源连接失败' in log_output  # error摘要
    
    def test_safe_get_event_data_unexpected_format_logging(self, mock_event, log_capture):
        """测试safe_get_event_data非预期格式告警"""
        # 创建返回非预期格式的provider
        unexpected_provider = Mock()
        unexpected_provider.get_event_window_data.return_value = "invalid_format"
        
        # 调用safe_get_event_data
        data, success = DataUtils.safe_get_event_data(
            data_provider=unexpected_provider,
            event=mock_event
        )
        
        # 验证返回值
        assert success is False
        assert data.empty
        
        # 验证告警日志
        log_output = log_capture.getvalue()
        assert 'unexpected format' in log_output
        assert 'event_id=test_event_001' in log_output
        assert 'type=str' in log_output
    
    def test_real_provider_health_check_skips_unavailable(self):
        """测试健康检查：跳过不可用数据源"""
        # 创建带stub适配器的provider
        provider = RealHistoricalDataProvider(
            primary_source='joinquant',  # stub未实现
            backup_sources=['wind', 'yahoo', 'mock'],
            enable_cross_validation=False
        )
        
        # 捕获日志
        import logging
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger = logging.getLogger('DeepSeekQuant.DataFragments')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            # 获取数据（应跳过joinquant和wind，回退到yahoo或mock）
            try:
                data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-15')
                
                # 验证最终成功获取数据（通过yahoo或mock）
                assert not data.empty
                
            except ValueError as e:
                # 如果所有源都失败，验证错误消息包含健康状态
                assert '健康状态=' in str(e)
            
            # 验证日志中包含健康检查跳过记录
            log_output = log_stream.getvalue()
            # stub适配器应该触发"未实现"或"不可用"消息
            assert ('不可用' in log_output) or ('未实现' in log_output) or ('尝试下一数据源' in log_output)
        
        finally:
            logger.removeHandler(handler)
    
    def test_regional_priority_cn_market(self):
        """测试区域化优先级：A股市场优先JoinQuant"""
        provider = RealHistoricalDataProvider(
            primary_source=DataSource.YAHOO.value,
            backup_sources=[DataSource.JOINQUANT.value, DataSource.WIND.value, DataSource.MOCK.value]
        )
        
        # A股指数应优先使用JoinQuant（即使primary是yahoo）
        priority = provider._get_regional_priority('000300.SH')
        
        assert priority[0] == DataSource.JOINQUANT.value
        assert priority[1] == DataSource.TUSHARE.value  # Tushare为A股备用
        assert DataSource.MOCK.value in priority  # 确保有兴底
    
    def test_regional_priority_us_market(self):
        """测试区域化优先级：美股市场优先Yahoo"""
        provider = RealHistoricalDataProvider(
            primary_source=DataSource.JOINQUANT.value,
            backup_sources=[DataSource.YAHOO.value, DataSource.WIND.value, DataSource.MOCK.value]
        )
        
        # 美股指数应优先使用Yahoo
        priority = provider._get_regional_priority('SPX')
        
        assert priority[0] == DataSource.YAHOO.value
    
    def test_regional_priority_hk_market(self):
        """测试区域化优先级：港股市场优先Wind"""
        provider = RealHistoricalDataProvider(
            primary_source=DataSource.YAHOO.value,
            backup_sources=[DataSource.JOINQUANT.value, DataSource.WIND.value, DataSource.MOCK.value]
        )
        
        # 港股指数应优先使用Wind
        priority = provider._get_regional_priority('HSI')
        
        assert priority[0] == DataSource.WIND.value
        assert priority[1] == DataSource.TUSHARE.value  # Tushare为港股备用
    
    def test_all_sources_fail_logs_health_summary(self, log_capture):
        """测试所有数据源失败时记录健康状态汇总"""
        # 创建所有适配器都失败的provider
        provider = RealHistoricalDataProvider(
            primary_source='joinquant',
            backup_sources=['wind', 'yahoo']  # 不包含mock，确保全部失败
        )
        
        # 强制所有适配器返回NotImplementedError
        for adapter in provider._adapters.values():
            if hasattr(adapter, 'get_index_prices'):
                adapter.get_index_prices = Mock(side_effect=NotImplementedError("stub"))
        
        # 尝试获取数据
        with pytest.raises(ValueError) as exc_info:
            provider.get_index_prices('INVALID_INDEX', '2015-01-01', '2015-01-10')
        
        # 验证错误消息包含健康状态汇总
        error_msg = str(exc_info.value)
        assert '健康状态=' in error_msg
        assert 'INVALID_INDEX' in error_msg
    
    def test_mock_provider_always_succeeds(self, mock_event):
        """测试Mock数据源作为兜底始终成功"""
        mock_provider = MockHistoricalDataProvider()
        
        # 获取事件数据
        data, success = DataUtils.safe_get_event_data(
            data_provider=mock_provider,
            event=mock_event
        )
        
        # Mock应始终成功返回数据
        assert success is True
        assert not data.empty
        assert 'close' in data.columns
    
    def test_tushare_stub_in_cn_priority(self):
        """测试Tushare在A股优先级中"""
        provider = RealHistoricalDataProvider(
            primary_source=DataSource.YAHOO.value,
            backup_sources=[DataSource.JOINQUANT.value, DataSource.TUSHARE.value, DataSource.MOCK.value]
        )
        
        priority = provider._get_regional_priority('000001.SH')
        assert DataSource.TUSHARE.value in priority
        assert priority.index(DataSource.TUSHARE.value) < priority.index(DataSource.YAHOO.value)
    
    def test_all_markets_covered(self):
        """测试所有market_config.py中的市场都有对应优先级"""
        from core_bak_refactored.core.share.market_enums import MarketCode, REGIONAL_DATA_SOURCE_PRIORITY
        
        # 所有市场代码都应在优先级映射中
        for market in MarketCode:
            assert market in REGIONAL_DATA_SOURCE_PRIORITY
            priority = REGIONAL_DATA_SOURCE_PRIORITY[market]
            assert len(priority) > 0
            assert DataSource.MOCK in priority  # 每个市场都应有Mock兜底
    
    def test_jp_market_priority(self):
        """测试日本市场优先级"""
        provider = RealHistoricalDataProvider()
        priority = provider._get_regional_priority('9984.T')  # Sony股票
        assert priority[0] == DataSource.YAHOO.value
    
    def test_eu_market_priority(self):
        """测试欧洲市场优先级"""
        provider = RealHistoricalDataProvider()
        priority = provider._get_regional_priority('BMW.DE')  # 宝马股票
        assert priority[0] == DataSource.YAHOO.value
    
    def test_sg_market_priority(self):
        """测试新加坡市场优先级"""
        provider = RealHistoricalDataProvider()
        priority = provider._get_regional_priority('STI.SI')  # 海峡时报指数
        assert priority[0] == DataSource.YAHOO.value


class TestAbnormalHandlingAlerts:
    """异常处理三级告警测试（专家answer.md第1轮5.3节）"""
    
    def test_level1_alert_15_to_20_percent(self):
        """测试Level 1告警：15%-20%误差"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, AlertLevel
        
        validator = UATValidator()
        
        # 触发Level 1告警
        alert = validator.handle_exception(
            prediction_error=0.17,
            event_id='test_event_level1'
        )
        
        assert alert is not None
        assert alert.level == AlertLevel.LEVEL_1
        assert alert.error_range == "15%-20%"
        assert alert.action == "内部记录，下周复核"
        assert alert.report_deadline == "3个工作日内"
    
    def test_level2_alert_20_to_25_percent(self):
        """测试Level 2告警：20%-25%误差"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, AlertLevel
        
        validator = UATValidator()
        
        # 触发Level 2告警
        alert = validator.handle_exception(
            prediction_error=0.22,
            event_id='test_event_level2'
        )
        
        assert alert is not None
        assert alert.level == AlertLevel.LEVEL_2
        assert alert.error_range == "20%-25%"
        assert alert.action == "预警，人工复核"
        assert alert.report_deadline == "24小时内"
    
    def test_level3_alert_above_25_percent(self):
        """测试Level 3告警：>25%误差"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, AlertLevel
        
        validator = UATValidator()
        
        # 触发Level 3告警
        alert = validator.handle_exception(
            prediction_error=0.30,
            event_id='test_event_level3'
        )
        
        assert alert is not None
        assert alert.level == AlertLevel.LEVEL_3
        assert alert.error_range == ">25%"
        assert alert.action == "暂停自动报送，立即干预"
        assert alert.report_deadline == "立即"
    
    def test_no_alert_below_15_percent(self):
        """测试无告警：<15%误差"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator
        
        validator = UATValidator()
        
        # 不应触发告警
        alert = validator.handle_exception(
            prediction_error=0.10,
            event_id='test_event_no_alert'
        )
        
        assert alert is None
    
    def test_alert_history_tracking(self):
        """测试告警历史记录"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, AlertLevel
        
        validator = UATValidator()
        
        # 触发多个告警
        validator.handle_exception(0.16, 'event1')  # Level 1
        validator.handle_exception(0.22, 'event2')  # Level 2
        validator.handle_exception(0.30, 'event3')  # Level 3
        
        # 获取所有告警
        all_alerts = validator.get_alert_history()
        assert len(all_alerts) == 3
        
        # 筛选Level 3告警
        critical_alerts = validator.get_alert_history(level=AlertLevel.LEVEL_3)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].metadata['event_id'] == 'event3'
