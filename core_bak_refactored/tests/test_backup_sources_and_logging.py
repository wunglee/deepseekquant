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


class TestCrossValidation:
    """数据质量交叉验证测试（专家answer.md第3轮5.1节）"""
    
    def test_cross_validate_mock_vs_mock(self):
        """测试Mock数据交叉验证基础功能"""
        provider = RealHistoricalDataProvider(
            primary_source=DataSource.MOCK.value,
            enable_cross_validation=True
        )
        
        # 获取两份数据进行比较（Mock数据有随机性）
        mock_provider = provider._adapters[DataSource.MOCK.value]
        data_a = mock_provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-15')
        data_b = mock_provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-15')
        
        # 直接调用比较方法
        comparison = provider._compare_two_sources(data_a, data_b, 'mock_a', 'mock_b')
        
        # 验证比较结构完整性
        assert 'passed' in comparison
        assert 'overlap_days' in comparison
        assert comparison['overlap_days'] > 0
        assert 'daily_divergence' in comparison
        assert 'mean_divergence' in comparison
        assert 'std_divergence' in comparison
    
    def test_daily_divergence_threshold(self):
        """测试逐日差异30%阈值"""
        provider = RealHistoricalDataProvider()
        
        # 构造测试数据：价格差异超过30%
        data_a = pd.DataFrame({
            'date': pd.date_range('2015-06-01', periods=10),
            'close': [100] * 10,
            'volume': [1000000] * 10
        })
        
        data_b = pd.DataFrame({
            'date': pd.date_range('2015-06-01', periods=10),
            'close': [140] * 10,  # 40%差异
            'volume': [1000000] * 10
        })
        
        comparison = provider._compare_two_sources(data_a, data_b, 'source_a', 'source_b')
        
        # 所有日期都有40%差异，应该触发
        assert comparison['daily_divergence']['ratio'] == 1.0
        assert not comparison['daily_divergence']['passed']
    
    def test_window_statistics_threshold(self):
        """测试窗口统计量阈值（均值3%，标准差10%）"""
        provider = RealHistoricalDataProvider()
        
        # 构造测试数据：均值差异>3%但<30%逐日差异
        data_a = pd.DataFrame({
            'date': pd.date_range('2015-06-01', periods=10),
            'close': [100 + i for i in range(10)],  # 100-109
            'volume': [1000000] * 10
        })
        
        data_b = pd.DataFrame({
            'date': pd.date_range('2015-06-01', periods=10),
            'close': [105 + i for i in range(10)],  # 105-114 (均值差异~4.8%)
            'volume': [1000000] * 10
        })
        
        comparison = provider._compare_two_sources(data_a, data_b, 'source_a', 'source_b')
        
        # 均值差异应>3%
        assert comparison['mean_divergence']['diff_pct'] > 0.03
        assert not comparison['mean_divergence']['passed']
    
    def test_cross_validation_log_tracking(self):
        """测试交叉验证历史记录"""
        provider = RealHistoricalDataProvider(
            primary_source=DataSource.MOCK.value,
            enable_cross_validation=True
        )
        
        # 执行多次验证（使用Mock避免Yahoo限流）
        mock_adapter = provider._adapters[DataSource.MOCK.value]
        data1 = mock_adapter.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
        data2 = mock_adapter.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
        
        comparison1 = provider._compare_two_sources(data1, data2, 'mock', 'mock')
        provider._cross_validation_log.append({
            'index_id': '000300.SH',
            'date_range': '2015-06-01 to 2015-06-10',
            'result': {'comparisons': [comparison1]}
        })
        
        data3 = mock_adapter.get_index_prices('000001.SH', '2015-07-01', '2015-07-10')
        data4 = mock_adapter.get_index_prices('000001.SH', '2015-07-01', '2015-07-10')
        
        comparison2 = provider._compare_two_sources(data3, data4, 'mock', 'mock')
        provider._cross_validation_log.append({
            'index_id': '000001.SH',
            'date_range': '2015-07-01 to 2015-07-10',
            'result': {'comparisons': [comparison2]}
        })
        
        # 获取历史记录
        log = provider.get_cross_validation_log()
        
        assert len(log) == 2
        assert log[0]['index_id'] == '000300.SH'
        assert log[1]['index_id'] == '000001.SH'
        assert 'result' in log[0]
    
    def test_insufficient_sources_handling(self):
        """测试数据源不足时的处理"""
        provider = RealHistoricalDataProvider()
        
        # 只提供一个数据源
        report = provider.cross_validate_sources(
            '000300.SH',
            '2015-06-01',
            '2015-06-15',
            sources=[DataSource.MOCK.value]
        )
        
        # 数据源不足应默认通过
        assert report['passed'] is True
        assert report['reason'] == 'insufficient_sources'
    
    def test_cross_validation_with_enable_flag(self):
        """测试enable_cross_validation标志控制"""
        # 启用交叉验证
        provider_enabled = RealHistoricalDataProvider(enable_cross_validation=True)
        assert provider_enabled.enable_cross_validation is True
        
        # 禁用交叉验证（默认）
        provider_disabled = RealHistoricalDataProvider(enable_cross_validation=False)
        assert provider_disabled.enable_cross_validation is False


class TestUATReportGeneration:
    """UAT报告生成测试（专家answer.md第3轮5.2节）"""
    
    def test_generate_basic_uat_report(self):
        """测试基础UAT报告生成"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, UATResult
        
        validator = UATValidator()
        
        # 模拟测试结果
        test_results = {
            'weighted_average_error': UATResult(
                test_item='weighted_average_error',
                passed=True,
                actual_value=0.12,
                threshold=0.15
            ),
            'cross_market_consistency': UATResult(
                test_item='cross_market_consistency',
                passed=True,
                actual_value=0.87,
                threshold=0.85
            ),
            'data_quality': UATResult(
                test_item='data_quality',
                passed=False,
                actual_value=0.88,
                threshold=0.90
            )
        }
        
        # 生成报告
        report = validator.generate_uat_report(test_results)
        
        # 验证报告结构
        assert 'uat_version' in report
        assert 'overall_status' in report
        assert 'core_metrics' in report
        assert 'abnormal_handling' in report
        assert 'cross_validation' in report
        assert 'recommendations' in report
        
        # 验证核心指标
        assert report['core_metrics']['total_tests'] == 3
        assert report['core_metrics']['passed_tests'] == 2
        assert report['core_metrics']['failed_tests'] == 1
        assert abs(report['core_metrics']['pass_rate'] - 0.667) < 0.01
        
        # 验证总体状态（有失败测试）
        assert report['overall_status']['passed'] == False
        assert report['overall_status']['core_tests_passed'] == False
    
    def test_uat_report_with_abnormal_handling(self):
        """测试包含异常处置记录的UAT报告"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, UATResult, AlertLevel
        
        validator = UATValidator()
        
        # 触发多个告警
        validator.handle_exception(0.17, 'event1')  # Level 1
        validator.handle_exception(0.22, 'event2')  # Level 2
        validator.handle_exception(0.30, 'event3')  # Level 3
        
        test_results = {
            'test1': UATResult('test1', True, 0.10, 0.15)
        }
        
        # 生成报告
        report = validator.generate_uat_report(test_results)
        
        # 验证异常处置记录
        assert report['abnormal_handling']['total_alerts'] == 3
        assert report['abnormal_handling']['level_breakdown']['LEVEL_1'] == 1
        assert report['abnormal_handling']['level_breakdown']['LEVEL_2'] == 1
        assert report['abnormal_handling']['level_breakdown']['LEVEL_3'] == 1
        
        # 验证告警详情
        assert len(report['abnormal_handling']['alert_details']) == 3
        assert report['abnormal_handling']['alert_details'][0]['level'] == 'LEVEL_1'
        assert report['abnormal_handling']['alert_details'][1]['level'] == 'LEVEL_2'
        assert report['abnormal_handling']['alert_details'][2]['level'] == 'LEVEL_3'
        
        # Level 3告警导致总体未通过
        assert report['overall_status']['passed'] == False
        assert report['overall_status']['no_critical_alerts'] == False
    
    def test_uat_report_with_cross_validation(self):
        """测试包含交叉验证结果的UAT报告"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, UATResult
        
        validator = UATValidator()
        
        test_results = {
            'test1': UATResult('test1', True, 0.10, 0.15)
        }
        
        # 模拟交叉验证结果
        cross_validation = {
            'enabled': True,
            'passed': True,
            'sources_compared': ['yahoo', 'mock'],
            'comparisons': [{
                'source_a': 'yahoo',
                'source_b': 'mock',
                'passed': True,
                'overlap_days': 100
            }]
        }
        
        # 生成报告
        report = validator.generate_uat_report(test_results, cross_validation_results=cross_validation)
        
        # 验证交叉验证集成
        assert report['cross_validation']['enabled'] == True
        assert report['cross_validation']['passed'] == True
        assert len(report['cross_validation']['sources_compared']) == 2
        
        # 交叉验证通过，总体应通过
        assert report['overall_status']['cross_validation_passed'] == True
    
    def test_uat_report_recommendations(self):
        """测试UAT报告建议生成"""
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator, UATResult
        
        validator = UATValidator()
        
        # 场景1：所有测试通过
        test_results_pass = {
            'test1': UATResult('test1', True, 0.10, 0.15)
        }
        report = validator.generate_uat_report(test_results_pass)
        assert any('正常' in rec for rec in report['recommendations'])
        
        # 场景2：有失败测试
        validator2 = UATValidator()
        test_results_fail = {
            'data_quality': UATResult('data_quality', False, 0.88, 0.90)
        }
        report2 = validator2.generate_uat_report(test_results_fail)
        assert any('data_quality' in rec for rec in report2['recommendations'])
        
        # 场景3：有Level 3告警
        validator3 = UATValidator()
        validator3.handle_exception(0.30, 'critical_event')
        report3 = validator3.generate_uat_report(test_results_pass)
        assert any('Level 3' in rec or '严重' in rec for rec in report3['recommendations'])
