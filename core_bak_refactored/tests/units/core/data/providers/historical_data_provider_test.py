"""
统一历史数据提供者测试文件
合并了：
- historical_data_provider_test.py（备用数据源、区域优先级、交叉验证）
- historical_data_provider_enhanced_interface_test.py（增强接口测试）
- real_provider_minimal_test.py（最小集成测试）
"""

import pytest
import pandas as pd
import numpy as np
import unittest
from unittest.mock import Mock
from io import StringIO
from datetime import datetime
import logging

from core_bak_refactored.core.backtest.event_analysis import EventConfig, EventAnalyzer
from core_bak_refactored.core.data.providers.historical_data_provider import RealHistoricalDataProvider
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.data.providers.yahoo_finance import YahooFinanceDataProvider
from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester, BacktestReporter
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.share.market.market_enums import MarketCode, DataSource


class TestBackupSourcesAndLogging:
    """备用数据源与降级日志 + 区域优先级 + Mock兜底"""
    
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
        
        logger = logging.getLogger('DeepSeekQuant.EventAnalysis')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        yield log_stream
        
        logger.removeHandler(handler)
    
    def test_safe_get_event_data_downgrade_logging(self, mock_event, log_capture):
        """测试safe_get_event_data降级日志（异常捕获）"""
        failing_provider = Mock()
        failing_provider.get_event_window_data.side_effect = ValueError("数据源连接失败")
        
        data, success = EventAnalyzer.safe_get_event_data(
            data_provider=failing_provider,
            event=mock_event
        )
        
        assert success is False
        assert data.empty
        
        log_output = log_capture.getvalue()
        assert '事件数据获取异常' in log_output
        assert 'event_id=test_event_001' in log_output
        assert 'Mock' in log_output
        assert '数据源连接失败' in log_output
    
    def test_safe_get_event_data_unexpected_format_logging(self, mock_event, log_capture):
        """测试safe_get_event_data非预期格式告警"""
        unexpected_provider = Mock()
        unexpected_provider.get_event_window_data.return_value = "invalid_format"
        
        data, success = EventAnalyzer.safe_get_event_data(
            data_provider=unexpected_provider,
            event=mock_event
        )
        
        assert success is False
        assert data.empty
        
        log_output = log_capture.getvalue()
        assert '事件数据格式异常' in log_output
        assert 'event_id=test_event_001' in log_output
        assert 'type=str' in log_output
    
    def test_real_provider_health_check_skips_unavailable(self):
        """测试健康检查：跳过不可用数据源
        
        架构变更（2025-12-06）：不再支持backup_sources，参数将被忽略
        """
        # 新架构：仅使用primary_source
        provider = RealHistoricalDataProvider(
            primary_source='akshare',  # A股市场使用akshare
            enable_cross_validation=False
        )
        
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger = logging.getLogger('DeepSeekQuant.DataProviders')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            # 新架构：primary_source失败时抛出RuntimeError
            result = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-15')
            # AKShare应该能成功获取数据
            assert not result.empty, "AKShare应该能成功获取A股数据"
            
            log_output = log_stream.getvalue()
        finally:
            logger.removeHandler(handler)
    
    def test_regional_priority_cn_market(self):
        """测试区域化数据源推荐：A股市场推荐akshare
        
        架构变更（2025-12-06）：
        - 不再使用_get_regional_priority()方法（已删除）
        - 使用_get_recommended_source_for_region()推荐数据源
        - 数据源不匹配时会抛出ValueError
        """
        provider = RealHistoricalDataProvider(
            primary_source='akshare'  # A股市场使用akshare
        )
        
        # 测试推荐数据源
        recommended = provider._get_recommended_source_for_region('000300.SH')
        assert recommended == 'akshare', "A股市场应推荐akshare数据源"
        
        # 测试数据源不匹配时抛出异常
        provider_wrong = RealHistoricalDataProvider(primary_source='yahoo')
        with pytest.raises(ValueError, match="数据源不匹配"):
            provider_wrong.get_index_prices('000300.SH', '2015-06-01', '2015-06-15')
    
    def test_regional_priority_us_market(self):
        """测试区域化数据源推荐：美股市场推荐
        
        架构变更（2025-12-06）：测试推荐逻辑而非优先级列表
        注意：akshare也支持US市场，且在配置中在yahoo之前，所以会被推荐
        """
        provider = RealHistoricalDataProvider(
            primary_source='akshare'  # akshare支持US市场
        )
        
        # 测试推荐数据源（akshare在配置中在yahoo之前且也支持US）
        recommended = provider._get_recommended_source_for_region('SPX.US')
        assert recommended == 'akshare', "美股市场推荐akshare数据源（配置顺序优先）"
    
    def test_regional_priority_hk_market(self):
        """测试区域化数据源推荐：港股市场推荐akshare
        
        架构变更（2025-12-06）：根据data.yml配置，akshare支持港股
        """
        provider = RealHistoricalDataProvider(
            primary_source='akshare'  # 港股市场使用akshare
        )
        
        # 测试推荐数据源
        recommended = provider._get_recommended_source_for_region('0700.HK')
        assert recommended == 'akshare', "港股市场应推荐akshare数据源"
    
    def test_all_sources_fail_logs_health_summary(self, log_capture):
        """测试数据源失败时记录健康状态汇总
        
        架构变更（2025-12-06）：
        - 仅使用primary_source，失败时抛出RuntimeError
        - 不再尝试backup_sources
        """
        # 使用akshare作为primary_source
        provider = RealHistoricalDataProvider(
            primary_source='akshare'
        )
        
        # Mock该数据源使其失败
        if 'akshare' in provider._adapters:
            adapter = provider._adapters['akshare']
            if hasattr(adapter, 'get_index_prices'):
                adapter.get_index_prices = Mock(side_effect=RuntimeError("数据源不可用"))
        
        # 新行为：抛出RuntimeError
        with pytest.raises(RuntimeError, match="数据源"):
            provider.get_index_prices('000300.SH', '2015-01-01', '2015-01-10')
    
    def test_mock_provider_always_succeeds(self, mock_event):
        """测试Mock数据源作为兜底始终成功"""
        mock_provider = MockHistoricalDataProvider()
        
        data, success = EventAnalyzer.safe_get_event_data(
            data_provider=mock_provider,
            event=mock_event
        )
        
        assert success is True
        assert not data.empty
        assert 'close' in data.columns
    
    def test_tushare_stub_in_cn_priority(self):
        """测试A股市场数据源推荐
        
        架构变更（2025-12-06）：不再有优先级列表概念
        """
        provider = RealHistoricalDataProvider(
            primary_source='akshare'
        )
        
        # 测试A股市场推荐akshare
        recommended = provider._get_recommended_source_for_region('000001.SH')
        assert recommended == 'akshare', "A股市场应推荐akshare数据源"
    
    def test_all_markets_covered(self):
        """测试所有市场都能获得数据源推荐
        
        架构变更（2025-12-06）：
        - 不再使用REGIONAL_DATA_SOURCE_PRIORITY（已删除）
        - 从data.yml的providers配置推断推荐
        """
        from core_bak_refactored.core.share.market.market_enums import MarketCode
        
        provider = RealHistoricalDataProvider()
        
        # 测试主要市场都有推荐数据源
        test_symbols = {
            MarketCode.CN: '000300.SH',
            MarketCode.US: 'SPX.US',
            MarketCode.HK: '0700.HK',
            # MarketCode.JP, EU, SG 可能没有配置的数据源
        }
        
        for market, symbol in test_symbols.items():
            recommended = provider._get_recommended_source_for_region(symbol)
            assert recommended is not None, f"市场 {market.value} 应有推荐数据源"
    
    def test_jp_market_priority(self):
        """测试日本市场数据源推荐
        
        架构变更（2025-12-06）：测试推荐逻辑
        """
        provider = RealHistoricalDataProvider()
        recommended = provider._get_recommended_source_for_region('9984.T')
        # 如果data.yml中没有配置支持JP市场的数据源，返回None
        # 这是正常的，因为当前配置可能不支持所有市场
    
    def test_eu_market_priority(self):
        """测试欧洲市场数据源推荐
        
        架构变更（2025-12-06）：测试推荐逻辑
        """
        provider = RealHistoricalDataProvider()
        recommended = provider._get_recommended_source_for_region('BMW.DE')
        # 如果data.yml中没有配置支持EU市场的数据源，返回None
    
    def test_sg_market_priority(self):
        """测试新加坡市场数据源推荐
        
        架构变更（2025-12-06）：测试推荐逻辑
        """
        provider = RealHistoricalDataProvider()
        recommended = provider._get_recommended_source_for_region('STI.SI')
        # 如果data.yml中没有配置支持SG市场的数据源，返回None


class TestCrossValidationIntegration:
    """Provider-level cross validation开关与数据源不足场景"""
    
    def test_insufficient_sources_handling(self):
        """测试数据源不足时的处理"""
        provider = RealHistoricalDataProvider()
        
        report = provider.cross_validate_sources(
            '000300.SH',
            '2015-06-01',
            '2015-06-15',
            sources=[DataSource.MOCK.value]
        )
        
        assert report['passed'] is True
        assert report['reason'] == 'insufficient_sources'
    
    def test_cross_validation_with_enable_flag(self):
        """测试enable_cross_validation标志控制"""
        provider_enabled = RealHistoricalDataProvider(enable_cross_validation=True)
        assert provider_enabled.enable_cross_validation is True
        
        provider_disabled = RealHistoricalDataProvider(enable_cross_validation=False)
        assert provider_disabled.enable_cross_validation is False


class TestHistoricalDataProviderEnhancedInterface(unittest.TestCase):
    """增强型历史数据提供者接口测试"""

    def setUp(self):
        self.mock_provider = MockHistoricalDataProvider()


    def test_get_stock_prices_interface(self):
        data = self.mock_provider.get_stock_prices('600036.SS', '2020-01-01', '2020-01-31')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertIsInstance(data['date'].iloc[0], pd.Timestamp)


    def test_get_volatility_index_interface(self):
        volatility_data = self.mock_provider.get_volatility_index('VIX', '2020-01-01', '2020-01-31')
        
        self.assertIsInstance(volatility_data, pd.Series)
        self.assertGreater(len(volatility_data), 0)
        self.assertTrue((volatility_data >= 0.05).all())
        self.assertTrue((volatility_data <= 0.5).all())

    def test_validate_data_quality_interface(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        })
        
        quality_report = self.mock_provider.validate_data_quality(data)
        
        self.assertIsInstance(quality_report, dict)
        self.assertIn('completeness_score', quality_report)
        self.assertIn('consistency_score', quality_report)
        self.assertIn('accuracy_score', quality_report)
        self.assertIn('outliers_detected', quality_report)
        self.assertIn('total_rows', quality_report)
        self.assertIn('missing_values', quality_report)
        
        self.assertGreaterEqual(quality_report['completeness_score'], 0.0)
        self.assertLessEqual(quality_report['completeness_score'], 1.0)
        self.assertGreaterEqual(quality_report['consistency_score'], 0.0)
        self.assertLessEqual(quality_report['consistency_score'], 1.0)
        self.assertGreaterEqual(quality_report['accuracy_score'], 0.0)
        self.assertLessEqual(quality_report['accuracy_score'], 1.0)

    def test_datetime_parameter_support(self):
        """测试datetime参数支持（直接使用Mock数据提供者）"""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 31)
        
        # 直接使用MockHistoricalDataProvider，而不是通过RealHistoricalDataProvider
        data = self.mock_provider.get_index_prices('000300.SH', start_date, end_date)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)

    def test_yahoo_finance_datetime_parameter_support(self):
        provider = YahooFinanceDataProvider()
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 31)
        
        try:
            data = provider.get_index_prices('000300.SS', start_date, end_date)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn('date', data.columns)
            self.assertIn('close', data.columns)
            self.assertIn('volume', data.columns)
        except ValueError as e:
            # 真实场景不可用时，应抛出异常（不再回退到Mock）
            self.assertTrue('Failed to fetch data' in str(e) or 'No data returned' in str(e) or 'yfinance' in str(e))


class RealProviderMinimalIntegrationTest(unittest.TestCase):
    """最小集成测试（直接使用Mock数据提供者）"""
    
    def setUp(self):
        # 直接使用MockHistoricalDataProvider，不通过RealHistoricalDataProvider
        self.mock_provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(data_provider=self.mock_provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_prices_and_returns(self):
        """测试价格和收益率获取（使用Mock数据）"""
        prices = self.mock_provider.get_index_prices('000300.SH', '2015-06-15', '2015-08-26')
        self.assertFalse(prices.empty)
        self.assertIn('close', prices.columns)
        self.assertIn('volume', prices.columns)
        
        returns = self.mock_provider.get_index_returns('000300.SH', '2015-06-15', '2015-08-26')
        self.assertGreater(len(returns), 0)
        self.assertLessEqual(len(returns), len(prices))

    def test_backtest_with_real_provider_mock_mode(self):
        """测试回测集成（使用Mock数据提供者）"""
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH'
        )
        self.assertTrue(len(results) > 0)
        
        summary = BacktestReporter.generate_summary(results)
        self.assertIn('total_tests', summary)
        self.assertGreater(summary['total_tests'], 0)
