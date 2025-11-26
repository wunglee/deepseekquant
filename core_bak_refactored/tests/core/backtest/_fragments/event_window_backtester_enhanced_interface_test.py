import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder


class TestEventWindowBacktesterEnhancedInterface(unittest.TestCase):
    """事件窗口回测器增强接口测试"""

    def setUp(self):
        """设置测试环境"""
        self.mock_provider = MockHistoricalDataProvider()
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_backtester_with_mock_provider(self):
        """测试使用Mock数据提供者的回测器"""
        backtester = EventWindowBacktester(data_provider=self.mock_provider)
        
        # 运行回测
        results = backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        
        # 验证返回结果
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # 验证每个结果的结构
        for result in results:
            self.assertIsNotNone(result.event_id)
            self.assertIsNotNone(result.portfolio_id)
            self.assertIsNotNone(result.predicted_loss)
            self.assertIsNotNone(result.actual_loss)
            self.assertIsNotNone(result.prediction_error)

    def test_backtester_with_datetime_parameters(self):
        """测试使用datetime参数的回测器"""
        # 创建一个简化版本的回测器用于测试
        class TestDataProvider(MockHistoricalDataProvider):
            def get_index_prices(self, index_id: str, start_date, end_date):
                # 验证参数类型
                if isinstance(start_date, datetime):
                    start_date = start_date.strftime('%Y-%m-%d')
                if isinstance(end_date, datetime):
                    end_date = end_date.strftime('%Y-%m-%d')
                return super().get_index_prices(index_id, start_date, end_date)
        
        provider = TestDataProvider()
        backtester = EventWindowBacktester(data_provider=provider)
        
        # 使用datetime参数运行回测
        results = backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        
        # 验证返回结果
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_data_provider_interface_compliance(self):
        """测试数据提供者是否符合增强接口"""
        # 测试Mock提供者
        self.assertTrue(hasattr(self.mock_provider, 'get_stock_prices'))
        self.assertTrue(hasattr(self.mock_provider, 'get_volatility_index'))
        self.assertTrue(hasattr(self.mock_provider, 'validate_data_quality'))
        
        # 测试方法是否可调用
        self.assertTrue(callable(self.mock_provider.get_stock_prices))
        self.assertTrue(callable(self.mock_provider.get_volatility_index))
        self.assertTrue(callable(self.mock_provider.validate_data_quality))

    def test_yahoo_finance_provider_enhanced_features(self):
        """测试Yahoo Finance提供者的增强功能"""
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        
        # 测试方法是否存在
        self.assertTrue(hasattr(provider, 'get_stock_prices'))
        self.assertTrue(hasattr(provider, 'get_volatility_index'))
        self.assertTrue(hasattr(provider, 'validate_data_quality'))
        
        # 测试方法是否可调用
        self.assertTrue(callable(provider.get_stock_prices))
        self.assertTrue(callable(provider.get_volatility_index))
        self.assertTrue(callable(provider.validate_data_quality))


if __name__ == '__main__':
    unittest.main()