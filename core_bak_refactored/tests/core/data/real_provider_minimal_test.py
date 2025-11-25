import unittest

from core_bak_refactored.core.data._fragments.historical_data_provider import RealHistoricalDataProvider
from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester, BacktestReporter
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder


class RealProviderMinimalIntegrationTest(unittest.TestCase):
    def setUp(self):
        # 使用最小真实提供者（mock模式），保证端到端运行
        self.real = RealHistoricalDataProvider(data_source='mock')
        self.backtester = EventWindowBacktester(data_provider=self.real)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_prices_and_returns(self):
        # 使用事件窗口日期生成数据
        prices = self.real.get_index_prices('000300.SH', '2015-06-15', '2015-08-26')
        self.assertFalse(prices.empty)
        self.assertIn('close', prices.columns)
        self.assertIn('volume', prices.columns)
        returns = self.real.get_index_returns('000300.SH', '2015-06-15', '2015-08-26')
        self.assertEqual(len(returns), len(prices))

    def test_backtest_with_real_provider_mock_mode(self):
        # 运行端到端回测
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,  # 简化：当前预测损失使用事件参数decline
            benchmark_index='000300.SH'
        )
        self.assertTrue(len(results) > 0)
        summary = BacktestReporter.generate_summary(results)
        self.assertIn('total_tests', summary)
        self.assertGreater(summary['total_tests'], 0)


if __name__ == '__main__':
    unittest.main()
