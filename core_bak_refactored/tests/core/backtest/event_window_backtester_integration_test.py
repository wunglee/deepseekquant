import unittest

from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester, BacktestReporter
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.risk.stress_testing import StressTester


class EventWindowBacktesterIntegrationTest(unittest.TestCase):
    def setUp(self):
        # 使用默认Mock数据源的回测器（构造函数支持自动回退）
        self.backtester = EventWindowBacktester()
        self.data_provider = MockHistoricalDataProvider()
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        # Phase 3A：避免引入未确认业务参数，使用事件参数/默认值
        self.stress_tester = None

    def test_run_backtest_with_mock_provider(self):
        # 运行回测
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=self.stress_tester,
            benchmark_index='000300.SH'
        )
        # 断言有结果
        self.assertTrue(len(results) > 0)
        # 断言结果字段完整
        first = results[0]
        self.assertTrue(hasattr(first, 'event_id'))
        self.assertTrue(hasattr(first, 'portfolio_id'))
        self.assertTrue(hasattr(first, 'predicted_loss'))
        self.assertTrue(hasattr(first, 'actual_loss'))
        self.assertTrue(hasattr(first, 'prediction_error'))

        # 生成摘要并断言主要统计项存在
        summary = BacktestReporter.generate_summary(results)
        self.assertIn('total_tests', summary)
        self.assertIn('avg_error', summary)
        self.assertIn('accuracy_20pct', summary)
        self.assertEqual(summary['total_tests'], len(results))
        # 新增质量断言：≤20%误差比例至少达到80%（扩展事件后放宽）
        self.assertGreaterEqual(summary['accuracy_20pct'], 0.8)
        # 新增字段断言：存在误差Top5
        self.assertIn('top_errors', summary)
        self.assertTrue(isinstance(summary['top_errors'], list))
        self.assertGreaterEqual(len(summary['top_errors']), 1)
        # 检查Top5元素包含name与period
        first_err = summary['top_errors'][0]
        self.assertIn('name', first_err)
        self.assertIn('period', first_err)


if __name__ == '__main__':
    unittest.main()
