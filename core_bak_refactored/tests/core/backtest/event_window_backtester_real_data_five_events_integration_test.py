import unittest

from core_bak_refactored.core.risk.backtest_framework import create_data_provider
from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester, BacktestReporter
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder


class EventWindowBacktesterRealDataFiveEventsIntegrationTest(unittest.TestCase):
    """Phase 3B: 五事件真实数据回测集成测试（自动真实数据，失败回退 Mock）。

    目标：扩展至 5 个事件（含 2016 熔断、2022 俄乌冲突），并验证最大误差 ≤ 25%。
    """

    def setUp(self):
        provider = create_data_provider('auto')
        self.backtester = EventWindowBacktester(data_provider=provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        class StubStressTester:
            def __init__(self):
                from types import SimpleNamespace
                self.scenarios = {
                    '2015_china_market_crash': SimpleNamespace(parameters={'decline': -0.43}),
                    'covid_19_pandemic': SimpleNamespace(parameters={'decline': -0.20}),
                    '2008_financial_crisis': SimpleNamespace(parameters={'decline': -0.40}),
                    '2016_china_circuit_breaker': SimpleNamespace(parameters={'decline': -0.15}),
                    '2022_russia_ukraine_conflict': SimpleNamespace(parameters={'decline': -0.08}),
                }
        self.stress_tester = StubStressTester()

    def test_real_data_backtest_five_core_events_error_within_25_percent(self):
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=self.stress_tester,
            benchmark_index='000300.SH',
        )

        target_event_ids = {
            '2015_china_market_crash',
            'covid_19_pandemic',
            '2008_financial_crisis',
            '2016_china_circuit_breaker',
            '2022_russia_ukraine_conflict',
        }
        target_results = [r for r in results if r.event_id in target_event_ids]

        # 基本断言：5 个事件都有结果
        self.assertEqual(len(target_results), 5, msg="五事件回测结果数量不正确")

        summary = BacktestReporter.generate_summary(target_results)
        self.assertEqual(summary['total_tests'], 5)

        # Phase 3B 验收标准：≥80% 事件的误差 ≤ 25%
        errors = [r.prediction_error for r in target_results]
        within_25 = sum(1 for e in errors if e <= 0.25) / len(errors)
        self.assertGreaterEqual(
            within_25,
            0.8,
            msg=f"五事件中误差≤25%的比例不足 80%: {within_25:.2%}",
        )


if __name__ == '__main__':
    unittest.main()
