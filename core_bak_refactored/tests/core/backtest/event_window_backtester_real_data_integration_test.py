import unittest

from core_bak_refactored.core.risk.backtest_framework import create_data_provider
from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester, BacktestReporter
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.risk.stress_testing import StressTester


class EventWindowBacktesterRealDataIntegrationTest(unittest.TestCase):
    """Phase 3B: 使用真实数据(优先)的事件窗口回测集成测试。

    目标：
    - 通过 create_data_provider('auto') 优先使用 Yahoo Finance 数据，失败自动回退 Mock。
    - 对 3 个核心事件执行端到端回测，计算预测误差。
    - 验证最大预测误差不超过 25%（Phase 3B 验收标准）。
    """

    def setUp(self):
        # 使用 auto 模式的数据提供者：优先真实数据，失败回退 Mock
        provider = create_data_provider('auto')
        self.backtester = EventWindowBacktester(data_provider=provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        # 使用默认 StressTester，让其场景参数参与预测损失计算
        self.stress_tester = StressTester(config={})

    def test_real_data_backtest_three_core_events_error_within_25_percent(self):
        # 运行回测
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=self.stress_tester,
            benchmark_index='000300.SH',
        )

        # 仅保留 3 个核心事件的结果
        core_event_ids = {
            '2015_china_market_crash',
            'covid_19_pandemic',
            '2008_financial_crisis',
        }
        core_results = [r for r in results if r.event_id in core_event_ids]

        # 基本断言：3 个核心事件都有结果
        self.assertEqual(len(core_results), 3, msg="三大核心事件回测结果数量不正确")

        # 生成摘要
        summary = BacktestReporter.generate_summary(core_results)
        self.assertEqual(summary['total_tests'], 3)

        max_error = summary['max_error']

        # Phase 3B 标准：误差阈值放宽到 25%
        self.assertLessEqual(
            max_error,
            0.25,
            msg=f"三大核心事件的最大预测误差超过 25% 阈值: {max_error:.2%}",
        )


if __name__ == '__main__':
    unittest.main()
