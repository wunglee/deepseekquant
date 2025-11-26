import unittest

from core_bak_refactored.core.backtest._fragments.event_window_backtester import (
    EventWindowBacktester,
)
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.backtest._fragments.stress_test_result import (
    StressTestResult,
    from_backtest_result,
)


class StressTestResultSchemaTest(unittest.TestCase):
    """验证 StressTestResult 数据结构与转换助手。"""

    def setUp(self):
        provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_convert_backtest_results_to_stress_test_results(self):
        # 运行回测获取5个事件结果
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        self.assertGreaterEqual(len(results), 5)

        # 转换为标准化结果
        std_results = [from_backtest_result(r) for r in results]
        self.assertEqual(len(std_results), len(results))

        # 验证关键字段存在（20字段完整性要求对应的必需键均存在于数据类中）
        required_fields = {
            'report_id',
            'portfolio_id',
            'scenario_id',
            'var_normal',
            'var_stressed',
            'stress_loss_amount',
            'stress_loss_percentage',
            'recovery_period',
            'risk_decomposition',
            'triggered_actions',
            'recommended_actions',
            'compliance_status',
            'metadata',
        }
        for s in std_results:
            d = s.to_dict()
            # 至少包含上述字段（其余元数据在 metadata 中统一存放）
            self.assertTrue(required_fields.issubset(d.keys()))
            # 检查基本类型
            self.assertIsInstance(s.report_id, str)
            self.assertIsInstance(s.portfolio_id, str)
            self.assertIsInstance(s.scenario_id, str)
            self.assertIsInstance(s.metadata, dict)
            # 映射的百分比为浮点或None
            self.assertTrue(
                isinstance(s.stress_loss_percentage, (float, type(None)))
            )


if __name__ == '__main__':
    unittest.main()
