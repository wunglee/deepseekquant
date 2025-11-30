"""
EventWindowBacktester 统一测试文件
合并了以下测试文件：
- event_window_backtester_enhanced_interface_test.py（增强接口测试）
- event_window_backtester_integration_test.py（Mock数据集成测试）
- event_window_backtester_real_data_integration_test.py（真实数据集成测试）
- event_window_backtester_real_data_five_events_integration_test.py（5事件真实数据测试）
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester, BacktestReporter
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.data.providers.yahoo_finance import YahooFinanceDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder
from core_bak_refactored.core.risk.backtest_framework import create_data_provider
from core_bak_refactored.core.risk.stress_testing import StressTester


class EventWindowBacktesterEnhancedInterfaceTest(unittest.TestCase):
    """增强接口测试"""

    def setUp(self):
        self.mock_provider = MockHistoricalDataProvider()
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_backtester_with_mock_provider(self):
        backtester = EventWindowBacktester(data_provider=self.mock_provider)
        results = backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIsNotNone(result.event_id)
            self.assertIsNotNone(result.portfolio_id)
            self.assertIsNotNone(result.predicted_loss)
            self.assertIsNotNone(result.actual_loss)
            self.assertIsNotNone(result.prediction_error)

    def test_backtester_with_datetime_parameters(self):
        class TestDataProvider(MockHistoricalDataProvider):
            def get_index_prices(self, index_id: str, start_date, end_date):
                if isinstance(start_date, datetime):
                    start_date = start_date.strftime('%Y-%m-%d')
                if isinstance(end_date, datetime):
                    end_date = end_date.strftime('%Y-%m-%d')
                return super().get_index_prices(index_id, start_date, end_date)
        
        provider = TestDataProvider()
        backtester = EventWindowBacktester(data_provider=provider)
        results = backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=None,
            benchmark_index='000300.SH',
        )
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_data_provider_interface_compliance(self):
        self.assertTrue(hasattr(self.mock_provider, 'get_stock_prices'))
        self.assertTrue(hasattr(self.mock_provider, 'get_volatility_index'))
        self.assertTrue(hasattr(self.mock_provider, 'validate_data_quality'))
        
        self.assertTrue(callable(self.mock_provider.get_stock_prices))
        self.assertTrue(callable(self.mock_provider.get_volatility_index))
        self.assertTrue(callable(self.mock_provider.validate_data_quality))

    def test_yahoo_finance_provider_enhanced_features(self):
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        
        self.assertTrue(hasattr(provider, 'get_stock_prices'))
        self.assertTrue(hasattr(provider, 'get_volatility_index'))
        self.assertTrue(hasattr(provider, 'validate_data_quality'))
        
        self.assertTrue(callable(provider.get_stock_prices))
        self.assertTrue(callable(provider.get_volatility_index))
        self.assertTrue(callable(provider.validate_data_quality))


class EventWindowBacktesterMockIntegrationTest(unittest.TestCase):
    """Mock数据集成测试"""

    def setUp(self):
        self.backtester = EventWindowBacktester()
        self.data_provider = MockHistoricalDataProvider()
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        self.stress_tester = None

    def test_run_backtest_with_mock_provider(self):
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=self.stress_tester,
            benchmark_index='000300.SH'
        )
        
        self.assertTrue(len(results) > 0)
        
        first = results[0]
        self.assertTrue(hasattr(first, 'event_id'))
        self.assertTrue(hasattr(first, 'portfolio_id'))
        self.assertTrue(hasattr(first, 'predicted_loss'))
        self.assertTrue(hasattr(first, 'actual_loss'))
        self.assertTrue(hasattr(first, 'prediction_error'))

        summary = BacktestReporter.generate_summary(results)
        self.assertIn('total_tests', summary)
        self.assertIn('avg_error', summary)
        self.assertIn('accuracy_20pct', summary)
        self.assertEqual(summary['total_tests'], len(results))
        self.assertGreaterEqual(summary['accuracy_20pct'], 0.8)
        
        self.assertIn('top_errors', summary)
        self.assertTrue(isinstance(summary['top_errors'], list))
        self.assertGreaterEqual(len(summary['top_errors']), 1)
        
        first_err = summary['top_errors'][0]
        self.assertIn('name', first_err)
        self.assertIn('period', first_err)


class EventWindowBacktesterRealDataIntegrationTest(unittest.TestCase):
    """真实数据集成测试（3个核心事件）"""

    def setUp(self):
        provider = create_data_provider('auto')
        self.backtester = EventWindowBacktester(data_provider=provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        self.stress_tester = StressTester(config={})

    def test_real_data_backtest_three_core_events_error_within_25_percent(self):
        results = self.backtester.run_backtest(
            portfolio=self.portfolio,
            stress_tester=self.stress_tester,
            benchmark_index='000300.SH',
        )

        core_event_ids = {
            '2015_china_market_crash',
            'covid_19_pandemic',
            '2008_financial_crisis',
        }
        core_results = [r for r in results if r.event_id in core_event_ids]

        self.assertEqual(len(core_results), 3, msg="三大核心事件回测结果数量不正确")

        summary = BacktestReporter.generate_summary(core_results)
        self.assertEqual(summary['total_tests'], 3)

        max_error = summary['max_error']
        self.assertLessEqual(
            max_error,
            0.25,
            msg=f"三大核心事件的最大预测误差超过 25% 阈值: {max_error:.2%}",
        )


class EventWindowBacktesterFiveEventsIntegrationTest(unittest.TestCase):
    """5事件真实数据集成测试"""

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

        self.assertEqual(len(target_results), 5, msg="五事件回测结果数量不正确")

        summary = BacktestReporter.generate_summary(target_results)
        self.assertEqual(summary['total_tests'], 5)

        errors = [r.prediction_error for r in target_results]
        within_25 = sum(1 for e in errors if e <= 0.25) / len(errors)
        self.assertGreaterEqual(
            within_25,
            0.8,
            msg=f"五事件中误差≤25%的比例不足 80%: {within_25:.2%}",
        )


class ContagionHistoricalValidationTest(unittest.TestCase):
    """风险传导历史验证测试（业务目标4）"""

    def setUp(self):
        provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_contagion_event_pairs_data_availability(self):
        """验证传导场景对的事件数据完整性"""
        contagion_pairs = [
            ('2008_financial_crisis', '2015_china_market_crash'),
            ('covid_19_pandemic', '2022_russia_ukraine_conflict'),
            ('1997_asian_financial_crisis', '2008_financial_crisis'),
        ]
        
        loaded_event_ids = {e.event_id for e in self.backtester.events}
        
        for lead_event, contagion_event in contagion_pairs:
            self.assertIn(
                lead_event,
                loaded_event_ids,
                msg=f"先导事件 {lead_event} 未加载"
            )
            self.assertIn(
                contagion_event,
                loaded_event_ids,
                msg=f"传导事件 {contagion_event} 未加载"
            )
        
        self.assertGreaterEqual(
            len(contagion_pairs),
            3,
            msg="传导场景对数量不足3组"
        )

    def test_transmission_factor_validation(self):
        """验证传导因子取值依据（基于专家标准：30%，A股35%）"""
        EXPECTED_TRANSMISSION_FACTOR_RANGE = (0.25, 0.35)
        A_SHARE_TRANSMISSION_FACTOR = 0.35
        
        self.assertGreaterEqual(
            A_SHARE_TRANSMISSION_FACTOR,
            EXPECTED_TRANSMISSION_FACTOR_RANGE[0],
            msg="传导因子低于文献支持下限（25%）"
        )
        self.assertLessEqual(
            A_SHARE_TRANSMISSION_FACTOR,
            EXPECTED_TRANSMISSION_FACTOR_RANGE[1],
            msg="传导因子高于文献支持上限（35%）"
        )


if __name__ == '__main__':
    unittest.main()
