import unittest
from datetime import datetime

from core_bak_refactored.core.risk.stress_test_validator import (
    StressTestValidator, HistoricalEvent, ValidationResult
)


class _MockDataSource:
    def get_event_returns(self, event: HistoricalEvent, asset_id: str) -> float:
        # Return expected_decline as actual loss to produce small error when predicted equals scenario param
        return event.expected_decline


class _MockPortfolioBuilder:
    def build_test_portfolio(self, portfolio_type: str):
        return {'000300.SH': 1.0}


class StressTestValidatorTest(unittest.TestCase):
    def setUp(self):
        self.validator = StressTestValidator(
            data_source=_MockDataSource(),
            portfolio_builder=_MockPortfolioBuilder()
        )

    def test_validate_single_scenario(self):
        # Use one known scenario id from _load_validation_events
        result = self.validator.validate_scenario(
            scenario_id='2015_china_market_crash',
            stress_tester=None,
            benchmark_asset='000300.SH'
        )
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.event_id, '2015_china_market_crash')
        self.assertIsInstance(result.validation_date, datetime)
        # With predicted from scenario_params and actual=expected_decline, error should be reasonable
        self.assertGreaterEqual(result.prediction_error, 0.0)

    def test_validate_all_and_report(self):
        results = self.validator.validate_all_scenarios(stress_tester=None)
        self.assertTrue(len(results) > 0)
        report = self.validator.generate_validation_report(results)
        self.assertIn('total_validations', report)
        self.assertIn('avg_error', report)
        self.assertEqual(report['total_validations'], len(results))


if __name__ == '__main__':
    unittest.main()
