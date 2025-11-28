import unittest

from core_bak_refactored.core.risk.backtest_framework import create_data_provider
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider


class BacktestFrameworkTest(unittest.TestCase):
    def test_create_mock_provider(self):
        provider = create_data_provider('mock')
        self.assertIsInstance(provider, MockHistoricalDataProvider)

    def test_create_yahoo_provider_with_fallback(self):
        provider = create_data_provider('yahoo', fallback_to_mock=True)
        # Depending on environment, Yahoo provider may or may not initialize; check type or mock fallback behavior
        self.assertTrue(isinstance(provider, YahooFinanceDataProvider))

    def test_auto_provider_returns_provider(self):
        provider = create_data_provider('auto')
        # Should return either YahooFinanceDataProvider or MockHistoricalDataProvider
        self.assertTrue(isinstance(provider, (YahooFinanceDataProvider, MockHistoricalDataProvider)))


if __name__ == '__main__':
    unittest.main()
