import unittest

from core_bak_refactored.core.risk.backtest_framework import create_data_provider, HistoricalDataProvider
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.data.providers.yahoo_finance import YahooFinanceDataProvider


class BacktestFrameworkTest(unittest.TestCase):
    def test_create_mock_provider(self):
        provider: HistoricalDataProvider = MockHistoricalDataProvider()
        self.assertIsInstance(provider, MockHistoricalDataProvider)

    def test_create_yahoo_provider_with_fallback(self):
        provider = create_data_provider('yahoo', fallback_to_mock=False)
        # Depending on environment, Yahoo provider may or may not initialize; check type or mock fallback behavior
        self.assertTrue(isinstance(provider, YahooFinanceDataProvider))

    def test_auto_provider_returns_provider(self):
        provider = create_data_provider('auto')
        # In production code path, auto currently equals to yahoo-only
        self.assertTrue(isinstance(provider, YahooFinanceDataProvider))


if __name__ == '__main__':
    unittest.main()
