import unittest
from core.data.data_fetcher import DataFetcher, DataValidator, MarketData

class TestDataFetcher(unittest.TestCase):
    def test_validator(self):
        validator = DataValidator()
        md = MarketData(symbol='TEST', price=100.0, timestamp='2024-01-01')
        result = validator.validate(md)
        self.assertTrue(result['is_valid'])

    def test_get_market_data(self):
        fetcher = DataFetcher()
        result = fetcher.get_market_data('AAPL', use_cache=False)
        self.assertEqual(result['status'], 'success')
        self.assertIn('data', result)

if __name__ == '__main__':
    unittest.main(verbosity=2)
