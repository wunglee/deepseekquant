import unittest
from core.data.data_fetcher import DataFetcher

class TestDataHistory(unittest.TestCase):
    def test_history_and_volatility(self):
        f = DataFetcher()
        hist = f.get_history('AAPL', lookback=30)
        self.assertEqual(len(hist), 30)
        closes = [row['close'] for row in hist]
        vol = f.compute_volatility(closes)
        self.assertGreaterEqual(vol, 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
