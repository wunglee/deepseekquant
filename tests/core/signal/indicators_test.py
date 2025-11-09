import unittest
from core.signal.indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        self.prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
                       110.0, 112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 118.0, 117.0, 119.0, 120.0,
                       121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0]
    
    def test_sma(self):
        sma = TechnicalIndicators.sma(self.prices, 5)
        self.assertIsNotNone(sma)
        if sma is not None:
            self.assertGreater(sma, 0.0)
    
    def test_ema(self):
        ema = TechnicalIndicators.ema(self.prices, 5)
        self.assertIsNotNone(ema)
        if ema is not None:
            self.assertGreater(ema, 0.0)
    
    def test_rsi(self):
        rsi = TechnicalIndicators.rsi(self.prices, 14)
        self.assertIsNotNone(rsi)
        if rsi is not None:
            self.assertGreaterEqual(rsi, 0.0)
            self.assertLessEqual(rsi, 100.0)
    
    def test_macd(self):
        macd = TechnicalIndicators.macd(self.prices)
        self.assertIsNotNone(macd)
        if macd is not None:
            self.assertIn('macd', macd)
            self.assertIn('signal', macd)
            self.assertIn('histogram', macd)
    
    def test_bollinger_bands(self):
        bb = TechnicalIndicators.bollinger_bands(self.prices, 20)
        self.assertIsNotNone(bb)
        if bb is not None:
            self.assertIn('upper', bb)
            self.assertIn('middle', bb)
            self.assertIn('lower', bb)
            self.assertGreater(bb['upper'], bb['middle'])
            self.assertGreater(bb['middle'], bb['lower'])
    
    def test_momentum(self):
        momentum = TechnicalIndicators.momentum(self.prices, 10)
        self.assertIsNotNone(momentum)
    
    def test_rate_of_change(self):
        roc = TechnicalIndicators.rate_of_change(self.prices, 10)
        self.assertIsNotNone(roc)

if __name__ == '__main__':
    unittest.main(verbosity=2)
