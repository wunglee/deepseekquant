import unittest

from core_bak_refactored.core.risk.market_detectors import MarketMechanismDetector, ChinaMarketDetector


class MarketDetectorsTest(unittest.TestCase):
    def test_detector_instantiation(self):
        detector = ChinaMarketDetector(config={})
        self.assertIsNotNone(detector)


if __name__ == '__main__':
    unittest.main()
