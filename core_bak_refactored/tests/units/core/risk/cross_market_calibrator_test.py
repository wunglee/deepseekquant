import unittest

from core_bak_refactored.core.risk.cross_market_calibrator import CrossMarketCalibrator


class CrossMarketCalibratorTest(unittest.TestCase):
    def test_calibrator_instantiation(self):
        calibrator = CrossMarketCalibrator()
        self.assertIsNotNone(calibrator)


if __name__ == '__main__':
    unittest.main()
