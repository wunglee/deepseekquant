import unittest
from core.risk.risk_processor import RiskProcessor

class TestRiskVolConcentration(unittest.TestCase):
    def test_volatility_threshold(self):
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        signal = {'price': 100.0, 'quantity': 10}
        prices = [i for i in range(1, 100)]
        limits = {'volatility_threshold': 0.05}
        res = rp.process(signal=signal, limits=limits, prices=prices)
        self.assertEqual(res['status'], 'success')
        rp.cleanup()

    def test_concentration_threshold(self):
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        res = rp.process(signal={'price': 100.0, 'quantity': 10}, limits={'max_position_value': 100000}, target_weight=0.8)
        self.assertEqual(res['status'], 'success')
        self.assertIn('assessment', res)
        rp.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
