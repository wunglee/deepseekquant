import unittest
from core.risk.risk_processor import RiskProcessor

class TestRiskLimits(unittest.TestCase):
    def test_position_limit_exceeded(self):
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        signal = {'price': 100.0, 'quantity': 2000}
        limits = {'max_position_value': 100000.0}
        result = rp.process(signal=signal, limits=limits)
        self.assertEqual(result['status'], 'success')
        assessment = result['assessment']
        self.assertFalse(assessment['approved'])
        self.assertIn('POSITION_LIMIT_EXCEEDED', assessment['warnings'])
        rp.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
