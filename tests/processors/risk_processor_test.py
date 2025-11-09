import unittest
from core.risk.risk_processor import RiskProcessor

class TestRiskProcessor(unittest.TestCase):
    def test_basic_risk(self):
        p = RiskProcessor()
        p.initialize()
        r = p.process(metric='var')
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['metric'], 'var')
        p.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
