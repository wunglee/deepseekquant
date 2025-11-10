import unittest
from core.portfolio.portfolio_processor import PortfolioProcessor
from common import AllocationMethod

class TestPortfolioRebalance(unittest.TestCase):
    def test_rebalance_and_turnover(self):
        p = PortfolioProcessor(processor_name='Portfolio')
        p.initialize()
        positions = [
            {'symbol': 'AAPL', 'quantity': 10, 'price': 150},
            {'symbol': 'GOOG', 'quantity': 5, 'price': 2800}
        ]
        current = {'AAPL': 8, 'GOOG': 6}
        result = p.process(positions=positions, method=AllocationMethod.EQUAL_WEIGHT, current_positions=current)
        self.assertEqual(result['status'], 'success')
        self.assertIn('rebalance', result)
        self.assertIn('turnover_rate', result)
        self.assertGreaterEqual(result['turnover_rate'], 0.0)
        p.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
