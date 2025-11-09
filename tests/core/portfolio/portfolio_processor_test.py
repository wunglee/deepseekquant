import unittest
from core.portfolio.portfolio_processor import PortfolioProcessor
from common import AllocationMethod, PortfolioObjective

class TestPortfolioProcessor(unittest.TestCase):
    def test_initialization(self):
        p = PortfolioProcessor(processor_name='Portfolio')
        self.assertTrue(p.initialize())

    def test_equal_weight_allocation(self):
        p = PortfolioProcessor(processor_name='Portfolio')
        p.initialize()
        positions = [
            {'symbol': 'AAPL', 'quantity': 10, 'price': 150},
            {'symbol': 'GOOG', 'quantity': 5, 'price': 2800}
        ]
        result = p.process(positions=positions, method=AllocationMethod.EQUAL_WEIGHT)
        self.assertEqual(result['status'], 'success')
        self.assertIn('weights', result)
        self.assertAlmostEqual(result['weights']['AAPL'], 0.5, places=2)
        p.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
