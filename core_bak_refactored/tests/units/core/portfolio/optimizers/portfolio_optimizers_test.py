import unittest
import numpy as np

from core_bak_refactored.core.portfolio.optimizers import PortfolioOptimizers


class PortfolioOptimizersTest(unittest.TestCase):
    def setUp(self):
        self.returns = np.array([0.1, 0.12, 0.15])
        self.cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.05, 0.015],
            [0.02, 0.015, 0.06]
        ])
        
    def test_mean_variance_optimization(self):
        weights = PortfolioOptimizers.mean_variance_optimization(
            self.returns, self.cov_matrix, risk_aversion=1.0
        )
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue(all(w >= 0 for w in weights))
        
    def test_minimum_variance(self):
        weights = PortfolioOptimizers.minimum_variance(self.cov_matrix)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        
    def test_risk_parity(self):
        weights = PortfolioOptimizers.risk_parity(self.cov_matrix)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
