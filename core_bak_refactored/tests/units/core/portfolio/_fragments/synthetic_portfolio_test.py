import unittest

from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio import (
    SyntheticPortfolio, SyntheticPortfolioBuilder
)


class SyntheticPortfolioTest(unittest.TestCase):
    def test_synthetic_portfolio_creation(self):
        portfolio = SyntheticPortfolio(
            portfolio_id='TEST_PORT',
            name='Test Portfolio',
            composition={'000300.SH': 0.6, 'HSI': 0.4},
            total_value=1000000.0,
            metadata={'type': 'test'}
        )
        
        self.assertEqual(portfolio.portfolio_id, 'TEST_PORT')
        self.assertEqual(portfolio.name, 'Test Portfolio')
        self.assertEqual(len(portfolio.composition), 2)
        self.assertEqual(portfolio.total_value, 1000000.0)
        
    def test_csi300_equal_weight_builder(self):
        portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        
        self.assertEqual(portfolio.portfolio_id, 'CSI300_EQ')
        self.assertIn('000300.SH', portfolio.composition)
        self.assertEqual(portfolio.composition['000300.SH'], 1.0)
        self.assertEqual(portfolio.metadata['type'], 'index_replication')
        
    def test_sector_rotation_builder(self):
        portfolio = SyntheticPortfolioBuilder.build_sector_rotation()
        
        self.assertEqual(portfolio.portfolio_id, 'SECTOR_ROT')
        self.assertEqual(len(portfolio.composition), 4)
        
        # 验证权重总和为1
        total_weight = sum(portfolio.composition.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
    def test_ah_hybrid_builder(self):
        portfolio = SyntheticPortfolioBuilder.build_ah_hybrid()
        
        self.assertEqual(portfolio.portfolio_id, 'AH_HYBRID')
        self.assertIn('000300.SH', portfolio.composition)
        self.assertIn('HSI', portfolio.composition)
        self.assertEqual(portfolio.composition['000300.SH'], 0.7)
        self.assertEqual(portfolio.composition['HSI'], 0.3)


if __name__ == '__main__':
    unittest.main()
