import unittest

from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import (
    SyntheticPortfolioBuilder, SyntheticPortfolio
)


class SyntheticPortfolioBuilderTest(unittest.TestCase):
    def test_build_csi300_equal_weight(self):
        p = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        self.assertIsInstance(p, SyntheticPortfolio)
        self.assertAlmostEqual(sum(p.composition.values()), 1.0, places=6)
        self.assertIn('000300.SH', p.composition)

    def test_build_sector_rotation(self):
        p = SyntheticPortfolioBuilder.build_sector_rotation()
        self.assertIsInstance(p, SyntheticPortfolio)
        self.assertAlmostEqual(sum(p.composition.values()), 1.0, places=6)
        self.assertIn('finance_index', p.composition)

    def test_build_ah_hybrid(self):
        p = SyntheticPortfolioBuilder.build_ah_hybrid()
        self.assertIsInstance(p, SyntheticPortfolio)
        self.assertAlmostEqual(sum(p.composition.values()), 1.0, places=6)
        self.assertIn('HSI', p.composition)

    def test_build_by_type(self):
        p1 = SyntheticPortfolioBuilder.build_by_type('csi300')
        self.assertIsInstance(p1, SyntheticPortfolio)
        with self.assertRaises(ValueError):
            SyntheticPortfolioBuilder.build_by_type('unknown_type')


if __name__ == '__main__':
    unittest.main()
