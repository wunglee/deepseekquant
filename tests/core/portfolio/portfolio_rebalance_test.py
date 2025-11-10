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

    def test_weight_bounds_and_min_trade(self):
        p = PortfolioProcessor(processor_name='Portfolio')
        p.initialize()
        positions = [
            {'symbol': 'AAPL', 'quantity': 10, 'price': 150},
            {'symbol': 'GOOG', 'quantity': 5, 'price': 2800}
        ]
        current = {'AAPL': 9.999, 'GOOG': 5.001}
        result = p.process(positions=positions, method=AllocationMethod.EQUAL_WEIGHT,
                           current_positions=current, min_weight=0.4, max_weight=0.6, min_trade_qty=0.1,
                           commission=0.001, slippage=0.0005)
        self.assertEqual(result['status'], 'success')
        self.assertIn('weights', result)
        # 检查权重在边界内
        for w in result['weights'].values():
            self.assertGreaterEqual(w, 0.4)
            self.assertLessEqual(w, 0.6)
        # 检查最小交易量规则生效（可能存在为0的quantity_change）
        self.assertIn('rebalance', result)
        self.assertIn('estimated_costs', result)
        p.cleanup()
