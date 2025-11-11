import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd

from core.risk.position_risk import PositionRiskAnalyzer


class DummyPosition:
    def __init__(self, current_value: float, weight: float):
        self.current_value = current_value
        self.weight = weight


class PositionRiskAnalyzerTest(unittest.TestCase):
    """测试持仓风险分析器"""

    def setUp(self):
        self.config = {}
        self.analyzer = PositionRiskAnalyzer(self.config)

    def test_analyze_position_with_valid_data(self):
        """分析持仓：有效数据"""
        symbol = 'AAPL'
        position = DummyPosition(current_value=10000, weight=0.1)
        market_data = {
            'prices': {
                'AAPL': {'close': list(100 + np.cumsum(np.random.randn(30)))}
            },
            'volumes': {
                'AAPL': {'volume': 1000000, 'avg_volume': 1200000}
            }
        }
        result = self.analyzer.analyze_position(symbol, position, market_data)
        self.assertIn('position_var', result)
        self.assertIn('liquidity_risk', result)
        self.assertIn('concentration', result)
        self.assertGreaterEqual(result['position_var'], 0)

    def test_analyze_position_insufficient_price_data(self):
        """分析持仓：价格数据不足"""
        symbol = 'XYZ'
        position = DummyPosition(current_value=5000, weight=0.05)
        market_data = {
            'prices': {
                'XYZ': {'close': [100, 101]}  # 少于20个点
            },
            'volumes': {}
        }
        result = self.analyzer.analyze_position(symbol, position, market_data)
        # 数据不足时应返回默认值
        self.assertEqual(result['position_var'], 0.0)

    def test_calculate_single_position_var(self):
        """计算单一持仓VaR"""
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        var = self.analyzer.calculate_single_position_var('TEST', returns, 0.95)
        self.assertGreater(var, 0)

    def test_calculate_single_position_var_empty_returns(self):
        """计算单一持仓VaR：空收益"""
        var = self.analyzer.calculate_single_position_var('TEST', pd.Series([]), 0.95)
        self.assertEqual(var, 0.0)

    def test_liquidity_risk_for_position_normal_volume(self):
        """流动性风险：正常成交量"""
        market_data = {
            'volumes': {
                'STOCK': {'volume': 1000000, 'avg_volume': 1000000}
            }
        }
        risk = self.analyzer.liquidity_risk_for_position('STOCK', market_data)
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1.0)

    def test_liquidity_risk_for_position_low_volume(self):
        """流动性风险：低成交量"""
        market_data = {
            'volumes': {
                'ILLIQUID': {'volume': 100, 'avg_volume': 1000000}
            }
        }
        risk = self.analyzer.liquidity_risk_for_position('ILLIQUID', market_data)
        # 成交量远低于平均，风险应较高
        self.assertGreater(risk, 0.5)

    def test_liquidity_risk_for_position_missing_data(self):
        """流动性风险：缺失数据"""
        market_data = {'volumes': {}}
        risk = self.analyzer.liquidity_risk_for_position('UNKNOWN', market_data)
        # 缺失数据返回默认中等风险
        self.assertEqual(risk, 0.5)

    def test_calculate_participation_rate_impact_normal(self):
        """参与率冲击：正常订单"""
        market_data = {
            'volumes': {
                'STOCK': {'avg_volume': 1000000}
            },
            'prices': {
                'STOCK': {'spread': 0.002}
            }
        }
        result = self.analyzer.calculate_participation_rate_impact('STOCK', 100000, market_data)
        self.assertIn('participation_rate', result)
        self.assertIn('price_impact', result)
        self.assertIn('liquidity_cost', result)
        # 10%参与率
        self.assertAlmostEqual(result['participation_rate'], 0.1, places=2)
        # 价格冲击应大于0
        self.assertGreater(result['price_impact'], 0)
        self.assertGreater(result['liquidity_cost'], 0)

    def test_calculate_participation_rate_impact_large_order(self):
        """参与率冲击：大订单冲击更大"""
        market_data = {
            'volumes': {'STOCK': {'avg_volume': 1000000}},
            'prices': {'STOCK': {'spread': 0.002}}
        }
        small_order = self.analyzer.calculate_participation_rate_impact('STOCK', 50000, market_data)
        large_order = self.analyzer.calculate_participation_rate_impact('STOCK', 500000, market_data)
        # 大订单冲击应更大
        self.assertGreater(large_order['price_impact'], small_order['price_impact'])
        self.assertGreater(large_order['liquidity_cost'], small_order['liquidity_cost'])

    def test_calculate_participation_rate_impact_missing_data(self):
        """参与率冲击：缺失数据"""
        market_data = {'volumes': {}}
        result = self.analyzer.calculate_participation_rate_impact('UNKNOWN', 100000, market_data)
        self.assertEqual(result['participation_rate'], 0.0)
        self.assertEqual(result['price_impact'], 0.0)

    def test_estimate_liquidation_time_small_position(self):
        """清算时间估算：小持仓快速清算"""
        market_data = {
            'volumes': {'STOCK': {'avg_volume': 1000000}},
            'prices': {'STOCK': {'spread': 0.002}}
        }
        result = self.analyzer.estimate_liquidation_time('STOCK', 50000, market_data, max_participation_rate=0.1)
        self.assertIn('days_required', result)
        self.assertIn('risk_level', result)
        # 5%参与率，1天内清算
        self.assertLessEqual(result['days_required'], 1)
        self.assertEqual(result['risk_level'], 'low')

    def test_estimate_liquidation_time_large_position(self):
        """清算时间估算：大持仓需要多天"""
        market_data = {
            'volumes': {'STOCK': {'avg_volume': 1000000}},
            'prices': {'STOCK': {'spread': 0.002}}
        }
        result = self.analyzer.estimate_liquidation_time('STOCK', 2000000, market_data, max_participation_rate=0.1)
        # 200%参与率，需褐20天
        self.assertGreater(result['days_required'], 10)
        self.assertIn(result['risk_level'], ['high', 'extreme'])

    def test_estimate_liquidation_time_missing_data(self):
        """清算时间估算：缺失数据"""
        market_data = {'volumes': {}}
        result = self.analyzer.estimate_liquidation_time('UNKNOWN', 100000, market_data)
        self.assertEqual(result['days_required'], 999)
        self.assertEqual(result['risk_level'], 'extreme')


if __name__ == '__main__':
    unittest.main(verbosity=2)
