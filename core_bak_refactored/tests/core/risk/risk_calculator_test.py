import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd

from core.risk.risk_calculator import RiskCalculator


class RiskCalculatorTest(unittest.TestCase):
    """测试风险计算器协调功能"""

    def setUp(self):
        self.config = {
            'trading_days_per_year': 252,
            'default_confidence_level': 0.95,
            'risk_free_rate': 0.03,
            'monte_carlo_sims': 1000
        }
        self.calculator = RiskCalculator(self.config)
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        self.market_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))

    def test_calculate_volatility_delegates_to_service(self):
        """波动率计算应委托给服务层"""
        vol = self.calculator.calculate_volatility(self.returns, annualize=True)
        self.assertGreater(vol, 0)
        self.assertLess(vol, 1.0)

    def test_calculate_correlation_matrix_valid_input(self):
        """相关性矩阵计算：有效输入"""
        asset_returns = pd.DataFrame({
            'A': self.returns,
            'B': self.market_returns
        })
        corr_matrix = self.calculator.calculate_correlation_matrix(asset_returns)
        self.assertEqual(corr_matrix.shape, (2, 2))
        # 对角线应为1
        self.assertAlmostEqual(corr_matrix.loc['A', 'A'], 1.0, places=5)
        self.assertAlmostEqual(corr_matrix.loc['B', 'B'], 1.0, places=5)

    def test_calculate_var_historical_delegates(self):
        """VaR历史法应委托给服务层"""
        var = self.calculator.calculate_var_historical(self.returns, 0.95)
        self.assertGreater(var, 0)

    def test_calculate_var_parametric_delegates(self):
        """VaR参数法应委托给服务层"""
        var = self.calculator.calculate_var_parametric(self.returns, 0.95)
        self.assertGreater(var, 0)

    def test_calculate_max_drawdown_delegates(self):
        """最大回撤应委托给服务层"""
        mdd = self.calculator.calculate_max_drawdown(self.returns)
        self.assertGreaterEqual(mdd, 0)

    def test_calculate_all_metrics_from_returns_data(self):
        """综合指标计算：从收益数据"""
        data = {
            'returns': self.returns.tolist(),
            'market_returns': self.market_returns.tolist()
        }
        metrics = self.calculator.calculate_all_metrics(data)
        self.assertIn('volatility', metrics)
        self.assertIn('var_95', metrics)
        self.assertIn('beta', metrics)

    def test_calculate_all_metrics_from_prices_data(self):
        """综合指标计算：从价格数据"""
        prices = np.exp(np.cumsum(self.returns.values))
        data = {'prices': prices.tolist()}
        metrics = self.calculator.calculate_all_metrics(data)
        self.assertIn('volatility', metrics)
        self.assertIn('var_95', metrics)

    def test_calculate_all_metrics_insufficient_data(self):
        """综合指标计算：数据不足应返回空"""
        data = {'returns': [0.01, 0.02]}  # 少于20个点
        metrics = self.calculator.calculate_all_metrics(data)
        self.assertEqual(metrics, {})

    def test_extract_returns_from_dict(self):
        """提取收益序列：使用预处理器"""
        data = {'returns': [0.01, 0.02, -0.01]}
        returns = self.calculator.preprocessor.extract_returns_from_dict(data)
        self.assertEqual(len(returns), 3)

    def test_extract_market_returns_from_dict(self):
        """提取市场收益：使用预处理器"""
        data = {'market_returns': [0.01, 0.02]}
        market_returns = self.calculator.preprocessor.extract_market_returns_from_dict(data)
        self.assertIsNotNone(market_returns)
        self.assertEqual(len(market_returns), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
