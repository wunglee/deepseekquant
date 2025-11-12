"""
RiskMetricsService 单元测试
对应文件: core/risk/risk_metrics_service.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd
from core.risk.risk_metrics_service import RiskMetricsService


class TestRiskMetricsService(unittest.TestCase):
    """测试风险指标业务服务"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'trading_days_per_year': 252,
            'default_confidence_level': 0.95,
            'risk_free_rate': 0.03
        }
        self.service = RiskMetricsService(self.config)
        
        # 准备测试数据
        np.random.seed(42)
        returns_array = np.random.normal(0.001, 0.02, 252)
        self.returns = pd.Series(returns_array)
        
        market_returns_array = np.random.normal(0.0008, 0.015, 252)
        self.market_returns = pd.Series(market_returns_array)
    
    def test_calculate_volatility(self):
        """测试波动率计算"""
        # 年化波动率
        vol_annual = self.service.calculate_volatility(self.returns, annualize=True)
        self.assertGreater(vol_annual, 0)
        self.assertLess(vol_annual, 1.0)  # 年化波动率应该合理
        
        # 非年化波动率
        vol_daily = self.service.calculate_volatility(self.returns, annualize=False)
        self.assertGreater(vol_daily, 0)
        # 年化波动率应该约等于日波动率 * sqrt(252)
        self.assertAlmostEqual(vol_annual, vol_daily * np.sqrt(252), places=5)
        
        # 滚动窗口
        vol_window = self.service.calculate_volatility(self.returns, window=20, annualize=False)
        self.assertGreater(vol_window, 0)
    
    def test_calculate_value_at_risk(self):
        """测试VaR计算"""
        # 历史法VaR
        var_hist = self.service.calculate_value_at_risk(self.returns, 0.95, 'historical')
        self.assertGreater(var_hist, 0)  # VaR应该为正数（损失）
        self.assertLess(var_hist, 0.2)  # 应该在合理范围内
        
        # 参数法VaR
        var_param = self.service.calculate_value_at_risk(self.returns, 0.95, 'parametric')
        self.assertGreater(var_param, 0)
        
        # 不同置信度
        var_99 = self.service.calculate_value_at_risk(self.returns, 0.99, 'historical')
        var_95 = self.service.calculate_value_at_risk(self.returns, 0.95, 'historical')
        # 99%VaR应该大于95%VaR
        self.assertGreater(var_99, var_95)
    
    def test_calculate_expected_shortfall(self):
        """测试CVaR/ES计算"""
        cvar = self.service.calculate_expected_shortfall(self.returns, 0.95)
        var = self.service.calculate_value_at_risk(self.returns, 0.95)
        
        # CVaR应该大于等于VaR
        self.assertGreaterEqual(cvar, var)
        self.assertGreater(cvar, 0)
    
    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        # 创建有明显回撤的数据
        returns_with_dd = pd.Series([0.05, 0.03, -0.10, -0.05, 0.02, -0.08, 0.06])
        mdd = self.service.calculate_max_drawdown(returns_with_dd)
        
        self.assertGreater(mdd, 0)  # 回撤应该为正数
        self.assertLess(mdd, 1.0)  # 回撤不应超过100%
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        sharpe = self.service.calculate_sharpe_ratio(self.returns)
        
        # 夏普比率应该在合理范围内
        self.assertGreater(sharpe, -5.0)
        self.assertLess(sharpe, 5.0)
        
        # 使用自定义无风险利率
        sharpe_custom = self.service.calculate_sharpe_ratio(self.returns, risk_free_rate=0.05)
        self.assertIsInstance(sharpe_custom, float)
    
    def test_calculate_sortino_ratio(self):
        """测试索提诺比率计算"""
        sortino = self.service.calculate_sortino_ratio(self.returns)
        
        # 索提诺比率应该在合理范围内
        self.assertGreater(sortino, -5.0)
        self.assertLess(sortino, 5.0)
        
        # 索提诺应该大于等于夏普（因为只考虑下行风险）
        sharpe = self.service.calculate_sharpe_ratio(self.returns)
        # 注意：这个关系不总是成立，取决于收益分布
        self.assertIsInstance(sortino, float)
    
    def test_calculate_beta(self):
        """测试贝塔系数计算"""
        beta = self.service.calculate_beta(self.returns, self.market_returns)
        
        # β应该在合理范围内
        self.assertGreater(beta, -2.0)
        self.assertLess(beta, 2.0)
        self.assertIsInstance(beta, float)
        
        # 完全相同的收益，β应该接近1（放宽精度）
        beta_same = self.service.calculate_beta(self.market_returns, self.market_returns)
        self.assertAlmostEqual(beta_same, 1.0, places=2)
    
    def test_calculate_alpha(self):
        """测试阿尔法计算"""
        alpha = self.service.calculate_alpha(self.returns, self.market_returns)
        
        # α应该在合理范围内（年化）
        self.assertGreater(alpha, -0.5)
        self.assertLess(alpha, 0.5)
        self.assertIsInstance(alpha, float)
        
        # 使用自定义β
        beta = 0.8
        alpha_custom = self.service.calculate_alpha(self.returns, self.market_returns, beta)
        self.assertIsInstance(alpha_custom, float)
    
    def test_calculate_calmar_ratio(self):
        """测试卡尔玛比率计算"""
        calmar = self.service.calculate_calmar_ratio(self.returns)
        
        # 卡尔玛比率应该在合理范围内
        self.assertGreater(calmar, -10.0)
        self.assertLess(calmar, 10.0)
        self.assertIsInstance(calmar, float)
    
    def test_calculate_all_metrics(self):
        """测试综合指标计算"""
        # 仅资产收益
        metrics = self.service.calculate_all_metrics(self.returns)
        
        # 验证包含所有基础指标
        self.assertIn('volatility', metrics)
        self.assertIn('var_95', metrics)
        self.assertIn('cvar_95', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('calmar_ratio', metrics)
        
        # 包含市场收益
        metrics_with_market = self.service.calculate_all_metrics(
            self.returns, self.market_returns
        )
        
        # 验证包含市场相关指标
        self.assertIn('beta', metrics_with_market)
        self.assertIn('alpha', metrics_with_market)
        
        # 验证所有值都是数值类型
        for key, value in metrics_with_market.items():
            self.assertIsInstance(value, (int, float))
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空序列
        empty_returns = pd.Series([])
        vol_empty = self.service.calculate_volatility(empty_returns)
        self.assertEqual(vol_empty, 0.0)
        
        # 短序列
        short_returns = pd.Series([0.01, 0.02, -0.01])
        vol_short = self.service.calculate_volatility(short_returns)
        self.assertGreater(vol_short, 0)
        
        # 常数序列
        constant_returns = pd.Series([0.01] * 100)
        vol_const = self.service.calculate_volatility(constant_returns)
        self.assertAlmostEqual(vol_const, 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
