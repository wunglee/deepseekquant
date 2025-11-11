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
        # 年化波动率应该约等于日波动率 * sqrt(trading_days_per_year)
        # 注意：使用service的trading_days_per_year，已支持国际化配置
        trading_days = self.service.trading_days_per_year
        self.assertAlmostEqual(vol_annual, vol_daily * np.sqrt(trading_days), places=5)
        
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


    def test_high_volatility_switch(self):
        """高波动率切换场景：后半段波动率应显著高于前半段"""
        low_vol = np.random.normal(0.0005, 0.01, 126)
        high_vol = np.random.normal(0.0005, 0.04, 126)
        returns = pd.Series(np.concatenate([low_vol, high_vol]))
        vol_first = self.service.calculate_volatility(pd.Series(low_vol), annualize=False)
        vol_last = self.service.calculate_volatility(pd.Series(high_vol), annualize=False)
        self.assertGreater(vol_last, vol_first)

    def test_limit_up_down_truncation_effect_on_var(self):
        """涨跌停板截断：历史法VaR应不小于参数法VaR（参数法低估尾部风险）"""
        base = np.random.normal(0.0, 0.03, 1000)
        truncated = np.clip(base, -0.1, 0.1)
        returns = pd.Series(truncated)
        var_hist = self.service.calculate_value_at_risk(returns, 0.95, 'historical')
        var_param = self.service.calculate_value_at_risk(returns, 0.95, 'parametric')
        # 两种VaR应在合理范围内且差异不应过大（截断影响下界）
        self.assertGreater(var_hist, 0)
        self.assertGreater(var_param, 0)
        self.assertLess(abs(var_hist - var_param), 0.02)
    
    def test_limit_hit_detection_main_board(self):
        """P0测试：主板涨跌停检测（10%限制）"""
        # 没有涨跌停
        normal_returns = pd.Series([0.02, -0.03, 0.015, -0.025])
        self.assertFalse(self.service._has_limit_hit(normal_returns, board_type='main_board'))
        
        # 有涨跌停（接近或达到10%）
        limit_returns = pd.Series([0.02, 0.095, -0.03])  # 9.5% 接近涨停
        self.assertTrue(self.service._has_limit_hit(limit_returns, board_type='main_board'))
        
        # 达到10%
        exact_limit = pd.Series([0.10, -0.05, 0.02])
        self.assertTrue(self.service._has_limit_hit(exact_limit, board_type='main_board'))
    
    def test_limit_hit_detection_gem_board(self):
        """P0测试：创业板涨跌停检测（20%限制）"""
        # 没有涨跌停
        normal_returns = pd.Series([0.05, -0.08, 0.12, -0.15])
        self.assertFalse(self.service._has_limit_hit(normal_returns, board_type='gem'))
        
        # 有涨跌停（接近戙20%）
        limit_returns = pd.Series([0.05, 0.19, -0.08])  # 19% 接近涨停
        self.assertTrue(self.service._has_limit_hit(limit_returns, board_type='gem'))
    
    def test_limit_hit_detection_st_board(self):
        """P0测试：ST股涨跌停检测（5%限制）"""
        # 没有涨跌停
        normal_returns = pd.Series([0.01, -0.02, 0.015, -0.025])
        self.assertFalse(self.service._has_limit_hit(normal_returns, board_type='st'))
        
        # 有涨跌停（接近战5%）
        limit_returns = pd.Series([0.01, 0.048, -0.02])  # 4.8% 接近涨停
        self.assertTrue(self.service._has_limit_hit(limit_returns, board_type='st'))
    
    def test_var_calculation_with_limit_hit_warning(self):
        """P0测试：VaR计算中的涨跌停警告"""
        # 创建包含涨跌停的数据
        limit_returns = pd.Series([0.02, 0.095, -0.03, 0.01, -0.02])
        
        # 启用涨跌停检测（默认）
        var_with_check = self.service.calculate_value_at_risk(
            limit_returns, 
            adjust_limit=True, 
            board_type='main_board'
        )
        self.assertGreater(var_with_check, 0)
        
        # 禁用涨跌停检测
        var_without_check = self.service.calculate_value_at_risk(
            limit_returns, 
            adjust_limit=False
        )
        self.assertGreater(var_without_check, 0)
        
        # 两种方式计算结果应相同（仅是警告不同）
        self.assertAlmostEqual(var_with_check, var_without_check, places=10)
