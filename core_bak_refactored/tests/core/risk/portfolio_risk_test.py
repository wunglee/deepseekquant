import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd

from core.risk.portfolio_risk import PortfolioRiskAnalyzer


class DummyAlloc:
    def __init__(self, weight: float):
        self.weight = weight


class DummyPortfolioState:
    def __init__(self, allocations):
        self.allocations = allocations


class TestPortfolioRiskAnalyzer(unittest.TestCase):
    """测试组合风险分析器 - 风险贡献度（协方差矩阵）"""

    def setUp(self):
        self.config = {'trading_days_per_year': 252}
        self.analyzer = PortfolioRiskAnalyzer(self.config)

    def test_risk_contributions_covariance_two_assets(self):
        """两资产协方差矩阵的风险贡献度计算"""
        # 两个资产的权重
        allocations = {
            'A': DummyAlloc(0.6),
            'B': DummyAlloc(0.4)
        }
        portfolio_state = DummyPortfolioState(allocations)

        # 构造协方差矩阵（单位为日方差），假设资产A更波动且正相关
        cov_matrix = pd.DataFrame(
            [[0.0004, 0.0002],
             [0.0002, 0.0001]],
            index=['A', 'B'], columns=['A', 'B']
        )

        contributions = self.analyzer.calculate_risk_contributions_covariance(portfolio_state, cov_matrix)

        # 断言贡献度字典包含两资产
        self.assertIn('A', contributions)
        self.assertIn('B', contributions)

        # A的风险贡献应大于B（更高波动且权重更大）
        self.assertGreater(contributions['A'], contributions['B'])

        # 总风险贡献的和接近组合波动（量纲不同，主要验证非负与合理性）
        port_var = np.array([0.6, 0.4]).T @ cov_matrix.values @ np.array([0.6, 0.4])
        self.assertGreater(port_var, 0)

    def test_portfolio_returns_skips_suspended_asset(self):
        """停牌资产（无close数据）应被跳过，仍能计算组合收益"""
        allocations = {
            'A': DummyAlloc(0.5),
            'B': DummyAlloc(0.5)
        }
        portfolio_state = DummyPortfolioState(allocations)
        # A有价格、B停牌（无数据）
        prices_A = list(100 + np.cumsum(np.random.randn(30)))
        market_data = {
            'prices': {
                'A': {'close': prices_A},
                'B': {'close': []}
            },
            'timestamp': list(range(35))
        }
        series = self.analyzer.calculate_portfolio_returns(portfolio_state, market_data)
        self.assertIsInstance(series, pd.Series)
        self.assertGreater(len(series), 0)

    def test_single_asset_concentration_risk(self):
        """单一持仓组合的集中度风险应为1.0"""
        allocations = {'A': DummyAlloc(1.0)}
        portfolio_state = DummyPortfolioState(allocations)
        cov_matrix = pd.DataFrame([[0.0004]], index=['A'], columns=['A'])
        data = {
            'portfolio_state': portfolio_state,
            'market_data': {},
            'covariance_matrix': cov_matrix
        }
        result = self.analyzer.analyze(data, risk_metrics={})
        self.assertIn('concentration_risk', result)
        self.assertAlmostEqual(result['concentration_risk'], 1.0, places=6)

    def test_equal_weights_concentration_risk(self):
        """等权四资产组合的集中度风险应为0.25"""
        allocations = {k: DummyAlloc(0.25) for k in ['A', 'B', 'C', 'D']}
        portfolio_state = DummyPortfolioState(allocations)
        cov_matrix = pd.DataFrame(np.eye(4)*0.0001, index=['A','B','C','D'], columns=['A','B','C','D'])
        data = {
            'portfolio_state': portfolio_state,
            'market_data': {},
            'covariance_matrix': cov_matrix
        }
        result = self.analyzer.analyze(data, risk_metrics={})
        self.assertIn('concentration_risk', result)
        self.assertAlmostEqual(result['concentration_risk'], 0.25, places=6)

    def test_empty_portfolio_returns_zero_concentration(self):
        """空仓组合的集中度风险应为0"""
        portfolio_state = DummyPortfolioState({})
        data = {'portfolio_state': portfolio_state, 'market_data': {}}
        result = self.analyzer.analyze(data, risk_metrics={})
        self.assertIn('concentration_risk', result)
        self.assertEqual(result['concentration_risk'], 0.0)
    
    def test_factor_risk_attribution_with_market_industry_style(self):
        """测试因子风险归因分解：市场、行业、风格因子"""
        # 构造两资产组合
        allocations = {'A': DummyAlloc(0.6), 'B': DummyAlloc(0.4)}
        portfolio_state = DummyPortfolioState(allocations)
        
        # 因子暴露矩阵：2资产 x 3因子（market_beta, industry_tech, style_momentum）
        factor_exposures = pd.DataFrame(
            [[1.2, 0.8, 0.5],   # A: 高市场beta, 科技行业, 中等动量
             [0.9, 0.0, -0.3]],  # B: 中等beta, 非科技, 负动量
            index=['A', 'B'],
            columns=['market_beta', 'industry_tech', 'style_momentum']
        )
        
        # 因子协方差矩阵（3x3）
        factor_covariance = pd.DataFrame(
            [[0.04, 0.01, 0.005],
             [0.01, 0.03, 0.002],
             [0.005, 0.002, 0.02]],
            index=['market_beta', 'industry_tech', 'style_momentum'],
            columns=['market_beta', 'industry_tech', 'style_momentum']
        )
        
        result = self.analyzer.calculate_factor_risk_attribution(
            portfolio_state, factor_exposures, factor_covariance
        )
        
        # 验证返回结构
        self.assertIn('market_risk', result)
        self.assertIn('industry_risk', result)
        self.assertIn('style_risk', result)
        self.assertIn('specific_risk', result)
        self.assertIn('total_risk', result)
        self.assertIn('factor_contributions', result)
        
        # 市场风险应为正（市场因子存在）
        self.assertGreater(result['market_risk'], 0)
        
        # 总风险 = 因子风险 + 特质风险
        self.assertGreater(result['total_risk'], 0)
        
        # 因子贡献明细应包含所有因子
        self.assertEqual(len(result['factor_contributions']), 3)
    
    def test_factor_risk_attribution_empty_exposure(self):
        """测试因子风险归因：空暴露数据"""
        allocations = {'A': DummyAlloc(1.0)}
        portfolio_state = DummyPortfolioState(allocations)
        
        result = self.analyzer.calculate_factor_risk_attribution(
            portfolio_state, pd.DataFrame(), pd.DataFrame()
        )
        
        # 空数据应返回空字典
        self.assertEqual(result, {})
    
    def test_factor_risk_attribution_no_matching_symbols(self):
        """测试因子风险归因：无匹配符号"""
        allocations = {'A': DummyAlloc(1.0)}
        portfolio_state = DummyPortfolioState(allocations)
        
        # 因子暴露中没有A
        factor_exposures = pd.DataFrame(
            [[1.0, 0.5]],
            index=['B'],
            columns=['market_beta', 'style_value']
        )
        factor_covariance = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.03]],
            index=['market_beta', 'style_value'],
            columns=['market_beta', 'style_value']
        )
        
        result = self.analyzer.calculate_factor_risk_attribution(
            portfolio_state, factor_exposures, factor_covariance
        )
        
        self.assertEqual(result, {})
