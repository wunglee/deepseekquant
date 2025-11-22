import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd

from core_bak_refactored.core.risk.portfolio_risk import PortfolioRiskAnalyzer


class DummyAlloc:
    def __init__(self, weight: float):
        self.weight = weight


class DummyPortfolioState:
    def __init__(self, allocations):
        self.allocations = allocations


class TestPortfolioRiskAnalyzer(unittest.TestCase):
    """测试组合风险分析器 - 风险贡献度（协方差矩阵） + P1增强：7维度分析"""

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
    
    def test_report_snapshot_fields_completeness(self):
        """验收测试：报告快照字段完整性（P2.1专家补充）"""
        allocations = {'A': DummyAlloc(0.5), 'B': DummyAlloc(0.5)}
        portfolio_state = DummyPortfolioState(allocations)
        cov_matrix = pd.DataFrame(
            [[0.0004, 0.0002], [0.0002, 0.0001]],
            index=['A', 'B'], columns=['A', 'B']
        )
        
        # 构造市场数据以生成 portfolio_returns
        prices_A = list(100 + np.cumsum(np.random.randn(80)))
        prices_B = list(100 + np.cumsum(np.random.randn(80)))
        market_data = {
            'prices': {
                'A': {'close': prices_A},
                'B': {'close': prices_B}
            },
            'timestamp': list(range(80)),
            'last_updated_ts': 1700000000
        }
        
        data = {
            'portfolio_state': portfolio_state,
            'market_data': market_data,
            'covariance_matrix': cov_matrix
        }
        
        result = self.analyzer.analyze(data, risk_metrics={})
        
        # 验证 report_snapshot 字段存在
        self.assertIn('report_snapshot', result)
        rs = result['report_snapshot']
        
        # 验证基础字段（第20轮迭代目标）
        required_fields = [
            'report_id', 'environment', 'timestamp', 'market_type',
            'calculation_id', 'trigger_reason', 'cache_status', 'data_freshness_seconds'
        ]
        for field in required_fields:
            self.assertIn(field, rs, f"缺少必要字段: {field}")
        
        # P2.1专家补充字段验证
        p21_fields = ['calculation_cost_ms', 'approval_status', 'risk_rating', 'compliance_flags']
        for field in p21_fields:
            self.assertIn(field, rs, f"P2.1缺少补充字段: {field}")
        
        # 验证字段类型
        self.assertIsInstance(rs['calculation_id'], str)
        self.assertIsInstance(rs['data_freshness_seconds'], int)
        self.assertIn(rs['trigger_reason'], ['SCHEDULED', 'VOLATILITY_SPIKE'])
        self.assertIsInstance(rs['calculation_cost_ms'], int)
        self.assertGreaterEqual(rs['calculation_cost_ms'], 0)
        self.assertIn(rs['approval_status'], ['AUTO_APPROVED', 'PENDING', 'REJECTED'])
        self.assertIn(rs['risk_rating'], ['LOW', 'MEDIUM', 'HIGH'])
        self.assertIsInstance(rs['compliance_flags'], list)
    
    def test_model_health_jp_market_thresholds(self):
        """验收测试：JP市场模型健康分级阈值（P2.1专家优化：60/240）"""
        # P2.1专家建议：JP市场阈值调整为min_points=60, optimal_points=240
        config = {'trading_days_per_year': 245, 'market_type': 'JP'}
        analyzer = PortfolioRiskAnalyzer(config)
        
        allocations = {'A': DummyAlloc(1.0)}
        portfolio_state = DummyPortfolioState(allocations)
        cov_matrix = pd.DataFrame([[0.0004]], index=['A'], columns=['A'])
        
        # 测试不同数据点数的分级（基于新阈值60/240）
        test_cases = [
            (240, 'EXCELLENT', 'NONE'),      # >= optimal_points (240)
            (100, 'GOOD', 'MINIMAL'),        # >= min_points (60)
            (50, 'FAIR', 'MODERATE'),        # >= min_points * 0.7 (42)
            (35, 'POOR', 'SIGNIFICANT'),     # >= min_points * 0.5 (30)
            (20, 'INSUFFICIENT', 'SEVERE')   # < min_points * 0.5
        ]
        
        for data_points, expected_quality, expected_degradation in test_cases:
            # 构造有data_points个数据点的市场数据
            prices = list(100 + np.cumsum(np.random.randn(data_points + 1)))
            market_data = {
                'prices': {'A': {'close': prices}},
                'timestamp': list(range(data_points + 1))
            }
            
            data = {
                'portfolio_state': portfolio_state,
                'market_data': market_data,
                'covariance_matrix': cov_matrix
            }
            result = analyzer.analyze(data, risk_metrics={})
            
            self.assertIn('model_health', result)
            mh = result['model_health']
            
            # 数据点数应该接近（收益序列比价格序列少1个点）
            self.assertAlmostEqual(mh['data_points'], data_points, delta=1)
            self.assertEqual(mh['quality'], expected_quality, 
                           f"数据点{data_points}时，期望质量为{expected_quality}，实际为{mh['quality']}")
            self.assertEqual(mh['degradation_level'], expected_degradation,
                           f"数据点{data_points}时，期望降级为{expected_degradation}，实际为{mh['degradation_level']}")

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
    
    def test_seven_dimension_analysis_complete(self):
        """测试P1增强：7维度组合风险分析"""
        # 构造两资产组合
        allocations = {'A': DummyAlloc(0.6), 'B': DummyAlloc(0.4)}
        portfolio_state = DummyPortfolioState(allocations)
        
        # 构造市场数据（模拟30天的价格）
        np.random.seed(42)
        prices_A = list(100 + np.cumsum(np.random.randn(30) * 2))
        prices_B = list(100 + np.cumsum(np.random.randn(30) * 1.5))
        
        market_data = {
            'prices': {
                'A': {'close': prices_A},
                'B': {'close': prices_B}
            },
            'timestamp': list(range(30))
        }
        
        # 协方差矩阵
        cov_matrix = pd.DataFrame(
            [[0.0004, 0.0002],
             [0.0002, 0.0001]],
            index=['A', 'B'], columns=['A', 'B']
        )
        
        data = {
            'portfolio_state': portfolio_state,
            'market_data': market_data,
            'covariance_matrix': cov_matrix
        }
        
        result = self.analyzer.analyze(data, risk_metrics={})
        
        # 验证7维度是否存在
        self.assertIn('total_risk', result)
        self.assertIn('volatility', result)
        self.assertIn('var_95', result)
        self.assertIn('cvar_95', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('max_drawdown', result)
        self.assertIn('risk_contributions', result)
        
        # 验证数值合理性
        self.assertGreater(result['volatility'], 0, "波动率应为正")
        self.assertEqual(result['total_risk'], result['volatility'], "总风险=波动率（专家指导）")
        self.assertGreater(result['var_95'], 0, "VaR应为正（损失金额）")
        self.assertGreater(result['cvar_95'], result['var_95'], "CVaR应大于VaR")
        self.assertIsInstance(result['sharpe_ratio'], float, "夏普比率应为float")
        self.assertGreaterEqual(result['max_drawdown'], 0, "最大回撤应非负")
        
        # 验证风险贡献
        self.assertIn('A', result['risk_contributions'])
        self.assertIn('B', result['risk_contributions'])
        
        # 验证传统字段仍然存在
        self.assertIn('portfolio_returns', result)
        self.assertIn('concentration_risk', result)
    
    def test_seven_dimension_analysis_with_empty_market_data(self):
        """测试7维度分析：空市场数据"""
        allocations = {'A': DummyAlloc(1.0)}
        portfolio_state = DummyPortfolioState(allocations)
        
        data = {
            'portfolio_state': portfolio_state,
            'market_data': None,
            'covariance_matrix': None
        }
        
        result = self.analyzer.analyze(data, risk_metrics={})
        
        # 应该返回零值，但结构完整
        self.assertEqual(result['total_risk'], 0.0)
        self.assertEqual(result['volatility'], 0.0)
        self.assertEqual(result['var_95'], 0.0)
        self.assertEqual(result['cvar_95'], 0.0)
        self.assertEqual(result['sharpe_ratio'], 0.0)
        self.assertEqual(result['max_drawdown'], 0.0)
        self.assertEqual(result['risk_contributions'], {})
    
    def test_auto_generate_robust_covariance_for_risk_contributions(self):
        """测试自动生成稳健协方差矩阵用于风险贡献计算"""
        allocations = {'A': DummyAlloc(0.5), 'B': DummyAlloc(0.5)}
        portfolio_state = DummyPortfolioState(allocations)
        
        # 构造市场数据（但不提供协方差/相关性矩阵）
        np.random.seed(42)
        prices_A = list(100 + np.cumsum(np.random.randn(50) * 2))
        prices_B = list(100 + np.cumsum(np.random.randn(50) * 1.5))
        
        market_data = {
            'prices': {
                'A': {'close': prices_A},
                'B': {'close': prices_B}
            },
            'timestamp': list(range(50))
        }
        
        data = {
            'portfolio_state': portfolio_state,
            'market_data': market_data,
            # 故意不提供 covariance_matrix 和 correlation_matrix
        }
        
        result = self.analyzer.analyze(data, risk_metrics={})
        
        # 验证风险贡献度已自动生成
        self.assertIn('risk_contributions', result)
        self.assertIn('A', result['risk_contributions'])
        self.assertIn('B', result['risk_contributions'])
        self.assertGreater(result['risk_contributions']['A'], 0)
        self.assertGreater(result['risk_contributions']['B'], 0)
        
        # 验证自动生成标记
        self.assertTrue(result.get('_auto_generated_covariance', False))
        
        # 验证其他指标仍然正常
        self.assertGreater(result['volatility'], 0)
        self.assertGreater(result['var_95'], 0)
    
    def test_auto_generate_robust_covariance_insufficient_data(self):
        """测试自动生成稳健矩阵：数据不足"""
        allocations = {'A': DummyAlloc(1.0)}
        portfolio_state = DummyPortfolioState(allocations)
        
        # 仅提供1个数据点（无法计算协方差）
        market_data = {
            'prices': {
                'A': {'close': [100]}
            }
        }
        
        data = {
            'portfolio_state': portfolio_state,
            'market_data': market_data
        }
        
        result = self.analyzer.analyze(data, risk_metrics={})
        
        # 数据不足时风险贡献为空
        self.assertEqual(result['risk_contributions'], {})
        self.assertFalse(result.get('_auto_generated_covariance', False))

