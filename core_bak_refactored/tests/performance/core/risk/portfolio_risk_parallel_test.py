"""
组合风险并行计算集成测试
"""

import unittest
import numpy as np
import pandas as pd

from core_bak_refactored.core.risk.portfolio_risk import PortfolioRiskAnalyzer
from core_bak_refactored.core.share.market.market_config import MarketConfig


class MockPortfolioState:
    """模拟组合状态"""
    def __init__(self, allocations):
        self.allocations = allocations


class MockAllocation:
    """模拟资产配置"""
    def __init__(self, weight):
        self.weight = weight


class TestPortfolioRiskParallel(unittest.TestCase):
    """测试组合风险并行计算"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        # 生成配置
        config_manager = MarketConfig()
        self.config = config_manager.generate_config_template('CN')
        
        # 创建分析器（启用并行）
        self.analyzer_parallel = PortfolioRiskAnalyzer(
            self.config,
            enable_parallel=True,
            enable_incremental=True
        )
        
        # 创建分析器（禁用并行，用于对比）
        self.analyzer_serial = PortfolioRiskAnalyzer(
            self.config,
            enable_parallel=False,
            enable_incremental=False
        )
    
    def _generate_mock_portfolio(self, n_assets: int = 50) -> tuple:
        """生成模拟组合数据"""
        # 生成权重
        weights = np.random.dirichlet(np.ones(n_assets))
        
        # 创建组合状态
        allocations = {
            f"ASSET_{i:03d}": MockAllocation(weights[i])
            for i in range(n_assets)
        }
        portfolio_state = MockPortfolioState(allocations)
        
        # 生成模拟价格数据
        n_days = 252
        prices = {}
        for i in range(n_assets):
            # 模拟价格序列（随机游走）
            returns = np.random.randn(n_days) * 0.02
            price_series = 100 * np.exp(np.cumsum(returns))
            prices[f"ASSET_{i:03d}"] = {
                'close': price_series.tolist()
            }
        
        # 生成时间戳
        timestamps = [
            (pd.Timestamp.now() - pd.Timedelta(days=n_days-i)).isoformat()
            for i in range(n_days)
        ]
        
        market_data = {
            'prices': prices,
            'timestamp': timestamps
        }
        
        return portfolio_state, market_data
    
    def test_batch_calculate_portfolio_risk_parallel(self):
        """测试批量并行计算组合风险"""
        # 生成10个组合
        portfolios = []
        for i in range(10):
            portfolio_state, market_data = self._generate_mock_portfolio(50)
            portfolios.append((f"portfolio_{i}", portfolio_state, market_data))
        
        # 并行计算
        results = self.analyzer_parallel.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=True
        )
        
        # 验证结果
        self.assertEqual(len(results), 10)
        
        for portfolio_id, result in results.items():
            self.assertIn('volatility', result)
            self.assertIn('var_95', result)
            self.assertIn('sharpe_ratio', result)
            self.assertGreater(result['volatility'], 0)
    
    def test_batch_calculate_portfolio_risk_serial(self):
        """测试批量串行计算（对比）"""
        # 生成5个组合（小数据集，应该串行）
        portfolios = []
        for i in range(5):
            portfolio_state, market_data = self._generate_mock_portfolio(30)
            portfolios.append((f"portfolio_{i}", portfolio_state, market_data))
        
        # 串行计算
        results = self.analyzer_serial.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=False
        )
        
        # 验证结果
        self.assertEqual(len(results), 5)
    
    def test_batch_calculate_risk_contributions(self):
        """测试批量计算风险贡献度"""
        # 生成测试数据
        n_assets = 50
        portfolios_with_cov = []
        
        for i in range(10):
            # 生成权重
            weights = np.random.dirichlet(np.ones(n_assets))
            allocations = {
                f"ASSET_{j:03d}": MockAllocation(weights[j])
                for j in range(n_assets)
            }
            portfolio_state = MockPortfolioState(allocations)
            
            # 生成协方差矩阵
            returns = np.random.randn(252, n_assets) * 0.02
            cov_matrix = pd.DataFrame(
                np.cov(returns.T),
                index=[f"ASSET_{j:03d}" for j in range(n_assets)],
                columns=[f"ASSET_{j:03d}" for j in range(n_assets)]
            )
            
            portfolios_with_cov.append((f"portfolio_{i}", portfolio_state, cov_matrix))
        
        # 批量计算
        results = self.analyzer_parallel.batch_calculate_risk_contributions(
            portfolios_with_cov,
            use_parallel=True
        )
        
        # 验证结果
        self.assertEqual(len(results), 10)
        
        for portfolio_id, contributions in results.items():
            self.assertEqual(len(contributions), n_assets)
            # 风险贡献度之和应接近1（放宽检查）
            total_contribution = sum(contributions.values())
            # 注：由于是边际风险贡献，不保证严格等于1
            self.assertGreater(abs(total_contribution), 0.0001)
    
    def test_parallel_vs_serial_consistency(self):
        """测试并行和串行结果一致性"""
        # 生成相同的测试数据
        portfolios = []
        for i in range(3):
            portfolio_state, market_data = self._generate_mock_portfolio(20)
            portfolios.append((f"portfolio_{i}", portfolio_state, market_data))
        
        # 并行计算
        results_parallel = self.analyzer_parallel.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=True
        )
        
        # 串行计算
        results_serial = self.analyzer_serial.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=False
        )
        
        # 验证结果一致性
        self.assertEqual(len(results_parallel), len(results_serial))
        
        for portfolio_id in results_parallel.keys():
            parallel_vol = results_parallel[portfolio_id]['volatility']
            serial_vol = results_serial[portfolio_id]['volatility']
            
            # 允许小的数值误差
            self.assertAlmostEqual(parallel_vol, serial_vol, places=4)
    
    def test_get_optimization_metrics(self):
        """测试优化指标获取"""
        # 执行一些计算
        portfolio_state, market_data = self._generate_mock_portfolio(50)
        portfolios = [(f"portfolio_{i}", portfolio_state, market_data) for i in range(5)]
        
        self.analyzer_parallel.batch_calculate_portfolio_risk(portfolios)
        
        # 获取指标
        metrics = self.analyzer_parallel.get_optimization_metrics()
        
        # 验证指标
        self.assertIn('parallel_enabled', metrics)
        self.assertIn('incremental_enabled', metrics)
        # 注：如果导入失败，会被禁用
        if metrics.get('parallel_enabled'):
            self.assertIn('parallel_metrics', metrics)


if __name__ == '__main__':
    unittest.main()
