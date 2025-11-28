"""
因子模型测试
"""

import unittest
import numpy as np
import pandas as pd
from core_bak_refactored.core.risk.factor_model import (
    FactorModelEstimator,
    FactorModelConfig
)


class TestFactorModel(unittest.TestCase):
    """测试因子模型"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.config = FactorModelConfig(market='US', n_factors=5)
        self.estimator = FactorModelEstimator(self.config)
    
    def _generate_factor_data(self, n_assets=50, n_factors=5, T=252):
        """生成模拟因子数据"""
        # 生成因子收益
        factor_returns = pd.DataFrame(
            np.random.randn(T, n_factors) * 0.01,
            columns=[f'Factor_{i}' for i in range(n_factors)]
        )
        
        # 生成因子载荷
        loadings = np.random.randn(n_assets, n_factors) * 0.5
        
        # 生成资产收益: r = B*f + epsilon
        specific_returns = np.random.randn(T, n_assets) * 0.005
        asset_returns = factor_returns.values @ loadings.T + specific_returns
        
        returns_df = pd.DataFrame(
            asset_returns,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        return returns_df, factor_returns
    
    def test_estimate_factor_loadings(self):
        """测试因子载荷估计"""
        returns, factor_returns = self._generate_factor_data(50, 5, 252)
        
        # 估计载荷
        loadings = self.estimator.estimate_factor_loadings(returns, factor_returns)
        
        # 验证
        self.assertEqual(loadings.shape, (50, 5))
        self.assertFalse(loadings.isnull().any().any())
        self.assertIsNotNone(self.estimator.specific_variance)
        self.assertEqual(len(self.estimator.specific_variance), 50)
    
    def test_estimate_factor_covariance(self):
        """测试因子协方差估计"""
        _, factor_returns = self._generate_factor_data(50, 5, 252)
        
        # 估计协方差
        factor_cov = self.estimator.estimate_factor_covariance(factor_returns)
        
        # 验证
        self.assertEqual(factor_cov.shape, (5, 5))
        self.assertTrue(np.allclose(factor_cov, factor_cov.T))  # 对称性
        self.assertTrue((np.linalg.eigvals(factor_cov) > 0).all())  # 正定性
    
    def test_compute_covariance_matrix_factor_model(self):
        """测试纯因子模型协方差"""
        returns, factor_returns = self._generate_factor_data(50, 5, 252)
        
        # 计算协方差
        cov_matrix, metadata = self.estimator.compute_covariance_matrix(
            returns,
            factor_returns,
            use_hybrid=False
        )
        
        # 验证
        self.assertEqual(cov_matrix.shape, (50, 50))
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))
        self.assertTrue(metadata['success'])
        self.assertEqual(metadata['method'], 'factor_model')
        self.assertEqual(metadata['n_factors'], 5)
    
    def test_compute_covariance_matrix_hybrid(self):
        """测试混合模型协方差"""
        returns, factor_returns = self._generate_factor_data(50, 5, 252)
        
        # 计算协方差
        cov_matrix, metadata = self.estimator.compute_covariance_matrix(
            returns,
            factor_returns,
            use_hybrid=True
        )
        
        # 验证
        self.assertEqual(cov_matrix.shape, (50, 50))
        self.assertTrue(metadata['success'])
        self.assertEqual(metadata['method'], 'hybrid')
        self.assertEqual(metadata['shrinkage_alpha'], 0.7)
    
    def test_generate_statistical_factors(self):
        """测试PCA统计因子生成"""
        returns, _ = self._generate_factor_data(50, 5, 252)
        
        # 生成统计因子
        factor_returns = self.estimator._generate_statistical_factors(returns, n_factors=10)
        
        # 验证
        self.assertEqual(factor_returns.shape, (252, 10))
        self.assertFalse(factor_returns.isnull().any().any())
    
    def test_auto_factor_generation(self):
        """测试自动因子生成"""
        returns, _ = self._generate_factor_data(100, 5, 252)
        
        # 不提供因子，自动生成
        cov_matrix, metadata = self.estimator.compute_covariance_matrix(
            returns,
            factor_returns=None,  # 自动生成
            use_hybrid=True
        )
        
        # 验证
        self.assertTrue(metadata['success'])
        self.assertEqual(cov_matrix.shape, (100, 100))
        self.assertGreater(metadata['n_factors'], 0)
    
    def test_performance_comparison(self):
        """测试性能对比：因子模型 vs 样本协方差"""
        import time
        
        returns, _ = self._generate_factor_data(100, 10, 252)
        
        # 样本协方差（基准）
        start = time.time()
        sample_cov = returns.cov()
        time_sample = time.time() - start
        
        # 因子模型协方差
        start = time.time()
        factor_cov, metadata = self.estimator.compute_covariance_matrix(
            returns,
            factor_returns=None,
            use_hybrid=False
        )
        time_factor = time.time() - start
        
        # 验证
        self.assertTrue(metadata['success'])
        print(f"\n样本协方差: {time_sample:.4f}秒")
        print(f"因子模型: {time_factor:.4f}秒")
        print(f"因子数: {metadata['n_factors']}")
        print(f"因子贡献: {metadata['factor_contribution']:.1%}")
    
    def test_small_sample_fallback(self):
        """测试小样本回退机制"""
        # 生成小样本数据
        returns, _ = self._generate_factor_data(50, 5, 30)  # 仅30个观测
        
        # 计算协方差
        cov_matrix, metadata = self.estimator.compute_covariance_matrix(
            returns,
            factor_returns=None
        )
        
        # 应该回退到样本协方差或使用简化方法
        self.assertEqual(cov_matrix.shape, (50, 50))
    
    def test_get_factor_summary(self):
        """测试因子摘要统计"""
        returns, factor_returns = self._generate_factor_data(50, 5, 252)
        
        # 计算协方差
        self.estimator.compute_covariance_matrix(returns, factor_returns)
        
        # 获取摘要
        summary = self.estimator.get_factor_summary()
        
        # 验证
        self.assertIn('n_assets', summary)
        self.assertIn('n_factors', summary)
        self.assertIn('avg_loading', summary)
        self.assertEqual(summary['n_assets'], 50)
        self.assertEqual(summary['n_factors'], 5)


if __name__ == '__main__':
    unittest.main()
