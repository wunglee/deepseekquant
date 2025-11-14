"""
StatisticalCalculator 单元测试
对应文件: infrastructure/risk_metrics.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import numpy as np
from infrastructure.risk_metrics import StatisticalCalculator


class TestStatisticalCalculator(unittest.TestCase):
    """测试纯数学计算器"""
    
    def setUp(self):
        """测试前准备"""
        self.calculator = StatisticalCalculator()
        # 准备测试数据
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
        self.prices = np.cumprod(1 + self.returns) * 100
    
    def test_calculate_standard_deviation(self):
        """测试标准差计算"""
        # 全样本标准差
        std_full = self.calculator.calculate_standard_deviation(self.returns)
        expected = np.std(self.returns, ddof=1)
        self.assertAlmostEqual(std_full, expected, places=10)
        
        # 窗口标准差
        window = 20
        std_window = self.calculator.calculate_standard_deviation(self.returns, window)
        expected_window = np.std(self.returns[-window:], ddof=1)
        self.assertAlmostEqual(std_window, expected_window, places=10)
        
        # 边界情况：空数组
        std_empty = self.calculator.calculate_standard_deviation(np.array([]))
        self.assertEqual(std_empty, 0.0)
    
    def test_calculate_quantile(self):
        """测试分位数计算"""
        # 5%分位数
        q5 = self.calculator.calculate_quantile(self.returns, 0.05)
        expected = np.percentile(self.returns, 5)
        self.assertAlmostEqual(q5, expected, places=10)
        
        # 95%分位数
        q95 = self.calculator.calculate_quantile(self.returns, 0.95)
        expected = np.percentile(self.returns, 95)
        self.assertAlmostEqual(q95, expected, places=10)
        
        # 边界情况
        q_empty = self.calculator.calculate_quantile(np.array([]), 0.5)
        self.assertEqual(q_empty, 0.0)
    
    def test_calculate_cumulative_peak_deviation(self):
        """测试累积峰值偏离计算"""
        cumulative = np.array([1.0, 1.1, 1.05, 1.15, 1.12, 1.20, 1.18])
        deviation = self.calculator.calculate_cumulative_peak_deviation(cumulative)
        
        # 验证所有偏离值 <= 0
        self.assertTrue(np.all(deviation <= 0))
        
        # 验证第一个峰值处偏离为0
        self.assertEqual(deviation[0], 0.0)
        
        # 验证数组长度一致
        self.assertEqual(len(deviation), len(cumulative))
        
        # 边界情况
        dev_empty = self.calculator.calculate_cumulative_peak_deviation(np.array([]))
        self.assertEqual(len(dev_empty), 0)
    
    def test_calculate_mean_std_ratio(self):
        """测试均值/标准差比率"""
        ratio = self.calculator.calculate_mean_std_ratio(self.returns)
        expected = (np.mean(self.returns) - 0) / np.std(self.returns, ddof=1)
        self.assertAlmostEqual(ratio, expected, places=10)
        
        # 带基准值
        baseline = 0.0001
        ratio_baseline = self.calculator.calculate_mean_std_ratio(self.returns, baseline)
        expected_baseline = (np.mean(self.returns) - baseline) / np.std(self.returns, ddof=1)
        self.assertAlmostEqual(ratio_baseline, expected_baseline, places=10)
        
        # 边界情况：零标准差
        constant_values = np.array([1.0, 1.0, 1.0])
        ratio_const = self.calculator.calculate_mean_std_ratio(constant_values)
        self.assertEqual(ratio_const, 0.0)
    
    def test_calculate_covariance_matrix(self):
        """测试协方差矩阵"""
        data = np.column_stack([self.returns, self.returns * 1.5, self.returns * 0.8])
        cov_matrix = self.calculator.calculate_covariance_matrix(data)
        
        # 验证对称性
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))
        
        # 验证维度
        self.assertEqual(cov_matrix.shape, (3, 3))
        
        # 验证对角线为方差
        var0 = np.var(data[:, 0], ddof=1)
        self.assertAlmostEqual(cov_matrix[0, 0], var0, places=10)
        
        # 边界情况
        cov_empty = self.calculator.calculate_covariance_matrix(np.array([]))
        self.assertEqual(cov_empty.size, 0)
    
    def test_calculate_correlation_matrix(self):
        """测试相关系数矩阵"""
        data = np.column_stack([self.returns, self.returns * 1.5, np.random.randn(len(self.returns))])
        corr_matrix = self.calculator.calculate_correlation_matrix(data)
        
        # 验证对称性
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))
        
        # 验证对角线为1
        self.assertTrue(np.allclose(np.diag(corr_matrix), 1.0))
        
        # 验证范围 [-1, 1]
        self.assertTrue(np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0))
    
    def test_calculate_covariance_variance_ratio(self):
        """测试协方差/方差比率（β）"""
        market_returns = self.returns
        asset_returns = self.returns * 0.8 + np.random.normal(0, 0.01, len(self.returns))
        
        beta = self.calculator.calculate_covariance_variance_ratio(asset_returns, market_returns)
        
        # 验证β的合理范围
        self.assertTrue(-2.0 < beta < 2.0)
        
        # 验证与numpy计算一致
        cov = np.cov(asset_returns, market_returns)[0, 1]
        var = np.var(market_returns)
        expected_beta = cov / var
        self.assertAlmostEqual(beta, expected_beta, places=10)
        
        # 边界情况：长度不一致
        beta_invalid = self.calculator.calculate_covariance_variance_ratio(
            asset_returns[:100], market_returns
        )
        self.assertTrue(np.isnan(beta_invalid))
    
    def test_calculate_downside_deviation(self):
        """测试下行标准差"""
        downside_std = self.calculator.calculate_downside_deviation(self.returns)
        
        # 验证只使用负收益
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) > 0:
            expected = np.std(negative_returns, ddof=1)
            self.assertAlmostEqual(downside_std, expected, places=10)
        
        # 带基准值
        baseline = 0.001
        downside_std_baseline = self.calculator.calculate_downside_deviation(self.returns, baseline)
        below_baseline = self.returns[self.returns < baseline]
        if len(below_baseline) > 0:
            expected = np.std(below_baseline, ddof=1)
            self.assertAlmostEqual(downside_std_baseline, expected, places=10)
        
        # 边界情况：所有值都大于基准
        positive_values = np.array([1.0, 2.0, 3.0])
        downside_zero = self.calculator.calculate_downside_deviation(positive_values)
        self.assertEqual(downside_zero, 0.0)
    
    def test_calculate_residual(self):
        """测试残差计算"""
        market_returns = self.returns
        beta = 0.8
        asset_returns = market_returns * beta + np.random.normal(0, 0.005, len(market_returns))
        
        residuals = self.calculator.calculate_residual(asset_returns, market_returns, beta)
        
        # 验证残差长度
        self.assertEqual(len(residuals), len(asset_returns))
        
        # 验证残差公式
        expected = asset_returns - beta * market_returns
        np.testing.assert_array_almost_equal(residuals, expected, decimal=10)
        
        # 边界情况：长度不一致
        residuals_invalid = self.calculator.calculate_residual(
            asset_returns[:100], market_returns, beta
        )
        self.assertEqual(len(residuals_invalid), 0)


if __name__ == '__main__':
    unittest.main()
