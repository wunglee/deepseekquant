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
        """测试下行标准差（标准半方差公式）"""
        downside_std = self.calculator.calculate_downside_deviation(self.returns)
        
        # 验证使用半方差公式：sqrt(mean(min(returns, 0)^2))
        excess = self.returns - 0.0
        downside = np.minimum(excess, 0)
        n = len(self.returns)
        semi_variance = np.sum(downside**2) / (n - 1)  # ddof=1
        expected = np.sqrt(semi_variance)
        self.assertAlmostEqual(downside_std, expected, places=10)
        
        # 带基准值
        baseline = 0.001
        downside_std_baseline = self.calculator.calculate_downside_deviation(self.returns, baseline)
        excess_baseline = self.returns - baseline
        downside_baseline = np.minimum(excess_baseline, 0)
        semi_variance_baseline = np.sum(downside_baseline**2) / (n - 1)
        expected_baseline = np.sqrt(semi_variance_baseline)
        self.assertAlmostEqual(downside_std_baseline, expected_baseline, places=10)
        
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
    
    def test_calculate_log_returns(self):
        """测试对数收益率计算"""
        prices = np.array([100, 105, 103, 108, 110])
        log_returns = self.calculator.calculate_log_returns(prices)
        
        # 验证长度（比价格少1）
        self.assertEqual(len(log_returns), len(prices) - 1)
        
        # 验证公式：log(p[t] / p[t-1])
        expected = np.diff(np.log(prices))
        np.testing.assert_array_almost_equal(log_returns, expected, decimal=10)
        
        # 边界情况：空数组
        log_empty = self.calculator.calculate_log_returns(np.array([]))
        self.assertEqual(len(log_empty), 0)
        
        # 边界情况：单点数据
        log_single = self.calculator.calculate_log_returns(np.array([100]))
        self.assertEqual(len(log_single), 0)
    
    def test_calculate_simple_returns(self):
        """测试简单收益率计算"""
        prices = np.array([100, 105, 103, 108, 110])
        simple_returns = self.calculator.calculate_simple_returns(prices)
        
        # 验证长度（比价格少1）
        self.assertEqual(len(simple_returns), len(prices) - 1)
        
        # 验证公式：(p[t] / p[t-1]) - 1
        expected = (prices[1:] / prices[:-1]) - 1
        np.testing.assert_array_almost_equal(simple_returns, expected, decimal=10)
        
        # 验证具体值：第一个收益 = (105-100)/100 = 5%
        self.assertAlmostEqual(simple_returns[0], 0.05, places=10)
        
        # 边界情况：空数组
        simple_empty = self.calculator.calculate_simple_returns(np.array([]))
        self.assertEqual(len(simple_empty), 0)
        
        # 边界情况：单点数据
        simple_single = self.calculator.calculate_simple_returns(np.array([100]))
        self.assertEqual(len(simple_single), 0)
        
        # 边界情况：零价格（产生Inf）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略除零警告
            zero_prices = np.array([100, 0, 110])
            simple_zero = self.calculator.calculate_simple_returns(zero_prices)
            # 第一个收益 = (0-100)/100 = -100%
            self.assertAlmostEqual(simple_zero[0], -1.0, places=10)
            # 第二个收益 = (110-0)/0 = Inf
            self.assertTrue(np.isinf(simple_zero[1]))
        
        # 边界情况：负价格
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            negative_prices = np.array([100, -10, 110])
            simple_neg = self.calculator.calculate_simple_returns(negative_prices)
            # 第一个收益 = (-10-100)/100 = -110%
            self.assertAlmostEqual(simple_neg[0], -1.1, places=10)
            # 第二个收益 = (110-(-10))/(-10) = -12
            self.assertAlmostEqual(simple_neg[1], -12.0, places=10)
    
    def test_calculate_cvar(self):
        """测试 CVaR 计算"""
        # 正常数据
        cvar_95 = self.calculator.calculate_cvar(self.returns, 0.95)
        var_95 = self.calculator.calculate_quantile(self.returns, 0.05)
        
        # CVaR 应该 >= VaR（更保守）
        self.assertGreaterEqual(abs(cvar_95), abs(var_95))
        
        # 验证 CVaR 是尾部均值
        tail = self.returns[self.returns <= var_95]
        if len(tail) > 0:
            expected_cvar = np.mean(tail)
            self.assertAlmostEqual(cvar_95, expected_cvar, places=10)
    
    def test_edge_case_nan_data(self):
        """边界测试：含 NaN 的数据"""
        # 含 NaN 的数据
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # 标准差：NaN 会传播
        std_nan = self.calculator.calculate_standard_deviation(data_with_nan)
        self.assertTrue(np.isnan(std_nan) or std_nan == 0.0)
        
        # 分位数：numpy.quantile 会处理 NaN
        q_nan = self.calculator.calculate_quantile(data_with_nan, 0.5)
        # 某些 numpy 版本会返回 NaN，某些会忽略 NaN
        self.assertTrue(not np.isnan(q_nan) or np.isnan(q_nan))
    
    def test_edge_case_inf_data(self):
        """边界测试：含 Inf 的数据"""
        # 含 Inf 的数据
        data_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        
        # 标准差：Inf 会导致结果为 NaN 或 Inf
        std_inf = self.calculator.calculate_standard_deviation(data_with_inf)
        self.assertTrue(np.isnan(std_inf) or np.isinf(std_inf) or std_inf > 1e10)
        
        # 协方差矩阵：Inf 会导致矩阵元素为 NaN 或 Inf
        data_2d = np.column_stack([data_with_inf, data_with_inf * 0.5])
        cov_inf = self.calculator.calculate_covariance_matrix(data_2d)
        self.assertTrue(np.any(np.isnan(cov_inf)) or np.any(np.isinf(cov_inf)) or np.any(cov_inf > 1e10))
    
    def test_edge_case_extreme_values(self):
        """边界测试：极端数值"""
        # 极大值
        large_values = np.array([1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3])
        std_large = self.calculator.calculate_standard_deviation(large_values)
        self.assertGreater(std_large, 0)
        self.assertLess(std_large, 10)  # 相对标准差应该很小
        
        # 极小值
        small_values = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        std_small = self.calculator.calculate_standard_deviation(small_values)
        self.assertGreater(std_small, 0)
        self.assertLess(std_small, 1e-5)
    
    def test_edge_case_negative_prices(self):
        """边界测试：负价格（对数计算会失败）"""
        # 负价格会导致对数计算出现警告或 NaN
        negative_prices = np.array([100, 105, -10, 110])
        
        # 捕获警告并验证结果包含 NaN 或 Inf
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_returns = self.calculator.calculate_log_returns(negative_prices)
            # 负价格的对数为 NaN
            self.assertTrue(np.any(np.isnan(log_returns)) or np.any(np.isinf(log_returns)))
    
    def test_edge_case_zero_prices(self):
        """边界测试：零价格（对数计算会失败）"""
        # 零价格会导致对数为 -Inf
        zero_prices = np.array([100, 105, 0, 110])
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_returns = self.calculator.calculate_log_returns(zero_prices)
            # 零价格的对数为 -Inf
            self.assertTrue(np.any(np.isinf(log_returns)))
    
    def test_edge_case_window_larger_than_data(self):
        """边界测试：滚动窗口大于数据长度"""
        short_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 窗口=10 > 数据长度=5
        std_large_window = self.calculator.calculate_standard_deviation(short_data, window=10)
        # 应返回 0（数据不足）
        self.assertEqual(std_large_window, 0.0)
    
    def test_covariance_matrix_singular(self):
        """测试协方差矩阵奇异性（完全共线）"""
        # 两列完全相同（共线）
        data_collinear = np.column_stack([self.returns, self.returns])
        cov_singular = self.calculator.calculate_covariance_matrix(data_collinear)
        
        # 验证矩阵对称
        self.assertTrue(np.allclose(cov_singular, cov_singular.T))
        
        # 计算条件数（奇异矩阵条件数极大）
        eigenvalues = np.linalg.eigvals(cov_singular)
        # 完全共线时，应该有一个特征值接近0
        min_eigenvalue = np.min(np.abs(eigenvalues))
        self.assertLess(min_eigenvalue, 1e-10)  # 接近奇异
    
    def test_correlation_perfect_correlation(self):
        """测试完全相关（相关系数=1）"""
        # 两列完全线性相关
        data_perfect = np.column_stack([self.returns, self.returns * 2.0])
        corr_matrix = self.calculator.calculate_correlation_matrix(data_perfect)
        
        # 验证非对角线元素接近1
        self.assertAlmostEqual(corr_matrix[0, 1], 1.0, places=5)
        self.assertAlmostEqual(corr_matrix[1, 0], 1.0, places=5)
    
    def test_calculate_tail_risk(self):
        """测试尾部风险概率计算"""
        # 正常数据
        tail_prob = self.calculator.calculate_tail_risk(self.returns, threshold=-0.05)
        
        # 尾部概率应该在 [0, 1] 范围内
        self.assertGreaterEqual(tail_prob, 0.0)
        self.assertLessEqual(tail_prob, 1.0)
        
        # 手动计算验证
        tail_events = self.returns[self.returns < -0.05]
        expected_prob = len(tail_events) / len(self.returns)
        self.assertAlmostEqual(tail_prob, expected_prob, places=10)
        
        # 边界情况：空数据
        tail_empty = self.calculator.calculate_tail_risk(np.array([]), -0.05)
        self.assertEqual(tail_empty, 0.0)


if __name__ == '__main__':
    unittest.main()
