"""
增量计算模块测试
"""

import unittest
import numpy as np
from core_bak_refactored.core.risk.incremental_calculator import (
    IncrementalCovarianceCalculator,
    IncrementalVaRCalculator,
    IncrementalBoundary,
    compare_incremental_vs_full
)


class TestIncrementalCovarianceCalculator(unittest.TestCase):
    """测试协方差增量计算器"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.calculator = IncrementalCovarianceCalculator()
        
        # 生成模拟数据
        self.n_assets = 50
        self.T = 252
        self.returns = np.random.randn(self.T, self.n_assets) * 0.01
        self.cov_matrix = np.cov(self.returns.T)
        
    def test_initialization(self):
        """测试初始化"""
        calc = IncrementalCovarianceCalculator()
        self.assertIsNotNone(calc.boundary)
        self.assertEqual(calc.consecutive_updates, 0)
        self.assertEqual(calc.cumulative_error, 0.0)
    
    def test_can_use_incremental_positive(self):
        """测试增量计算可行性判断 - 正面案例"""
        old_assets = [f"ASSET_{i}" for i in range(50)]
        new_assets = old_assets.copy()  # 无资产变化
        
        can_use, reason = self.calculator.can_use_incremental(
            old_assets=old_assets,
            new_assets=new_assets,
            weights_changed_ratio=0.10  # 10%权重变化
        )
        
        self.assertTrue(can_use)
        self.assertIn("满足", reason)
    
    def test_can_use_incremental_too_many_changes(self):
        """测试增量计算可行性判断 - 变化过多"""
        old_assets = [f"ASSET_{i}" for i in range(50)]
        new_assets = old_assets.copy()
        
        can_use, reason = self.calculator.can_use_incremental(
            old_assets=old_assets,
            new_assets=new_assets,
            weights_changed_ratio=0.25  # 25%权重变化，超过20%阈值
        )
        
        self.assertFalse(can_use)
        self.assertIn("权重变化", reason)
    
    def test_can_use_incremental_asset_removed(self):
        """测试增量计算可行性判断 - 资产移除"""
        old_assets = [f"ASSET_{i}" for i in range(50)]
        new_assets = old_assets[:-5]  # 移除5个资产
        
        can_use, reason = self.calculator.can_use_incremental(
            old_assets=old_assets,
            new_assets=new_assets,
            weights_changed_ratio=0.10
        )
        
        self.assertFalse(can_use)
        self.assertIn("移除资产", reason)
    
    def test_incremental_update_add_new_data(self):
        """测试增量更新 - 仅添加新数据"""
        new_return = np.random.randn(self.n_assets) * 0.01
        
        updated_cov, metadata = self.calculator.incremental_update(
            current_cov=self.cov_matrix,
            current_returns=self.returns,
            new_return=new_return,
            old_return=None  # 不移除旧数据
        )
        
        # 验证形状
        self.assertEqual(updated_cov.shape, self.cov_matrix.shape)
        
        # 验证元数据
        self.assertIn('computation_time_ms', metadata)
        self.assertIn('incremental_error', metadata)
        self.assertIn('consecutive_updates', metadata)
        
        # 验证误差在合理范围内（单次更新可能较高，累积才重要）
        self.assertLess(metadata['incremental_error'], 0.05)
    
    def test_incremental_update_sliding_window(self):
        """测试增量更新 - 滑动窗口"""
        new_return = np.random.randn(self.n_assets) * 0.01
        old_return = self.returns[0]  # 移除最旧的数据
        
        updated_cov, metadata = self.calculator.incremental_update(
            current_cov=self.cov_matrix,
            current_returns=self.returns,
            new_return=new_return,
            old_return=old_return
        )
        
        # 验证形状
        self.assertEqual(updated_cov.shape, self.cov_matrix.shape)
        
        # 验证方法标识
        self.assertEqual(metadata['method'], 'sliding_window')
        
        # 验证误差
        self.assertLess(metadata['incremental_error'], 0.05)
    
    def test_incremental_vs_full_accuracy(self):
        """测试增量计算精度 - 与全量计算对比"""
        # 准备新数据
        new_return = np.random.randn(self.n_assets) * 0.01
        
        # 增量计算
        updated_cov_incremental, _ = self.calculator.incremental_update(
            current_cov=self.cov_matrix,
            current_returns=self.returns,
            new_return=new_return,
            old_return=None
        )
        
        # 全量计算
        extended_returns = np.vstack([self.returns, new_return])
        updated_cov_full = np.cov(extended_returns.T)
        
        # 对比精度
        comparison = compare_incremental_vs_full(
            updated_cov_incremental,
            updated_cov_full
        )
        
        # 验证误差在可接受范围内
        self.assertLess(comparison['frobenius_relative_error'], 0.01)
        self.assertTrue(comparison['is_acceptable'])
    
    def test_consecutive_updates_limit(self):
        """测试连续更新次数限制"""
        # 模拟连续50次更新
        for i in range(50):
            new_return = np.random.randn(self.n_assets) * 0.01
            self.calculator.incremental_update(
                current_cov=self.cov_matrix,
                current_returns=self.returns,
                new_return=new_return
            )
        
        # 第51次应该被边界检查拒绝
        old_assets = [f"ASSET_{i}" for i in range(self.n_assets)]
        can_use, reason = self.calculator.can_use_incremental(
            old_assets=old_assets,
            new_assets=old_assets,
            weights_changed_ratio=0.10
        )
        
        self.assertFalse(can_use)
        self.assertIn("连续增量更新超过", reason)
    
    def test_reset_state(self):
        """测试状态重置"""
        # 执行一些更新
        for i in range(5):
            new_return = np.random.randn(self.n_assets) * 0.01
            self.calculator.incremental_update(
                current_cov=self.cov_matrix,
                current_returns=self.returns,
                new_return=new_return
            )
        
        # 重置
        self.calculator.reset()
        
        # 验证状态已重置
        self.assertEqual(self.calculator.consecutive_updates, 0)
        self.assertEqual(self.calculator.cumulative_error, 0.0)
        self.assertIsNotNone(self.calculator.last_full_calculation_time)


class TestIncrementalVaRCalculator(unittest.TestCase):
    """测试VaR增量计算器"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.calculator = IncrementalVaRCalculator(confidence_level=0.95)
        
        # 生成模拟数据
        self.n_assets = 50
        self.T = 252
        self.returns = np.random.randn(self.T, self.n_assets) * 0.01
        self.cov_matrix = np.cov(self.returns.T)
        self.weights = np.ones(self.n_assets) / self.n_assets
        
    def test_initialization(self):
        """测试初始化"""
        calc = IncrementalVaRCalculator(confidence_level=0.99)
        self.assertEqual(calc.confidence_level, 0.99)
        self.assertIsNone(calc.base_returns)
        self.assertIsNone(calc.base_var)
    
    def test_update_var_on_weight_change(self):
        """测试权重变化后VaR更新"""
        # 原权重
        old_weights = self.weights.copy()
        
        # 新权重（微调）
        new_weights = old_weights.copy()
        new_weights[:10] *= 1.1  # 前10个资产权重增加10%
        new_weights /= new_weights.sum()  # 归一化
        
        # 计算组合收益率
        portfolio_returns = self.returns @ old_weights
        
        # 更新VaR
        new_var, metadata = self.calculator.update_var_on_weight_change(
            base_portfolio_returns=portfolio_returns,
            base_weights=old_weights,
            new_weights=new_weights,
            cov_matrix=self.cov_matrix
        )
        
        # 验证结果
        self.assertIsInstance(new_var, (float, np.floating))
        self.assertGreater(new_var, 0)
        
        # 验证元数据
        self.assertIn('method', metadata)
        self.assertEqual(metadata['method'], 'parametric_incremental')
        self.assertIn('computation_time_ms', metadata)
        self.assertIn('portfolio_volatility', metadata)
    
    def test_update_var_on_data_change(self):
        """测试新增数据点后VaR更新"""
        # 基准收益率
        base_returns = np.random.randn(252) * 0.01
        
        # 新收益率
        new_return = 0.015
        
        # 更新VaR
        new_var, metadata = self.calculator.update_var_on_data_change(
            base_returns=base_returns,
            new_return=new_return,
            window_size=252
        )
        
        # 验证结果
        self.assertIsInstance(new_var, (float, np.floating))
        self.assertGreater(new_var, 0)
        
        # 验证元数据
        self.assertEqual(metadata['method'], 'historical_simulation_incremental')
        self.assertIn('sample_size', metadata)
        self.assertIn('computation_time_ms', metadata)


class TestComparisonFunction(unittest.TestCase):
    """测试对比函数"""
    
    def test_compare_identical_matrices(self):
        """测试相同矩阵对比"""
        matrix = np.random.randn(50, 50)
        matrix = matrix @ matrix.T  # 确保对称正定
        
        comparison = compare_incremental_vs_full(matrix, matrix)
        
        # 误差应该接近0
        self.assertLess(comparison['frobenius_relative_error'], 1e-10)
        self.assertLess(comparison['max_absolute_error'], 1e-10)
        self.assertTrue(comparison['is_acceptable'])
    
    def test_compare_different_matrices(self):
        """测试不同矩阵对比"""
        matrix1 = np.random.randn(50, 50)
        matrix1 = matrix1 @ matrix1.T
        
        matrix2 = matrix1 + np.random.randn(50, 50) * 0.001
        
        comparison = compare_incremental_vs_full(matrix1, matrix2)
        
        # 应该有一定误差
        self.assertGreater(comparison['frobenius_relative_error'], 0)
        self.assertIn('max_relative_error', comparison)
        self.assertIn('diagonal_mean_error', comparison)


if __name__ == '__main__':
    unittest.main()
