"""
StatisticalCalculator 性能基准测试
对应文件: infrastructure/risk_metrics.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import time
import numpy as np
from infrastructure.risk_metrics import StatisticalCalculator


class PerformanceBenchmarkTest(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """准备测试数据"""
        self.calculator = StatisticalCalculator()
        np.random.seed(42)
        
        # 准备不同规模的数据
        self.data_1000 = np.random.normal(0.001, 0.02, 1000)  # 1000点日线（约4年）
        self.data_5000 = np.random.normal(0.001, 0.02, 5000)  # 5000点（约20年）
        
        # 准备多资产数据
        self.returns_100_assets = np.random.normal(0.001, 0.02, (252, 100))  # 100资产×252天
        self.returns_500_assets = np.random.normal(0.001, 0.02, (252, 500))  # 500资产×252天
    
    def test_benchmark_standard_deviation_1000_points(self):
        """基准测试：1000点标准差计算"""
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_standard_deviation(self.data_1000)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"\n标准差计算（1000点）：平均 {avg_time_ms:.3f}ms")
        
        # 性能目标：< 1ms
        self.assertLess(avg_time_ms, 1.0, "标准差计算应该在1ms内完成")
    
    def test_benchmark_log_returns_1000_points(self):
        """基准测试：1000点对数收益率计算"""
        prices = np.cumprod(1 + self.data_1000) * 100
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_log_returns(prices)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"对数收益率计算（1000点）：平均 {avg_time_ms:.3f}ms")
        
        # 性能目标：< 0.5ms
        self.assertLess(avg_time_ms, 0.5, "对数收益率计算应该在0.5ms内完成")
    
    def test_benchmark_covariance_matrix_100_assets(self):
        """基准测试：100资产协方差矩阵（100×252）"""
        iterations = 100
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_covariance_matrix(self.returns_100_assets)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"协方差矩阵（100资产×252天）：平均 {avg_time_ms:.1f}ms")
        
        # 性能目标：< 50ms
        self.assertLess(avg_time_ms, 50.0, "100资产协方差矩阵应该在50ms内完成")
    
    def test_benchmark_covariance_matrix_500_assets(self):
        """基准测试：500资产协方差矩阵（500×252）"""
        iterations = 20
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_covariance_matrix(self.returns_500_assets)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"协方差矩阵（500资产×252天）：平均 {avg_time_ms:.1f}ms")
        
        # 性能目标：< 500ms（500×500矩阵）
        self.assertLess(avg_time_ms, 500.0, "500资产协方差矩阵应该在500ms内完成")
    
    def test_benchmark_correlation_matrix_100_assets(self):
        """基准测试：100资产相关性矩阵"""
        iterations = 100
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_correlation_matrix(self.returns_100_assets)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"相关性矩阵（100资产×252天）：平均 {avg_time_ms:.1f}ms")
        
        # 性能目标：< 50ms
        self.assertLess(avg_time_ms, 50.0, "100资产相关性矩阵应该在50ms内完成")
    
    def test_benchmark_quantile_calculation(self):
        """基准测试：分位数计算"""
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_quantile(self.data_1000, 0.05)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"分位数计算（1000点）：平均 {avg_time_ms:.3f}ms")
        
        # 性能目标：< 0.5ms
        self.assertLess(avg_time_ms, 0.5, "分位数计算应该在0.5ms内完成")
    
    def test_benchmark_cvar_calculation(self):
        """基准测试：CVaR计算"""
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_cvar(self.data_1000, 0.95)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"CVaR计算（1000点）：平均 {avg_time_ms:.3f}ms")
        
        # 性能目标：< 1ms
        self.assertLess(avg_time_ms, 1.0, "CVaR计算应该在1ms内完成")
    
    def test_benchmark_downside_deviation(self):
        """基准测试：下行标准差计算"""
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            self.calculator.calculate_downside_deviation(self.data_1000)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        print(f"下行标准差计算（1000点）：平均 {avg_time_ms:.3f}ms")
        
        # 性能目标：< 1ms
        self.assertLess(avg_time_ms, 1.0, "下行标准差计算应该在1ms内完成")
    
    def test_memory_covariance_matrix_1000_assets(self):
        """内存测试：1000资产协方差矩阵"""
        # 1000×1000 float64 矩阵
        returns_1000 = np.random.normal(0.001, 0.02, (252, 1000))
        cov_matrix = self.calculator.calculate_covariance_matrix(returns_1000)
        
        # 计算内存占用（字节）
        memory_bytes = cov_matrix.nbytes
        memory_mb = memory_bytes / (1024 * 1024)
        
        print(f"\n1000资产协方差矩阵内存占用：{memory_mb:.2f}MB")
        
        # 预期：1000×1000×8字节 = 8MB
        self.assertLess(memory_mb, 10.0, "1000资产协方差矩阵内存应该<10MB")
    
    def test_vectorization_efficiency(self):
        """测试向量化效率：numpy vs 循环"""
        data = self.returns_100_assets
        
        # Numpy 向量化（当前实现）
        start_numpy = time.time()
        for _ in range(100):
            cov_numpy = self.calculator.calculate_covariance_matrix(data)
        time_numpy = time.time() - start_numpy
        
        # Python 循环实现（低效版本，仅对比）
        def cov_matrix_loop(data):
            n_vars = data.shape[1]
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    cov[i, j] = np.cov(data[:, i], data[:, j])[0, 1]
            return cov
        
        start_loop = time.time()
        for _ in range(10):  # 循环版本太慢，只跑10次
            cov_loop = cov_matrix_loop(data)
        time_loop = (time.time() - start_loop) * 10  # 归一化到100次
        
        speedup = time_loop / time_numpy
        print(f"\n向量化加速比：{speedup:.1f}x（numpy vs 循环）")
        
        # 向量化应该至少快10倍
        self.assertGreater(speedup, 10.0, "numpy向量化应该比循环快至少10倍")


if __name__ == '__main__':
    # 运行性能测试并显示结果
    suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmarkTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
