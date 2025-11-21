"""
并行计算性能基准测试
用于验证并行计算的实际加速效果
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from core_bak_refactored.core.risk.portfolio_risk import PortfolioRiskAnalyzer
from core_bak_refactored.core.share.market_config import MarketConfigManager


class MockPortfolioState:
    """模拟组合状态"""
    def __init__(self, allocations):
        self.allocations = allocations


class MockAllocation:
    """模拟资产配置"""
    def __init__(self, weight):
        self.weight = weight


def generate_mock_portfolio(n_assets: int = 100) -> tuple:
    """生成模拟组合数据"""
    np.random.seed(42)
    
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
        (datetime.now() - timedelta(days=n_days-i)).isoformat()
        for i in range(n_days)
    ]
    
    market_data = {
        'prices': prices,
        'timestamp': timestamps
    }
    
    return portfolio_state, market_data


def benchmark_parallel_vs_serial():
    """基准测试：并行 vs 串行"""
    print("=" * 80)
    print("并行计算性能基准测试")
    print("=" * 80)
    
    # 准备配置
    config_manager = MarketConfigManager()
    config = config_manager.generate_config_template('CN')
    
    # 创建分析器
    analyzer_parallel = PortfolioRiskAnalyzer(config, enable_parallel=True)
    analyzer_serial = PortfolioRiskAnalyzer(config, enable_parallel=False)
    
    # 测试用例
    test_cases = [
        (10, 50),   # 10个组合，每个50资产
        (20, 50),   # 20个组合，每个50资产
        (50, 100),  # 50个组合，每个100资产
    ]
    
    results = []
    
    for n_portfolios, n_assets in test_cases:
        print(f"\n测试用例: {n_portfolios}个组合 x {n_assets}资产")
        print("-" * 80)
        
        # 生成测试数据
        portfolios = []
        for i in range(n_portfolios):
            portfolio_state, market_data = generate_mock_portfolio(n_assets)
            portfolios.append((f"portfolio_{i}", portfolio_state, market_data))
        
        # 串行测试
        print(f"串行计算...")
        start = time.time()
        results_serial = analyzer_serial.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=False
        )
        time_serial = time.time() - start
        print(f"  耗时: {time_serial:.3f}秒")
        
        # 并行测试
        print(f"并行计算...")
        start = time.time()
        results_parallel = analyzer_parallel.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=True
        )
        time_parallel = time.time() - start
        print(f"  耗时: {time_parallel:.3f}秒")
        
        # 计算加速比
        speedup = time_serial / time_parallel if time_parallel > 0 else 0
        efficiency = speedup / 8  # 假设8核CPU
        
        print(f"\n性能指标:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  并行效率: {efficiency*100:.1f}%")
        print(f"  时间节省: {(time_serial - time_parallel):.3f}秒 ({(1-time_parallel/time_serial)*100:.1f}%)")
        
        # 验证结果一致性
        consistent = True
        for pid in results_serial.keys():
            vol_serial = results_serial[pid].get('volatility', 0)
            vol_parallel = results_parallel[pid].get('volatility', 0)
            if abs(vol_serial - vol_parallel) > 0.0001:
                consistent = False
                break
        
        print(f"  结果一致性: {'✓ 通过' if consistent else '✗ 失败'}")
        
        results.append({
            'n_portfolios': n_portfolios,
            'n_assets': n_assets,
            'time_serial': time_serial,
            'time_parallel': time_parallel,
            'speedup': speedup,
            'efficiency': efficiency,
            'consistent': consistent
        })
    
    # 总结
    print("\n" + "=" * 80)
    print("性能基准测试总结")
    print("=" * 80)
    print(f"{'组合数':<10} {'资产数':<10} {'串行(s)':<12} {'并行(s)':<12} {'加速比':<10} {'效率':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['n_portfolios']:<10} {r['n_assets']:<10} "
              f"{r['time_serial']:<12.3f} {r['time_parallel']:<12.3f} "
              f"{r['speedup']:<10.2f}x {r['efficiency']*100:<10.1f}%")
    
    # 计算平均加速比
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_efficiency = np.mean([r['efficiency'] for r in results])
    
    print("-" * 80)
    print(f"平均加速比: {avg_speedup:.2f}x")
    print(f"平均并行效率: {avg_efficiency*100:.1f}%")
    print(f"所有测试结果一致性: {'✓ 全部通过' if all(r['consistent'] for r in results) else '✗ 有失败'}")
    
    return results


def benchmark_optimization_impact():
    """基准测试：优化组件的整体影响"""
    print("\n" + "=" * 80)
    print("优化组件影响评估")
    print("=" * 80)
    
    config_manager = MarketConfigManager()
    config = config_manager.generate_config_template('CN')
    
    # 创建不同配置的分析器
    analyzers = {
        '无优化': PortfolioRiskAnalyzer(config, enable_parallel=False, enable_incremental=False),
        '仅并行': PortfolioRiskAnalyzer(config, enable_parallel=True, enable_incremental=False),
        '仅增量': PortfolioRiskAnalyzer(config, enable_parallel=False, enable_incremental=True),
        '全优化': PortfolioRiskAnalyzer(config, enable_parallel=True, enable_incremental=True),
    }
    
    # 生成测试数据（30个组合，50资产）
    n_portfolios, n_assets = 30, 50
    portfolios = []
    for i in range(n_portfolios):
        portfolio_state, market_data = generate_mock_portfolio(n_assets)
        portfolios.append((f"portfolio_{i}", portfolio_state, market_data))
    
    print(f"\n测试配置: {n_portfolios}个组合 x {n_assets}资产")
    print("-" * 80)
    
    results = {}
    baseline_time = None
    
    for name, analyzer in analyzers.items():
        print(f"\n{name}:")
        start = time.time()
        calc_results = analyzer.batch_calculate_portfolio_risk(
            portfolios,
            use_parallel=None  # 使用analyzer的默认配置
        )
        elapsed = time.time() - start
        results[name] = elapsed
        
        if baseline_time is None:
            baseline_time = elapsed
        
        improvement = (baseline_time - elapsed) / baseline_time * 100
        print(f"  耗时: {elapsed:.3f}秒")
        print(f"  vs基准: {improvement:+.1f}%")
        print(f"  完成率: {len(calc_results)}/{n_portfolios}")
    
    # 总结
    print("\n" + "-" * 80)
    print("优化效果对比:")
    for name, elapsed in results.items():
        improvement = (baseline_time - elapsed) / baseline_time * 100
        speedup = baseline_time / elapsed
        print(f"  {name:<10}: {elapsed:6.3f}秒 ({improvement:+5.1f}%) [{speedup:.2f}x]")
    
    return results


if __name__ == '__main__':
    print("DeepSeekQuant - 并行计算性能基准测试\n")
    
    # 运行并行vs串行基准测试
    parallel_results = benchmark_parallel_vs_serial()
    
    # 运行优化组件影响评估
    optimization_results = benchmark_optimization_impact()
    
    print("\n" + "=" * 80)
    print("基准测试完成！")
    print("=" * 80)
