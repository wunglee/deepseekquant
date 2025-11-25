"""
历史回测框架使用演示
展示如何使用模拟数据进行压力测试回测验证

运行方式：
    python core_bak_refactored/examples/risk_backtest_demo.py
"""

import sys
from pathlib import Path

# 添加core_bak_refactored到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk.backtest_framework import (
    MockHistoricalDataProvider,
    SyntheticPortfolioBuilder,
    EventWindowBacktester,
    BacktestReporter
)


def main():
    """运行回测演示"""
    
    print("="*70)
    print("历史回测框架演示 - Phase 3A MVP（模拟数据版本）")
    print("="*70)
    
    # 1. 创建数据提供者（模拟数据）
    print("\n[步骤1] 创建数据提供者...")
    data_provider = MockHistoricalDataProvider()
    print("✓ 模拟数据提供者已创建")
    print("  注：真实数据集成需等待core_bak_refactored/core/data模块完成")
    
    # 2. 创建回测引擎
    print("\n[步骤2] 初始化回测引擎...")
    backtester = EventWindowBacktester(data_provider)
    print(f"✓ 回测引擎已创建，加载{len(backtester.events)}个历史事件")
    for event in backtester.events:
        print(f"  - {event.name} ({event.period[0]} ~ {event.period[1]})")
    
    # 3. 构造测试组合
    print("\n[步骤3] 构造合成组合...")
    portfolios = [
        SyntheticPortfolioBuilder.build_csi300_equal_weight(),
        SyntheticPortfolioBuilder.build_sector_rotation(),
        SyntheticPortfolioBuilder.build_ah_hybrid()
    ]
    print(f"✓ 已构造{len(portfolios)}个合成组合:")
    for p in portfolios:
        print(f"  - {p.name} (ID: {p.portfolio_id})")
    
    # 4. 运行回测（演示第一个组合）
    print("\n[步骤4] 运行回测验证...")
    print(f"演示组合: {portfolios[0].name}")
    print("-"*70)
    
    results = backtester.run_backtest(
        portfolios[0], 
        stress_tester=None  # Phase 3A使用简化版本
    )
    
    print(f"✓ 回测完成，共{len(results)}个结果")
    
    # 5. 生成报告
    print("\n[步骤5] 生成回测报告...")
    summary = BacktestReporter.generate_summary(results)
    BacktestReporter.print_summary(summary)
    
    # 6. 后续工作提示
    print("\n" + "="*70)
    print("Phase 3B 待办事项（等待数据模块）")
    print("="*70)
    print("1. 集成真实历史数据源（Yahoo Finance + JoinQuant）")
    print("2. 实现 RealHistoricalDataProvider")
    print("3. 与 StressTester 深度集成，使用真实压力测试预测")
    print("4. 验证预测误差≤20%目标")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
