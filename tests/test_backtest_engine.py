"""
回测引擎模块测试
"""
import sys
sys.path.insert(0, '.')

from core.backtest.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMetrics
)


def test_single_strategy_backtest():
    """测试单策略回测"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005
    )
    
    engine = BacktestEngine(config)
    
    # 简单的买入持有策略
    def buy_hold_strategy(data):
        if data.get('index', 0) == 0:  # 第一天买入
            return {
                'side': 'buy',
                'quantity': 100,
                'price': data.get('price', 100.0)
            }
        return None
    
    # 模拟市场数据（价格上涨10%）
    market_data = [
        {'index': 0, 'price': 100.0, 'timestamp': '2024-01-01'},
        {'index': 1, 'price': 102.0, 'timestamp': '2024-01-02'},
        {'index': 2, 'price': 105.0, 'timestamp': '2024-01-03'},
        {'index': 3, 'price': 108.0, 'timestamp': '2024-01-04'},
        {'index': 4, 'price': 110.0, 'timestamp': '2024-01-05'},
    ]
    
    metrics = engine.run(buy_hold_strategy, market_data)
    
    assert metrics is not None
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_trades >= 0
    print(f"✓ 单策略回测测试通过: 总收益={metrics.total_return:.2%}, 交易次数={metrics.total_trades}")


def test_portfolio_backtest():
    """测试组合回测"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        rebalance_frequency="monthly",
        transaction_cost=0.001
    )
    
    engine = BacktestEngine(config)
    
    # 固定权重策略（不调整）
    def fixed_weights_fn(current_weights, market_data):
        return current_weights  # 保持当前权重
    
    # 模拟历史数据
    historical_data = {
        '2024-01-01': {
            'prices': {'AAPL': 100.0, 'MSFT': 200.0, 'GOOGL': 150.0}
        },
        '2024-01-02': {
            'prices': {'AAPL': 102.0, 'MSFT': 205.0, 'GOOGL': 152.0}
        },
        '2024-01-03': {
            'prices': {'AAPL': 105.0, 'MSFT': 210.0, 'GOOGL': 155.0}
        },
        '2024-01-04': {
            'prices': {'AAPL': 108.0, 'MSFT': 215.0, 'GOOGL': 158.0}
        },
        '2024-01-05': {
            'prices': {'AAPL': 110.0, 'MSFT': 220.0, 'GOOGL': 160.0}
        }
    }
    
    metrics = engine.run_portfolio_backtest(
        fixed_weights_fn,
        historical_data,
        initial_weights={'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
    )
    
    assert metrics is not None
    assert metrics.total_return > 0  # 所有价格都上涨，应该盈利
    assert metrics.max_drawdown >= 0
    print(f"✓ 组合回测测试通过: 年化收益={metrics.annualized_return:.2%}, 夏普={metrics.sharpe_ratio:.3f}")


def test_portfolio_rebalancing():
    """测试组合再平衡"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_capital=100000.0,
        rebalance_frequency="weekly",  # 每周再平衡
        transaction_cost=0.001
    )
    
    engine = BacktestEngine(config)
    
    # 等权重再平衡策略
    def equal_weight_fn(current_weights, market_data):
        symbols = list(current_weights.keys())
        equal_w = 1.0 / len(symbols)
        return {sym: equal_w for sym in symbols}
    
    # 模拟数据（资产价格分化）
    historical_data = {}
    for i in range(25):  # 25个交易日
        date = f'2024-01-{i+1:02d}'
        # AAPL涨，MSFT跌，GOOGL平
        historical_data[date] = {
            'prices': {
                'AAPL': 100.0 + i * 2.0,  # 每天+2
                'MSFT': 200.0 - i * 1.0,  # 每天-1
                'GOOGL': 150.0
            }
        }
    
    metrics = engine.run_portfolio_backtest(
        equal_weight_fn,
        historical_data,
        initial_weights={'AAPL': 0.33, 'MSFT': 0.33, 'GOOGL': 0.34}
    )
    
    assert len(engine.rebalance_dates) > 0  # 应该有再平衡
    assert metrics.turnover > 0  # 应该有换手
    assert metrics.total_costs > 0  # 应该有成本
    print(f"✓ 组合再平衡测试通过: 再平衡次数={len(engine.rebalance_dates)}, 换手率={metrics.turnover:.2%}")


def test_metrics_calculation():
    """测试指标计算"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    engine = BacktestEngine(config)
    
    # 创建模拟权益曲线（先涨后跌）
    portfolio_values = [
        100000,  # 初始
        105000,  # +5%
        110000,  # +10%
        108000,  # -1.8%（峰值回撤开始）
        112000,  # +3.7%（新高）
        115000   # +2.7%
    ]
    
    dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06']
    
    metrics = engine._calculate_portfolio_metrics(portfolio_values, dates)
    
    assert metrics.total_return == 0.15  # (115000-100000)/100000
    assert metrics.max_drawdown > 0  # 应该有回撤
    assert metrics.sharpe_ratio != 0  # 应该有夏普比率
    print(f"✓ 指标计算测试通过: 总收益={metrics.total_return:.2%}, 回撤={metrics.max_drawdown:.2%}")


def test_period_returns():
    """测试期间收益计算"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    engine = BacktestEngine(config)
    
    prev_prices = {'AAPL': 100.0, 'MSFT': 200.0}
    curr_prices = {'AAPL': 105.0, 'MSFT': 210.0}
    
    returns = engine._calculate_period_returns(prev_prices, curr_prices)
    
    assert returns['AAPL'] == 0.05  # 5%
    assert returns['MSFT'] == 0.05  # 5%
    print("✓ 期间收益计算测试通过")


def test_rebalance_costs():
    """测试再平衡成本计算"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        transaction_cost=0.001
    )
    
    engine = BacktestEngine(config)
    
    current_weights = {'AAPL': 0.5, 'MSFT': 0.5}
    target_weights = {'AAPL': 0.6, 'MSFT': 0.4}
    portfolio_value = 100000.0
    
    turnover, trades, costs = engine._calculate_rebalance_costs(
        current_weights, target_weights, portfolio_value
    )
    
    # 换手率 = |0.6-0.5| + |0.4-0.5| = 0.2
    assert abs(turnover - 0.2) < 0.01
    # 成本 = 0.2 * 100000 * 0.001 = 20
    assert abs(costs - 20.0) < 0.01
    assert len(trades) == 2
    print(f"✓ 再平衡成本测试通过: 换手={turnover:.2%}, 成本=${costs:.2f}")


def test_sortino_ratio():
    """测试Sortino比率计算"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    engine = BacktestEngine(config)
    
    # 有正负收益的序列
    returns = [0.02, -0.01, 0.03, -0.02, 0.04, 0.01]
    
    sortino = engine._calculate_sortino_ratio(returns)
    
    assert sortino != 0
    assert not math.isnan(sortino)
    print(f"✓ Sortino比率测试通过: Sortino={sortino:.3f}")


def test_max_drawdown():
    """测试最大回撤计算"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    engine = BacktestEngine(config)
    
    # 模拟权益曲线：峰值110000，谷值90000
    values = [100000, 105000, 110000, 105000, 95000, 90000, 100000]
    
    max_dd = engine._calculate_max_drawdown_from_values(values)
    
    # 最大回撤 = (110000 - 90000) / 110000 ≈ 0.1818
    expected_dd = (110000 - 90000) / 110000
    assert abs(max_dd - expected_dd) < 0.01
    print(f"✓ 最大回撤测试通过: MDD={max_dd:.2%}")


def test_empty_data_handling():
    """测试空数据处理"""
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    engine = BacktestEngine(config)
    
    # 空权益曲线
    metrics = engine._calculate_portfolio_metrics([], [])
    assert metrics.total_return == 0.0
    assert metrics.sharpe_ratio == 0.0
    
    print("✓ 空数据处理测试通过")


if __name__ == "__main__":
    import math
    
    print("=== 回测引擎模块测试 ===\n")
    
    test_single_strategy_backtest()
    test_portfolio_backtest()
    test_portfolio_rebalancing()
    test_metrics_calculation()
    test_period_returns()
    test_rebalance_costs()
    test_sortino_ratio()
    test_max_drawdown()
    test_empty_data_handling()
    
    print("\n=== 全部10项测试通过！ ===")
