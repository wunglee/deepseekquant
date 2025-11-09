"""
简单量化策略示例 - 使用核心处理器流水线
"""
from infrastructure.interfaces import InfrastructureProvider
from core.signal.signal_processor import SignalProcessor
from core.risk.risk_processor import RiskProcessor
from core.exec.exec_processor import ExecProcessor
from core.portfolio.portfolio_processor import PortfolioProcessor
from core.data.data_fetcher import DataFetcher
from core.backtest.backtest_engine import BacktestEngine, BacktestConfig
from common import TradeDirection, OrderType, AllocationMethod

def simple_strategy_example():
    """
    简单策略示例：双均线信号生成 + 风险评估 + 组合分配 + 订单执行
    """
    print("=" * 60)
    print("DeepSeekQuant 简单策略示例")
    print("=" * 60)

    # 1. 初始化编排器
    orchestrator = InfrastructureProvider.get('orchestrator')

    # 2. 创建处理器
    signal_proc = SignalProcessor(processor_name='SignalProcessor')
    risk_proc = RiskProcessor(processor_name='RiskProcessor')
    exec_proc = ExecProcessor(processor_name='ExecProcessor')
    portfolio_proc = PortfolioProcessor(processor_name='PortfolioProcessor')

    # 3. 注册处理器
    orchestrator.register_processor(signal_proc)
    orchestrator.register_processor(risk_proc)
    orchestrator.register_processor(exec_proc)
    orchestrator.register_processor(portfolio_proc)

    # 4. 初始化所有处理器
    orchestrator.initialize_all()
    print("\n处理器初始化完成\n")

    # 5. 数据抓取
    fetcher = DataFetcher()
    symbols = ['AAPL', 'GOOG', 'MSFT']
    market_data = {}
    for symbol in symbols:
        md = fetcher.get_market_data(symbol, use_cache=False)
        if md.get('status') == 'success':
            market_data[symbol] = md.get('data')
    
    print(f"抓取市场数据: {len(market_data)} 个标的\n")

    # 6. 信号生成
    signals = []
    for symbol, data in market_data.items():
        sig_result = signal_proc.process(symbol=symbol, price=data.get('price', 0))
        if sig_result.get('status') == 'success':
            signals.append(sig_result.get('signal'))
            print(f"信号生成: {symbol} @ {data.get('price', 0)}")
    
    print(f"\n生成 {len(signals)} 个信号\n")

    # 7. 风险评估
    for signal in signals:
        risk_result = risk_proc.process(signal={'symbol': signal['symbol'], 'price': signal['price']})
        if risk_result.get('status') == 'success':
            assessment = risk_result.get('assessment')
            print(f"风险评估: {signal['symbol']} - {assessment.get('reason')}")
    
    print()

    # 8. 组合分配
    positions = [
        {'symbol': 'AAPL', 'quantity': 10, 'price': market_data.get('AAPL', {}).get('price', 150)},
        {'symbol': 'GOOG', 'quantity': 5, 'price': market_data.get('GOOG', {}).get('price', 2800)},
        {'symbol': 'MSFT', 'quantity': 8, 'price': market_data.get('MSFT', {}).get('price', 380)}
    ]
    
    port_result = portfolio_proc.process(positions=positions, method=AllocationMethod.EQUAL_WEIGHT)
    if port_result.get('status') == 'success':
        weights = port_result.get('weights')
        print(f"组合权重: {weights}\n")

    # 9. 订单执行
    for symbol in symbols[:2]:  # 仅执行前两个标的
        order_data = {
            'symbol': symbol,
            'quantity': 10,
            'side': TradeDirection.LONG,
            'order_type': OrderType.MARKET,
            'price': market_data.get(symbol, {}).get('price', 100)
        }
        exec_result = exec_proc.process(order=order_data)
        if exec_result.get('status') == 'success':
            print(f"订单执行: {symbol} x10 @ {order_data['price']}")
    
    print()

    # 10. 清理
    orchestrator.cleanup_processors()
    print("\n" + "=" * 60)
    print("策略示例完成")
    print("=" * 60)


def backtest_strategy_example():
    """
    回测示例：测试简单策略表现
    """
    print("\n" + "=" * 60)
    print("回测示例")
    print("=" * 60)

    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005
    )

    engine = BacktestEngine(config)

    # 模拟市场数据
    market_data = [
        {'timestamp': '2024-01-01', 'price': 100},
        {'timestamp': '2024-01-02', 'price': 102},
        {'timestamp': '2024-01-03', 'price': 105},
        {'timestamp': '2024-01-04', 'price': 103},
        {'timestamp': '2024-01-05', 'price': 107}
    ]

    # 简单策略：价格上涨时买入
    def simple_strategy(data):
        if data.get('price', 0) > 101:
            return {'side': 'buy', 'quantity': 1}
        return None

    metrics = engine.run(simple_strategy, market_data)

    print(f"\n回测结果:")
    print(f"  总收益率: {metrics.total_return:.2%}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"  最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  胜率: {metrics.win_rate:.2%}")
    print(f"  总交易次数: {metrics.total_trades}")
    print("=" * 60)


if __name__ == '__main__':
    simple_strategy_example()
    backtest_strategy_example()
