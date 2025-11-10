from infrastructure.interfaces import InfrastructureProvider
from core.signal.signal_processor import SignalProcessor
from core.risk.risk_processor import RiskProcessor
from core.exec.exec_processor import ExecProcessor
from core.portfolio.portfolio_processor import PortfolioProcessor
from core.data.data_fetcher import DataFetcher
from common import TradeDirection, OrderType


def run():
    orchestrator = InfrastructureProvider.get('orchestrator')
    p_signal = SignalProcessor(processor_name='SignalProcessor')
    p_risk = RiskProcessor(processor_name='RiskProcessor')
    p_exec = ExecProcessor(processor_name='ExecProcessor')
    p_port = PortfolioProcessor(processor_name='PortfolioProcessor')

    orchestrator.register_processor(p_signal)
    orchestrator.register_processor(p_risk)
    orchestrator.register_processor(p_exec)
    orchestrator.register_processor(p_port)

    orchestrator.initialize_all()
    report = orchestrator.get_health_report()
    print("=== 系统健康报告 ===")
    print(report)

    # 示例：数据抓取
    fetcher = DataFetcher()
    md = fetcher.get_market_data('AAPL', use_cache=False)
    print(f"\n市场数据: {md.get('status')}")

    # 示例：信号生成
    signal_result = p_signal.process(symbol='AAPL', price=150.0)
    print(f"信号: {signal_result.get('status')}")

    # 示例：组合分配
    positions = [
        {'symbol': 'AAPL', 'quantity': 10, 'price': 150},
        {'symbol': 'GOOG', 'quantity': 5, 'price': 2800}
    ]
    port_result = p_port.process(positions=positions)
    print(f"组合权重: {port_result.get('weights')}")
    if port_result.get('rebalance'):
        print(f"再平衡指令: {port_result.get('rebalance')}")
        print(f"换手率: {port_result.get('turnover_rate')}\n")

    # 风险评估：使用目标权重与历史波动率阈值
    weights = port_result.get('weights', {})
    fetch_hist = {sym: DataFetcher().get_history(sym, lookback=30) for sym in weights.keys()}
    for sym, w in weights.items():
        closes = [row['close'] for row in fetch_hist.get(sym, [])]
        risk_limits = {'volatility_threshold': 0.05, 'concentration_threshold': 0.6}
        risk_result = p_risk.process(signal={'price': 100.0, 'quantity': 10}, limits=risk_limits, prices=closes, target_weight=w)
        print(f"风险评估[{sym}]: {risk_result.get('assessment', {}).get('warnings', [])}")

    # 组合层面的整体风险（HHI/集中度）
    overall_histories = {sym: [row['close'] for row in fetch_hist.get(sym, [])] for sym in weights.keys()}
    overall_limits = {'hhi_threshold': 0.45, 'concentration_threshold': 0.6, 'correlation_threshold': 0.6, 'max_drawdown_threshold': 0.2}
    overall_risk = p_risk.process(signal={'price': 0.0, 'quantity': 0.0}, limits=overall_limits, weights=weights, histories=overall_histories)
    print(f"组合整体风险: {overall_risk.get('assessment', {}).get('warnings', [])}")


    # 示例：订单执行
    order_data = {
        'symbol': 'AAPL',
        'quantity': 10,
        'side': TradeDirection.LONG,
        'order_type': OrderType.MARKET,
        'price': 150.0
    }
    exec_result = p_exec.process(order=order_data, commission=0.001, slippage=0.0005)
    print(f"订单执行: {exec_result.get('status')}")
    if exec_result.get('execution_report'):
        print(f"执行报告: {exec_result['execution_report']}")

    orchestrator.cleanup_processors()
    print("\n=== 示例完成 ===")


if __name__ == '__main__':
    run()
