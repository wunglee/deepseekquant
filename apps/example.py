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

    # 示例：订单执行
    order_data = {
        'symbol': 'AAPL',
        'quantity': 10,
        'side': TradeDirection.LONG,
        'order_type': OrderType.MARKET,
        'price': 150.0
    }
    exec_result = p_exec.process(order=order_data)
    print(f"订单执行: {exec_result.get('status')}")

    orchestrator.cleanup_all()
    print("\n=== 示例完成 ===")


if __name__ == '__main__':
    run()
