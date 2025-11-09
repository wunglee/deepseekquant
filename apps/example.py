from infrastructure.interfaces import InfrastructureProvider
from core.signal.signal_processor import SignalProcessor
from core.risk.risk_processor import RiskProcessor
from core.exec.exec_processor import ExecProcessor
from core.portfolio.portfolio_processor import PortfolioProcessor


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
    print(report)


if __name__ == '__main__':
    run()
