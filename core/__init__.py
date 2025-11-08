"""
DeepSeekQuant - 专业级量化交易系统
版本: 1.0.0
作者: DeepSeek AI
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "DeepSeek AI"
__license__ = "MIT"
__email__ = "quant@deepseek.com"

from .main import DeepSeekQuantSystem
from .config.config_manager import ConfigManager
from .core.data_fetcher import DataFetcher
from .core.signal_engine import SignalEngine, Signal, SignalType
from .core.portfolio_manager import PortfolioManager, AllocationMethod
from .core.risk_manager import RiskManager, RiskLevel, RiskAssessment
from .core.execution_engine import ExecutionEngine, ExecutionStrategy, TradeCost
from .core.bayesian_optimizer import BayesianOptimizer
from .analytics.backtesting import BacktestingEngine
from .analytics.performance import PerformanceAnalyzer
from .infrastructure.monitoring import MonitoringSystem
from .infrastructure.api_gateway import APIGateway

__all__ = [
    'DeepSeekQuantSystem',
    'ConfigManager',
    'DataFetcher',
    'SignalEngine',
    'Signal',
    'SignalType',
    'PortfolioManager',
    'AllocationMethod',
    'RiskManager',
    'RiskLevel',
    'RiskAssessment',
    'ExecutionEngine',
    'ExecutionStrategy',
    'TradeCost',
    'BayesianOptimizer',
    'BacktestingEngine',
    'PerformanceAnalyzer',
    'MonitoringSystem',
    'APIGateway'
]

class DeepSeekQuant:
    """DeepSeekQuant 主类 - 提供简化的入口点"""

    def __init__(self, config_path: str = None):
        """初始化系统"""
        self.system = DeepSeekQuantSystem(config_path)

    def start(self):
        """启动系统"""
        return self.system.start()

    def stop(self):
        """停止系统"""
        return self.system.stop()

    def run_backtest(self, strategy_config: dict):
        """运行回测"""
        return self.system.run_backtest(strategy_config)

    def get_status(self):
        """获取系统状态"""
        return self.system.get_status()

# 版本兼容性
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("deepseekquant")
except PackageNotFoundError:
    pass