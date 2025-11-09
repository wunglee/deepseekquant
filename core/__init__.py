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

# 轻量化初始化，避免在包导入时加载重模块
try:
    from .main import DeepSeekQuantSystem
except Exception:
    DeepSeekQuantSystem = None

__all__ = []

class DeepSeekQuant:
    """DeepSeekQuant 主类 - 提供简化的入口点"""

    def __init__(self, config_path: str = None):
        """初始化系统"""
        if DeepSeekQuantSystem is not None:
            self.system = DeepSeekQuantSystem(config_path)
        else:
            self.system = None

    def start(self):
        """启动系统"""
        return self.system.start() if self.system else None

    def stop(self):
        """停止系统"""
        return self.system.stop() if self.system else None

    def run_backtest(self, strategy_config: dict):
        """运行回测"""
        return self.system.run_backtest(strategy_config) if self.system else None

    def get_status(self):
        """获取系统状态"""
        return self.system.get_status() if self.system else {'status': 'uninitialized'}

# 版本兼容性
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("deepseekquant")
except PackageNotFoundError:
    pass