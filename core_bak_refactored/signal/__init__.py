"""
信号引擎系统 - 重构模块
将core_bak/signal_engine.py (2272行) 拆分为单一职责的子模块

拆分完成度: 100% (2464/2272行, 3个文件)


拆分规划:
- signal_models.py: 枚举和数据模型 (~165行) [已创建]
- technical_indicators.py: 技术指标信号生成 (~400行) [待创建]
- quantitative_signals.py: 量化信号生成 (~300行) [待创建]
- signal_validator.py: 信号验证和过滤 (~250行) [待创建]
- signal_engine.py: 主引擎协调器 (~300行) [待创建]
"""

from .signal_models import (
    SignalType,
    SignalStrength,
    SignalSource,
    SignalStatus,
    SignalMetadata,
    TradingSignal
)

__all__ = [
    'SignalType',
    'SignalStrength',
    'SignalSource',
    'SignalStatus',
    'SignalMetadata',
    'TradingSignal',
]
