"""兼容旧测试：暴露core.signal.signal_models接口"""
from core_bak_refactored.core.signal.signal_models import (
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
    'TradingSignal'
]
