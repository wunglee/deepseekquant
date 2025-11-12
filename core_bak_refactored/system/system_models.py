"""
主系统 - 枚举和数据模型
拆分自: core_bak/main.py (line 46-105)
职责: 定义系统相关的枚举和数据类
"""

from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum


class SystemState(Enum):
    """系统状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    SAFE_MODE = "safe_mode"
    MAINTENANCE = "maintenance"
    RECOVERY = "recovery"


class TradingMode(Enum):
    """交易模式枚举"""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    BACKTESTING = "backtesting"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"


@dataclass
class SystemStatus:
    """系统状态数据类"""
    state: SystemState
    trading_mode: TradingMode
    uptime: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_modules: int
    error_count: int
    last_update: str
    performance_metrics: Dict[str, float]
    positions_count: int
    trading_volume: float
    risk_level: str  # 避免循环导入，使用字符串


@dataclass
class TradingCycleMetrics:
    """交易周期指标数据类"""
    cycle_id: str
    start_time: str
    end_time: str
    duration: float
    signals_generated: int
    signals_executed: int
    risk_assessment: Dict[str, Any]
    trades_executed: int
    trading_volume: float
    slippage: float
    commission: float
    performance_metrics: Dict[str, float]
    error_count: int
    status: str
