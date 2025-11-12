"""
执行引擎系统 - 枚举和数据模型
拆分自: core_bak/execution_engine.py (line 46-320)
职责: 定义执行相关的所有枚举和数据类
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    MOC = "moc"
    LOC = "loc"


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING_CANCEL = "pending_cancel"


class ExecutionAlgorithm(Enum):
    """执行算法枚举"""
    SIMPLE = "simple"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    LIQUIDITY_SEEKING = "liquidity_seeking"
    ADAPTIVE = "adaptive"
    SNIPER = "sniper"
    STEALTH = "stealth"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


class BrokerType(Enum):
    """经纪商类型枚举"""
    SIMULATED = "simulated"
    IBKR = "interactive_brokers"
    ALPACA = "alpaca"
    ROBINHOOD = "robinhood"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    OANDA = "oanda"
    FXCM = "fxcm"
    CUSTOM = "custom"


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"


class TimeInForce(Enum):
    """订单有效期枚举"""
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    IOC = "ioc"
    FOK = "fok"
    GTD = "gtd"


@dataclass
class OrderParameters:
    """订单参数"""
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    good_till_date: Optional[str] = None
    iceberg: bool = False
    display_quantity: Optional[int] = None
    min_quantity: Optional[int] = None
    execution_instructions: List[str] = field(default_factory=list)
    routing_instructions: Dict[str, Any] = field(default_factory=lambda: {})
    algo_parameters: Dict[str, Any] = field(default_factory=lambda: {})
    smart_routing: bool = True
    destination: Optional[str] = None
    allocation_strategy: str = "fifo"
    notional: bool = False


@dataclass
class ExecutionParameters:
    """执行参数"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SIMPLE
    urgency: str = "medium"
    max_participation_rate: float = 0.1
    target_performance: str = "price"
    price_deviation_limit: float = 0.02
    volume_limit: float = 0.2
    time_horizon: int = 300
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    allow_iceberg: bool = False
    allow_dark_pools: bool = False
    avoid_ecns: bool = False
    benchmark: str = "arrival_price"
    tolerance_bands: Dict[str, float] = field(default_factory=lambda: {"upper": 0.01, "lower": -0.01})
    slippage_control: bool = True
    real_time_adjustment: bool = True


@dataclass
class Order:
    """订单数据类"""
    order_id: str
    portfolio_id: str
    symbol: str
    quantity: float
    side: OrderSide
    parameters: OrderParameters
    execution_params: ExecutionParameters
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    timing_risk: float = 0.0
    execution_quality: float = 0.0
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=lambda: {})
    risk_check_passed: bool = False
    broker_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None
    expiration_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['side'] = self.side.value
        result['status'] = self.status.value
        return result


@dataclass
class ExecutionReport:
    """执行报告"""
    report_id: str
    order_id: str
    timestamp: str
    fill_price: float
    fill_quantity: float
    remaining_quantity: float
    cumulative_quantity: float
    execution_type: str
    liquidity: str
    venue: str
    execution_id: str
    transaction_time: str
    commission: float
    fees: float
    slippage: float
    market_impact: float
    timing_risk: float
    execution_quality: float
    benchmark_comparison: Dict[str, float]
    flags: List[str]
    metadata: Dict[str, Any]


@dataclass
class BrokerConnection:
    """经纪商连接配置"""
    broker_type: BrokerType
    account_id: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_token: Optional[str] = None
    base_url: str = ""
    websocket_url: Optional[str] = None
    paper_trading: bool = True
    rate_limit: int = 100
    connection_timeout: int = 30
    request_timeout: int = 10
    retry_attempts: int = 3
    retry_delay: int = 1
    ssl_verification: bool = True
    proxy_settings: Optional[Dict[str, Any]] = None
    custom_headers: Dict[str, str] = field(default_factory=lambda: {})
    session_token: Optional[str] = None
    token_expiry: Optional[str] = None
    last_heartbeat: Optional[str] = None
    connection_status: str = "disconnected"
