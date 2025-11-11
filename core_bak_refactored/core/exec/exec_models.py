"""
执行引擎数据模型
从 core_bak/execution_engine.py 拆分
职责: 定义订单、执行相关的枚举和数据结构
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 跟踪止损单
    ICEBERG = "iceberg"  # 冰山订单
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    POV = "pov"  # 参与率订单
    MOC = "moc"  # 收盘市价单
    LOC = "loc"  # 收盘限价单


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"  # 等待提交
    SUBMITTED = "submitted"  # 已提交
    ACCEPTED = "accepted"  # 已接受
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期
    SUSPENDED = "suspended"  # 已暂停
    PENDING_CANCEL = "pending_cancel"  # 等待取消


class ExecutionAlgorithm(Enum):
    """执行算法枚举"""
    SIMPLE = "simple"  # 简单执行
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    POV = "pov"  # 参与率
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"  # 执行差额
    LIQUIDITY_SEEKING = "liquidity_seeking"  # 流动性寻找
    ADAPTIVE = "adaptive"  # 自适应执行
    SNIPER = "sniper"  # 狙击执行
    STEALTH = "stealth"  # 隐身执行
    AGGRESSIVE = "aggressive"  # 激进执行
    PASSIVE = "passive"  # 被动执行


class BrokerType(Enum):
    """经纪商类型枚举"""
    SIMULATED = "simulated"  # 模拟经纪商
    IBKR = "interactive_brokers"  # Interactive Brokers
    ALPACA = "alpaca"  # Alpaca
    ROBINHOOD = "robinhood"  # Robinhood
    BINANCE = "binance"  # Binance
    COINBASE = "coinbase"  # Coinbase
    KRAKEN = "kraken"  # Kraken
    OANDA = "oanda"  # OANDA
    FXCM = "fxcm"  # FXCM
    CUSTOM = "custom"  # 自定义接口


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"  # 买入
    SELL = "sell"  # 卖出
    SELL_SHORT = "sell_short"  # 卖空
    BUY_TO_COVER = "buy_to_cover"  # 平仓买入


class TimeInForce(Enum):
    """订单有效期枚举"""
    DAY = "day"  # 当日有效
    GTC = "gtc"  # 撤销前有效
    OPG = "opg"  # 开盘时执行
    IOC = "ioc"  # 立即或取消
    FOK = "fok"  # 全部或取消
    GTD = "gtd"  # 指定日期前有效


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
    routing_instructions: Dict[str, Any] = field(default_factory=dict)
    algo_parameters: Dict[str, Any] = field(default_factory=dict)
    smart_routing: bool = True
    destination: Optional[str] = None
    allocation_strategy: str = "fifo"
    notional: bool = False  # 是否按名义价值下单


@dataclass
class ExecutionParameters:
    """执行参数"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SIMPLE
    urgency: str = "medium"  # low, medium, high, urgent
    max_participation_rate: float = 0.1  # 最大市场参与率
    target_performance: str = "price"  # price, time, liquidity
    price_deviation_limit: float = 0.02  # 价格偏离限制
    volume_limit: float = 0.2  # 成交量限制
    time_horizon: int = 300  # 执行时间窗口（秒）
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    allow_iceberg: bool = False
    allow_dark_pools: bool = False
    avoid_ecns: bool = False
    benchmark: str = "arrival_price"  # 执行基准
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_check_passed: bool = False
    broker_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None
    expiration_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'portfolio_id': self.portfolio_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'side': self.side.value,
            'parameters': asdict(self.parameters),
            'execution_params': asdict(self.execution_params),
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'fees': self.fees,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'timing_risk': self.timing_risk,
            'execution_quality': self.execution_quality,
            'parent_order_id': self.parent_order_id,
            'child_orders': self.child_orders,
            'tags': self.tags,
            'metadata': self.metadata,
            'risk_check_passed': self.risk_check_passed,
            'broker_order_id': self.broker_order_id,
            'exchange_order_id': self.exchange_order_id,
            'rejection_reason': self.rejection_reason,
            'cancellation_reason': self.cancellation_reason,
            'expiration_time': self.expiration_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """从字典创建订单"""
        params_data = data.pop('parameters', {})
        exec_params_data = data.pop('execution_params', {})

        parameters = OrderParameters(**params_data)
        execution_params = ExecutionParameters(**exec_params_data)

        return cls(
            order_id=data['order_id'],
            portfolio_id=data['portfolio_id'],
            symbol=data['symbol'],
            quantity=data['quantity'],
            side=OrderSide(data['side']),
            parameters=parameters,
            execution_params=execution_params,
            status=OrderStatus(data.get('status', 'pending')),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            filled_quantity=data.get('filled_quantity', 0.0),
            average_fill_price=data.get('average_fill_price', 0.0),
            commission=data.get('commission', 0.0),
            fees=data.get('fees', 0.0),
            slippage=data.get('slippage', 0.0),
            market_impact=data.get('market_impact', 0.0),
            timing_risk=data.get('timing_risk', 0.0),
            execution_quality=data.get('execution_quality', 0.0),
            parent_order_id=data.get('parent_order_id'),
            child_orders=data.get('child_orders', []),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            risk_check_passed=data.get('risk_check_passed', False),
            broker_order_id=data.get('broker_order_id'),
            exchange_order_id=data.get('exchange_order_id'),
            rejection_reason=data.get('rejection_reason'),
            cancellation_reason=data.get('cancellation_reason'),
            expiration_time=data.get('expiration_time')
        )


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
    execution_type: str  # partial, full, cancelled, etc.
    liquidity: str  # added, removed, hidden
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    custom_headers: Dict[str, str] = field(default_factory=dict)
    session_token: Optional[str] = None
    token_expiry: Optional[str] = None
    last_heartbeat: Optional[str] = None
    connection_status: str = "disconnected"
    supported_features: List[str] = field(default_factory=list)
    limitations: Dict[str, Any] = field(default_factory=dict)


