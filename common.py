"""
DeepSeekQuant 公共模块
包含系统共享的枚举、常量和数据类型定义
避免在各模块中重复定义，确保一致性
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from typing import Type, TypeVar
import math

T = TypeVar('T')

class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REVERSE = "reverse"
    SCALP = "scalp"
    SWING = "swing"
    POSITION = "position"
    ARBITRAGE = "arbitrage"
    HEDGE = "hedge"

class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = "weak"  # 0-25%
    MILD = "mild"  # 25-50%
    STRONG = "strong"  # 50-75%
    VERY_STRONG = "very_strong"  # 75-100%
    EXTREME = "extreme"  # 特殊事件驱动

class SignalSource(Enum):
    """信号来源枚举"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"
    MACHINE_LEARNING = "machine_learning"
    SENTIMENT = "sentiment"
    MARKET_MAKER = "market_maker"
    ARBITRAGE = "arbitrage"
    MANUAL = "manual"
    COMPOSITE = "composite"

class SignalStatus(Enum):
    """信号状态枚举"""
    GENERATED = "generated"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"

class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = "very_low"  # 0-20%
    LOW = "low"  # 20-40%
    MODERATE = "moderate"  # 40-60%
    HIGH = "high"  # 60-80%
    VERY_HIGH = "very_high"  # 80-95%
    EXTREME = "extreme"  # 95-100%
    BLACK_SWAN = "black_swan"  # 极端事件
    UNKNOWN = "unknown"  # 未知风险

class PortfolioObjective(Enum):
    """组合目标枚举"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_SORTINO = "maximize_sortino"
    MAXIMIZE_ALPHA = "maximize_alpha"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MATCH_BENCHMARK = "match_benchmark"
    INCOME_GENERATION = "income_generation"
    CAPITAL_PRESERVATION = "capital_preservation"
    SPECULATIVE_GROWTH = "speculative_growth"

class AllocationMethod(Enum):
    """资产配置方法枚举"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    MINIMUM_VARIANCE = "minimum_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    CONSTANT_PROPORTION = "constant_proportion"
    DYNAMIC_ALLOCATION = "dynamic_allocation"
    OPTIMAL_RISK = "optimal_risk"
    OPTIMAL_RETURN = "optimal_return"
    CUSTOM_WEIGHTS = "custom_weights"

class ExecutionStrategy(Enum):
    """执行策略枚举"""
    MARKET_ORDER = "market_order"
    LIMIT_ORDER = "limit_order"
    STOP_ORDER = "stop_order"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    PARTICIPATE = "participate"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"

class TradeDirection(Enum):
    """交易方向枚举"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    HEDGE = "hedge"

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TRAILING_LIMIT = "trailing_limit"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    FOK = "fok"  # Fill-or-Kill
    IOC = "ioc"  # Immediate-or-Cancel

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionStatus(Enum):
    """仓位状态枚举"""
    OPEN = "open"
    CLOSED = "closed"
    HEDGED = "hedged"
    FLAT = "flat"

class DataFrequency(Enum):
    """数据频率枚举"""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class DataSourceType(Enum):
    """数据源类型枚举"""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    FINNHUB = "finnhub"
    TIINGO = "tiingo"
    QUANDL = "quandl"
    INTRINIO = "intrinio"
    EOD_HISTORICAL = "eod_historical"
    CUSTOM_API = "custom_api"
    DATABASE = "database"
    BROKER_API = "broker_api"

class ProcessorState(Enum):
    """处理器状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    SAFE_MODE = "safe_mode"
    MAINTENANCE = "maintenance"
    RECOVERY = "recovery"
    TERMINATED = "terminated"

class TradingMode(Enum):
    """交易模式枚举"""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    BACKTESTING = "backtesting"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"

class ConfigFormat(Enum):
    """配置文件格式枚举"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"

class ConfigSource(Enum):
    """配置来源枚举"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    ENVIRONMENT = "environment"
    DEFAULT = "default"

class AcquisitionFunctionType(Enum):
    """采集函数类型枚举"""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    THOMPSON_SAMPLING = "thompson_sampling"
    ENTROPY_SEARCH = "entropy_search"

class OptimizationObjective(Enum):
    """优化目标枚举"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass
class SignalMetadata:
    """信号元数据"""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: SignalSource = SignalSource.TECHNICAL
    confidence: float = 0.0  # 0.0 - 1.0
    strength: SignalStrength = SignalStrength.MILD
    priority: int = 1  # 1-10, 10为最高优先级
    expiration: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "1.0.0"
    backtest_id: Optional[str] = None
    strategy_name: Optional[str] = None

@dataclass
class TradingSignal:
    """交易信号数据类"""
    id: str
    symbol: str
    signal_type: SignalType
    price: float
    timestamp: str
    metadata: SignalMetadata
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1d"
    status: SignalStatus = SignalStatus.GENERATED
    reason: str = ""
    weight: float = 1.0  # 信号权重
    correlation: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0  # 0.0-1.0, 越低越好
    expected_return: float = 0.0
    expected_hold_period: int = 0  # 预期持有期（分钟）
    volume_ratio: float = 1.0  # 成交量比率
    volatility: float = 0.0  # 波动率指标
    liquidity_score: float = 1.0  # 流动性评分

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'price': self.price,
            'timestamp': self.timestamp,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timeframe': self.timeframe,
            'status': self.status.value,
            'reason': self.reason,
            'weight': self.weight,
            'correlation': self.correlation,
            'risk_score': self.risk_score,
            'expected_return': self.expected_return,
            'expected_hold_period': self.expected_hold_period,
            'volume_ratio': self.volume_ratio,
            'volatility': self.volatility,
            'liquidity_score': self.liquidity_score,
            'metadata': asdict(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """从字典创建信号"""
        metadata_data = data.pop('metadata', {})
        metadata = SignalMetadata(**metadata_data)

        return cls(
            id=data['id'],
            symbol=data['symbol'],
            signal_type=SignalType(data['signal_type']),
            price=data['price'],
            timestamp=data['timestamp'],
            metadata=metadata,
            quantity=data.get('quantity'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            timeframe=data.get('timeframe', '1d'),
            status=SignalStatus(data.get('status', 'generated')),
            reason=data.get('reason', ''),
            weight=data.get('weight', 1.0),
            correlation=data.get('correlation', {}),
            risk_score=data.get('risk_score', 0.0),
            expected_return=data.get('expected_return', 0.0),
            expected_hold_period=data.get('expected_hold_period', 0),
            volume_ratio=data.get('volume_ratio', 1.0),
            volatility=data.get('volatility', 0.0),
            liquidity_score=data.get('liquidity_score', 1.0)
        )

@dataclass
class RiskAssessment:
    """风险评估数据类"""
    approved: bool = False
    reason: str = ""
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    warnings: List[str] = field(default_factory=list)
    max_position_size: float = 0.0
    suggested_allocation: float = 0.0
    risk_score: float = 0.0
    var_95: float = 0.0  # 95%置信度的VaR
    expected_shortfall: float = 0.0
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    liquidity_impact: float = 0.0
    correlation_risks: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'approved': self.approved,
            'reason': self.reason,
            'risk_level': self.risk_level.value,
            'warnings': self.warnings,
            'max_position_size': self.max_position_size,
            'suggested_allocation': self.suggested_allocation,
            'risk_score': self.risk_score,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'stress_test_results': self.stress_test_results,
            'liquidity_impact': self.liquidity_impact,
            'correlation_risks': self.correlation_risks
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """从字典创建风险评估"""
        return cls(
            approved=data.get('approved', False),
            reason=data.get('reason', ''),
            risk_level=RiskLevel(data.get('risk_level', 'unknown')),
            warnings=data.get('warnings', []),
            max_position_size=data.get('max_position_size', 0.0),
            suggested_allocation=data.get('suggested_allocation', 0.0),
            risk_score=data.get('risk_score', 0.0),
            var_95=data.get('var_95', 0.0),
            expected_shortfall=data.get('expected_shortfall', 0.0),
            stress_test_results=data.get('stress_test_results', {}),
            liquidity_impact=data.get('liquidity_impact', 0.0),
            correlation_risks=data.get('correlation_risks', {})
        )

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    last_operation_time: Optional[str] = None
    throughput: float = 0.0  # 操作数/秒
    error_rate: float = 0.0
    availability: float = 1.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    def update(self, success: bool, processing_time: float):
        """更新性能指标"""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.total_processing_time += processing_time
        self.avg_processing_time = self.total_processing_time / self.total_operations
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.last_operation_time = datetime.now().isoformat()

        if self.total_processing_time > 0:
            self.throughput = self.total_operations / self.total_processing_time

        if self.total_operations > 0:
            self.error_rate = self.failed_operations / self.total_operations
            self.availability = self.successful_operations / self.total_operations

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.avg_processing_time,
            'max_processing_time': self.max_processing_time,
            'min_processing_time': self.min_processing_time,
            'last_operation_time': self.last_operation_time,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'availability': self.availability,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hit_rate
        }

@dataclass
class SystemStatus:
    """系统状态数据类"""
    state: ProcessorState
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
    risk_level: RiskLevel

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state': self.state.value,
            'trading_mode': self.trading_mode.value,
            'uptime': self.uptime,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'active_modules': self.active_modules,
            'error_count': self.error_count,
            'last_update': self.last_update,
            'performance_metrics': self.performance_metrics,
            'positions_count': self.positions_count,
            'trading_volume': self.trading_volume,
            'risk_level': self.risk_level.value
        }

# 常量定义
DEFAULT_CONFIG_PATH = "config/deepseekquant.json"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE = 100 * 1024 * 1024  # 100MB
BACKUP_LOG_COUNT = 10
DEFAULT_ENCODING = "utf-8"

# 错误消息常量
ERROR_CONFIG_LOAD = "配置加载失败"
ERROR_CONFIG_VALIDATION = "配置验证失败"
ERROR_DATA_FETCH = "数据获取失败"
ERROR_SIGNAL_GENERATION = "信号生成失败"
ERROR_RISK_ASSESSMENT = "风险评估失败"
ERROR_EXECUTION = "执行失败"
ERROR_PORTFOLIO_UPDATE = "组合更新失败"
ERROR_MONITORING = "监控错误"
ERROR_API_CALL = "API调用失败"

# 成功消息常量
SUCCESS_CONFIG_LOAD = "配置加载成功"
SUCCESS_CONFIG_VALIDATION = "配置验证成功"
SUCCESS_DATA_FETCH = "数据获取成功"
SUCCESS_SIGNAL_GENERATION = "信号生成成功"
SUCCESS_RISK_ASSESSMENT = "风险评估成功"
SUCCESS_EXECUTION = "执行成功"
SUCCESS_PORTFOLIO_UPDATE = "组合更新成功"

# 警告消息常量
WARNING_CONFIG_DEFAULT = "使用默认配置"
WARNING_DATA_STALE = "数据可能过时"
WARNING_HIGH_RISK = "高风险警告"
WARNING_LIQUIDITY_LOW = "流动性不足"
WARNING_VOLATILITY_HIGH = "波动率过高"

# 信息消息常量
INFO_SYSTEM_START = "系统启动"
INFO_SYSTEM_SHUTDOWN = "系统关闭"
INFO_MODULE_INIT = "模块初始化"
INFO_MODULE_READY = "模块就绪"
INFO_TRADE_EXECUTED = "交易已执行"
INFO_PORTFOLIO_REBALANCED = "组合再平衡完成"

# 配置常量
DEFAULT_INITIAL_CAPITAL = 1000000.0
DEFAULT_COMMISSION_RATE = 0.001
DEFAULT_SLIPPAGE_RATE = 0.0005
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_MAX_POSITION_SIZE = 0.1
DEFAULT_MAX_LEVERAGE = 2.0
DEFAULT_STOP_LOSS_PCT = 0.08
DEFAULT_TAKE_PROFIT_PCT = 0.15
DEFAULT_VAR_CONFIDENCE_LEVEL = 0.95
DEFAULT_STRESS_TEST_SCENARIOS = ["flash_crash", "liquidity_crisis", "black_swan"]

# 时间常量
MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5  # 美股交易时间

# 数学常量
EPSILON = 1e-10  # 极小值，避免除以零
try:
    import numpy as np
    ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR)  # 年化因子
except ImportError:
    # 如果没有安装 numpy，使用 math 替代
    ANNUALIZATION_FACTOR = math.sqrt(TRADING_DAYS_PER_YEAR)  # 年化因子

def enum_to_dict(enum_class: Type[Enum]) -> Dict[str, Any]:
    """将枚举类转换为字典"""
    return {member.name: member.value for member in enum_class}

def validate_enum_value(enum_class: Type[Enum], value: Any) -> bool:
    """验证值是否属于枚举类"""
    try:
        enum_class(value)
        return True
    except ValueError:
        return False

def get_enum_values(enum_class: Type[Enum]) -> List[Any]:
    """获取枚举的所有值"""
    return [member.value for member in enum_class]

def get_enum_names(enum_class: Type[Enum]) -> List[str]:
    """获取枚举的所有名称"""
    return [member.name for member in enum_class]

def serialize_dict(data: Dict[str, Any]) -> str:
    """序列化字典为JSON字符串"""
    return json.dumps(data, indent=2, ensure_ascii=False)

def deserialize_dict(data_str: str) -> Dict[str, Any]:
    """反序列化JSON字符串为字典"""
    return json.loads(data_str)

class DeepSeekQuantEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持枚举和日期时间"""
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, 'to_dict'):
            return o.to_dict()
        return super().default(o)

def deep_seek_quant_object_hook(dct):
    """自定义JSON对象钩子，用于反序列化"""
    # 这里可以添加自定义的反序列化逻辑
    return dct