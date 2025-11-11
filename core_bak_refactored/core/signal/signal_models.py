"""
信号数据模型
从 core_bak/signal_engine.py 拆分
职责: 定义交易信号相关的枚举和数据结构
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

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


