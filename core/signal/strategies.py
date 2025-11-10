"""
量化策略模块
从 core_bak/signal_engine.py 提取的真实业务逻辑实现
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import math

from common import TradingSignal, SignalType, SignalSource, SignalStrength, SignalMetadata


@dataclass
class StrategyContext:
    """策略上下文"""
    symbol: str
    prices: List[float]
    volumes: List[float]
    highs: Optional[List[float]] = None
    lows: Optional[List[float]] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class QuantitativeStrategies:
    """量化策略集合"""

    @staticmethod
    def mean_reversion(context: StrategyContext, lookback: int = 20, entry_std: float = 2.0) -> Optional[TradingSignal]:
        """
        均值回归策略：基于价格偏离移动平均的标准差倍数
        从 core_bak/signal_engine.py:_mean_reversion_strategy 提取（line 1228-1311）
        """
        if len(context.prices) < lookback:
            return None
        
        # 计算均值和标准差（与core_bak一致使用numpy方法）
        recent_prices = context.prices[-lookback:]
        mean = sum(recent_prices) / len(recent_prices)
        variance = sum((p - mean) ** 2 for p in recent_prices) / len(recent_prices)
        std = math.sqrt(variance)
        
        if std == 0:
            return None
        
        # 计算Z-score
        current_price = context.prices[-1]
        z_score = (current_price - mean) / std
        
        # 生成信号
        signal_type = None
        expected_return = 0.0
        
        # Z-score过高 - 卖出信号（价格会回归均值）
        if z_score > entry_std:
            signal_type = SignalType.SELL
            expected_return = mean - current_price  # 预期回归到均值
        # Z-score过低 - 买入信号（价格会回归均值）
        elif z_score < -entry_std:
            signal_type = SignalType.BUY
            expected_return = current_price - mean  # 预期回归到均值
        else:
            return None
        
        # 置信度基于Z-score
        confidence = min(abs(z_score) / 3, 0.9)
        
        return TradingSignal(
            id=f"mean_reversion_{context.symbol}_{int(time.time())}",
            symbol=context.symbol,
            signal_type=signal_type,
            price=current_price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.QUANTITATIVE,
                confidence=confidence,
                strength=SignalStrength.STRONG,
                parameters={
                    'lookback_period': lookback,
                    'entry_threshold': entry_std,
                    'exit_threshold': 0.5,  # 默认退出阈值
                    'z_score': float(z_score),
                    'type': 'mean_reversion'
                }
            ),
            weight=0.8,
            risk_score=0.4,
            expected_return=float(expected_return)
        )

    @staticmethod
    def momentum(context: StrategyContext, period: int = 10, threshold: float = 5.0) -> Optional[TradingSignal]:
        """
        动量策略：基于价格百分比动量
        从 core_bak/signal_engine.py:_momentum_strategy 提取（line 1313-1387）
        """
        if len(context.prices) < period + 1:
            return None
        
        # 计算动量（百分比）
        momentum = (context.prices[-1] / context.prices[-period] - 1) * 100
        
        # 生成信号
        signal_type = None
        
        # 正动量强劲 - 买入信号
        if momentum > threshold:
            signal_type = SignalType.BUY
        # 负动量强劲 - 卖出信号
        elif momentum < -threshold:
            signal_type = SignalType.SELL
        else:
            return None
        
        # 置信度基于动量强度
        confidence = min(abs(momentum) / 20, 0.9)
        
        return TradingSignal(
            id=f"momentum_{context.symbol}_{int(time.time())}",
            symbol=context.symbol,
            signal_type=signal_type,
            price=context.prices[-1],
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.QUANTITATIVE,
                confidence=confidence,
                strength=SignalStrength.VERY_STRONG,
                parameters={
                    'momentum_period': period,
                    'entry_threshold': threshold,
                    'exit_threshold': 2.0,  # 默认2%退出阈值
                    'momentum': float(momentum),
                    'type': 'momentum'
                }
            ),
            weight=0.9,
            risk_score=0.5,  # 动量策略风险较高
            expected_return=float(momentum / 2)  # 预期回报为动量的一半
        )

    @staticmethod
    def breakout(context: StrategyContext, lookback: int = 20, threshold: float = 0.02) -> Optional[TradingSignal]:
        """
        突破策略：价格突破阻力位/支撑位
        从 core_bak/signal_engine.py:_breakout_strategy 提取（line 1389-1465）
        """
        # 需要高低价数据
        if len(context.prices) < lookback or not context.highs or not context.lows:
            return None
        
        if len(context.highs) < lookback or len(context.lows) < lookback:
            return None
        
        # 计算阻力位和支撑位
        resistance = max(context.highs[-lookback:])
        support = min(context.lows[-lookback:])
        current_price = context.prices[-1]
        
        # 生成突破信号
        signal_type = None
        parameters = {
            'lookback_period': lookback,
            'breakout_threshold': threshold,
            'type': 'breakout'
        }
        
        # 上破阻力位 - 买入信号
        if current_price > resistance * (1 + threshold):
            signal_type = SignalType.BUY
            parameters['resistance'] = float(resistance)
        # 下破支撑位 - 卖出信号
        elif current_price < support * (1 - threshold):
            signal_type = SignalType.SELL
            parameters['support'] = float(support)
        else:
            return None
        
        return TradingSignal(
            id=f"breakout_{context.symbol}_{int(time.time())}",
            symbol=context.symbol,
            signal_type=signal_type,
            price=current_price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.QUANTITATIVE,
                confidence=0.7,
                strength=SignalStrength.STRONG,
                parameters=parameters
            ),
            weight=0.8,
            risk_score=0.4,
            expected_return=0.05  # 预期5%回报
        )

    @staticmethod
    def volatility_contraction(context: StrategyContext, lookback: int = 20, 
                              contraction_threshold: float = 0.5) -> Optional[TradingSignal]:
        """
        波动率收缩策略：波动率降低后的突破预期
        TODO：补充了波动率收缩策略实现，待确认（core_bak中未找到明确实现）
        """
        if len(context.prices) < lookback + 10:
            return None
        
        # 计算当前波动率（使用对数收益）
        recent_returns = []
        for i in range(-lookback+1, 0):
            if context.prices[i-1] > 0:
                ret = math.log(context.prices[i] / context.prices[i-1])
                recent_returns.append(ret)
        
        if len(recent_returns) < 5:
            return None
        
        recent_mean = sum(recent_returns) / len(recent_returns)
        recent_var = sum((r - recent_mean) ** 2 for r in recent_returns) / len(recent_returns)
        current_volatility = math.sqrt(recent_var) * math.sqrt(252)  # 年化波动率
        
        # 计算历史波动率
        historical_returns = []
        for i in range(-lookback-9, -9):
            if i-1 >= -len(context.prices) and context.prices[i-1] > 0:
                ret = math.log(context.prices[i] / context.prices[i-1])
                historical_returns.append(ret)
        
        if len(historical_returns) < 5:
            return None
        
        hist_mean = sum(historical_returns) / len(historical_returns)
        hist_var = sum((r - hist_mean) ** 2 for r in historical_returns) / len(historical_returns)
        historical_volatility = math.sqrt(hist_var) * math.sqrt(252)
        
        # 波动率收缩检测
        if historical_volatility == 0:
            return None
        
        volatility_ratio = current_volatility / historical_volatility
        
        # 波动率收缩幅度不足
        if volatility_ratio > contraction_threshold:
            return None
        
        # 波动率收缩后通常预示突破，默认多头方向
        # TODO：可根据趋势方向优化信号类型
        return TradingSignal(
            id=f"vol_contraction_{context.symbol}_{int(time.time())}",
            symbol=context.symbol,
            signal_type=SignalType.BUY,
            price=context.prices[-1],
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.QUANTITATIVE,
                confidence=0.6,
                strength=SignalStrength.MILD,
                parameters={
                    'lookback': lookback,
                    'current_volatility': float(current_volatility),
                    'historical_volatility': float(historical_volatility),
                    'volatility_ratio': float(volatility_ratio),
                    'type': 'volatility_contraction'
                }
            ),
            weight=0.7,
            risk_score=0.3,
            expected_return=0.03
        )
