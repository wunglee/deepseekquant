"""
量化策略模块
从 core_bak/signal_engine.py 提取的量化策略方法占位实现
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from common import TradingSignal, SignalType


@dataclass
class StrategyContext:
    """策略上下文"""
    symbol: str
    prices: List[float]
    volumes: List[float]
    timestamp: str
    metadata: Dict[str, Any]


class QuantitativeStrategies:
    """量化策略集合"""

    @staticmethod
    def mean_reversion(context: StrategyContext, lookback: int = 20, entry_std: float = 2.0) -> Optional[TradingSignal]:
        # TODO：补充了均值回归策略占位实现，待确认
        """
        均值回归策略：基于价格偏离移动平均的标准差倍数
        
        从 core_bak/signal_engine.py:_mean_reversion_strategy 提取
        """
        if len(context.prices) < lookback:
            return None
        
        recent_prices = context.prices[-lookback:]
        mean_price = sum(recent_prices) / len(recent_prices)
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = variance ** 0.5
        
        current_price = context.prices[-1]
        z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0.0
        
        signal_type = SignalType.HOLD
        if z_score > entry_std:
            signal_type = SignalType.SELL  # 价格过高，均值回归卖出
        elif z_score < -entry_std:
            signal_type = SignalType.BUY  # 价格过低，均值回归买入
        
        if signal_type == SignalType.HOLD:
            return None
        
        from datetime import datetime
        from common import SignalMetadata
        return TradingSignal(
            id=f"{context.symbol}-{datetime.now().timestamp()}",
            symbol=context.symbol,
            signal_type=signal_type,
            price=current_price,
            timestamp=context.timestamp,
            reason=f"Mean reversion z-score={z_score:.2f}",
            metadata=SignalMetadata(parameters={'z_score': z_score, 'mean': mean_price, 'std': std_dev})
        )

    @staticmethod
    def momentum(context: StrategyContext, lookback: int = 10, threshold: float = 0.02) -> Optional[TradingSignal]:
        # TODO：补充了动量策略占位实现，待确认
        """
        动量策略：基于价格变化率
        
        从 core_bak/signal_engine.py:_momentum_strategy 提取
        """
        if len(context.prices) < lookback + 1:
            return None
        
        old_price = context.prices[-(lookback + 1)]
        current_price = context.prices[-1]
        
        if old_price <= 0:
            return None
        
        momentum_pct = (current_price - old_price) / old_price
        
        signal_type = SignalType.HOLD
        if momentum_pct > threshold:
            signal_type = SignalType.BUY
        elif momentum_pct < -threshold:
            signal_type = SignalType.SELL
        
        if signal_type == SignalType.HOLD:
            return None
        
        from datetime import datetime
        from common import SignalMetadata
        return TradingSignal(
            id=f"{context.symbol}-{datetime.now().timestamp()}",
            symbol=context.symbol,
            signal_type=signal_type,
            price=current_price,
            timestamp=context.timestamp,
            reason=f"Momentum {momentum_pct:.2%}",
            metadata=SignalMetadata(parameters={'momentum': momentum_pct, 'threshold': threshold})
        )

    @staticmethod
    def breakout(context: StrategyContext, lookback: int = 20, volume_confirm: bool = True) -> Optional[TradingSignal]:
        # TODO：补充了突破策略占位实现，待确认
        """
        突破策略：价格突破历史高点/低点
        
        从 core_bak/signal_engine.py:_breakout_strategy 提取
        """
        if len(context.prices) < lookback:
            return None
        
        recent_prices = context.prices[-lookback:-1]  # 排除当前价格
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        
        current_price = context.prices[-1]
        
        signal_type = SignalType.HOLD
        if current_price > recent_high:
            signal_type = SignalType.BUY  # 突破上轨
        elif current_price < recent_low:
            signal_type = SignalType.SELL  # 突破下轨
        
        # 可选：成交量确认
        if volume_confirm and signal_type != SignalType.HOLD and len(context.volumes) >= lookback:
            recent_avg_vol = sum(context.volumes[-lookback:-1]) / (lookback - 1) if lookback > 1 else 1.0
            current_vol = context.volumes[-1]
            if current_vol < recent_avg_vol:
                # 成交量不足，忽略信号
                signal_type = SignalType.HOLD
        
        if signal_type == SignalType.HOLD:
            return None
        
        from datetime import datetime
        from common import SignalMetadata
        return TradingSignal(
            id=f"{context.symbol}-{datetime.now().timestamp()}",
            symbol=context.symbol,
            signal_type=signal_type,
            price=current_price,
            timestamp=context.timestamp,
            reason=f"Breakout {'上轨' if signal_type == SignalType.BUY else '下轨'}",
            metadata=SignalMetadata(parameters={'high': recent_high, 'low': recent_low})
        )

    @staticmethod
    def volatility_contraction(context: StrategyContext, lookback: int = 20, contraction_pct: float = 0.5) -> Optional[TradingSignal]:
        # TODO：补充了波动率收缩策略占位实现，待确认
        """
        波动率收缩策略：低波动后预期扩张
        
        从 core_bak/signal_engine.py:_volatility_strategy 提取
        """
        if len(context.prices) < lookback * 2:
            return None
        
        # 计算历史波动率
        recent_returns = []
        for i in range(-lookback, -1):
            if context.prices[i-1] > 0:
                ret = (context.prices[i] - context.prices[i-1]) / context.prices[i-1]
                recent_returns.append(ret)
        
        if not recent_returns:
            return None
        
        recent_std = (sum((r - sum(recent_returns)/len(recent_returns)) ** 2 for r in recent_returns) / len(recent_returns)) ** 0.5
        
        # 长期波动率
        long_returns = []
        for i in range(-lookback*2, -lookback):
            if context.prices[i-1] > 0:
                ret = (context.prices[i] - context.prices[i-1]) / context.prices[i-1]
                long_returns.append(ret)
        
        if not long_returns:
            return None
        
        long_std = (sum((r - sum(long_returns)/len(long_returns)) ** 2 for r in long_returns) / len(long_returns)) ** 0.5
        
        # 波动率收缩检测
        if recent_std < long_std * contraction_pct:
            # 低波动，暂不生成信号（可结合其他指标）
            pass
        
        # 占位：当前简化返回None，后续可结合方向预测
        return None
