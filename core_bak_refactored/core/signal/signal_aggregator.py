"""
信号聚合器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 聚合多个信号源、计算综合信号强度
"""

import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
from collections import defaultdict

from .signal_models import TradingSignal, SignalStrength, SignalType, SignalSource, SignalMetadata

logger = logging.getLogger('DeepSeekQuant.SignalAggregator')


class SignalAggregator:
    """信号聚合器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def aggregate(self, signals: Dict[str, List[TradingSignal]]) -> Dict[str, List[TradingSignal]]:
        """聚合多个信号源"""
        aggregated_signals = {}
        
        for symbol, symbol_signals in signals.items():
            if not symbol_signals:
                continue
            
            # 1. 按信号类型分组
            signal_groups = defaultdict(list)
            for signal in symbol_signals:
                key = (signal.signal_type, signal.timeframe)
                signal_groups[key].append(signal)
            
            # 2. 合并同组信号
            merged_signals = []
            for (signal_type, timeframe), signal_list in signal_groups.items():
                if len(signal_list) == 1:
                    merged_signals.append(signal_list[0])
                else:
                    # 创建复合信号
                    composite_signal = self._merge_signals(signal_type, timeframe, signal_list)
                    if composite_signal:
                        merged_signals.append(composite_signal)
            
            # 3. 计算综合强度
            for signal in merged_signals:
                group_key = (signal.signal_type, signal.timeframe)
                if len(signal_groups[group_key]) > 1:
                    signal.metadata.strength = self.calculate_weighted_strength(
                        signal_groups[group_key]
                    )
            
            aggregated_signals[symbol] = merged_signals
        
        return aggregated_signals
    
    def _merge_signals(self, signal_type: SignalType, timeframe: str, signals: List[TradingSignal]) -> TradingSignal:
        """合并相同方向的信号"""
        # 计算加权平均值
        total_weight = sum(signal.weight for signal in signals)
        if total_weight == 0:
            return None
        
        weighted_price = sum(signal.price * signal.weight for signal in signals) / total_weight
        weighted_confidence = sum(signal.metadata.confidence * signal.weight for signal in signals) / total_weight
        
        # 使用置信度最高的信号作为基础
        base_signal = max(signals, key=lambda s: s.metadata.confidence)
        
        return TradingSignal(
            id=f"aggregated_{base_signal.symbol}_{int(datetime.now().timestamp())}",
            symbol=base_signal.symbol,
            signal_type=signal_type,
            price=weighted_price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.COMPOSITE,
                confidence=weighted_confidence,
                strength=self.calculate_weighted_strength(signals),
                tags=['aggregated'] + base_signal.metadata.tags
            ),
            weight=total_weight / len(signals),
            timeframe=timeframe
        )
    
    def calculate_weighted_strength(self, signals: List[TradingSignal]) -> SignalStrength:
        """计算加权信号强度"""
        if not signals:
            return SignalStrength.WEAK

        # 计算加权平均置信度
        total_weight = sum(signal.weight for signal in signals)
        if total_weight == 0:
            return SignalStrength.WEAK

        weighted_confidence = sum(signal.metadata.confidence * signal.weight for signal in signals) / total_weight

        # 根据置信度映射到强度等级
        if weighted_confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif weighted_confidence >= 0.6:
            return SignalStrength.STRONG
        elif weighted_confidence >= 0.4:
            return SignalStrength.MILD
        else:
            return SignalStrength.WEAK


