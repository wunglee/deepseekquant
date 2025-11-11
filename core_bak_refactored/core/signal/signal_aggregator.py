"""
信号聚合器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 聚合多个信号源、计算综合信号强度
"""

import numpy as np
from typing import Dict, List
import logging

from .signal_models import TradingSignal, SignalStrength

logger = logging.getLogger('DeepSeekQuant.SignalAggregator')


class SignalAggregator:
    """信号聚合器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def aggregate(self, signals: Dict[str, List[TradingSignal]]) -> Dict[str, List[TradingSignal]]:
        """聚合多个信号源"""
        return signals
    
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


