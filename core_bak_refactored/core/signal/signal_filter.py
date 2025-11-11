"""
信号过滤器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 过滤和筛选信号
"""

from typing import Dict, List
import logging
from datetime import datetime

from .signal_models import TradingSignal, SignalStrength

logger = logging.getLogger('DeepSeekQuant.SignalFilter')


class SignalFilter:
    """信号过滤器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def filter_signals(self, signals: Dict[str, List[TradingSignal]]) -> Dict[str, List[TradingSignal]]:
        """过滤和筛选信号"""
        filtered_signals = {}
        
        for symbol, symbol_signals in signals.items():
            valid_signals = []
            
            for signal in symbol_signals:
                if self._should_keep_signal(signal):
                    valid_signals.append(signal)
            
            # 按优先级排序并限制数量
            valid_signals.sort(
                key=lambda s: (s.metadata.priority, s.metadata.confidence), 
                reverse=True
            )
            max_signals = self.config.get('max_signals_per_symbol', 5)
            filtered_signals[symbol] = valid_signals[:max_signals]
        
        return filtered_signals
    
    def _should_keep_signal(self, signal: TradingSignal) -> bool:
        """判断是否保留信号"""
        filters = self.config.get('filters', {})
        
        # 1. 置信度过滤
        min_confidence = filters.get('min_confidence', 0.4)
        if signal.metadata.confidence < min_confidence:
            return False
        
        # 2. 强度过滤
        min_strength = filters.get('min_strength', SignalStrength.MILD)
        if isinstance(min_strength, str):
            min_strength = SignalStrength(min_strength)
        # 比较枚举值
        strength_order = {
            SignalStrength.WEAK: 1,
            SignalStrength.MILD: 2,
            SignalStrength.STRONG: 3,
            SignalStrength.VERY_STRONG: 4,
            SignalStrength.EXTREME: 5
        }
        if strength_order.get(signal.metadata.strength, 0) < strength_order.get(min_strength, 0):
            return False
        
        # 3. 风险过滤
        max_risk_score = filters.get('max_risk_score', 0.6)
        if signal.risk_score > max_risk_score:
            return False
        
        # 4. 时间过滤（过期信号）
        if signal.metadata.expiration:
            try:
                expiration_time = datetime.fromisoformat(signal.metadata.expiration)
                if expiration_time < datetime.now():
                    return False
            except Exception:
                pass  # 忽略时间解析错误
        
        # 5. 来源过滤
        excluded_sources = filters.get('excluded_sources', [])
        if signal.metadata.source in excluded_sources:
            return False
        
        return True
