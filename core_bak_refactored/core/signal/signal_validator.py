"""
信号验证器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 验证信号质量、检查信号有效性
"""

from typing import Dict, List
import logging
from datetime import datetime

from .signal_models import TradingSignal, SignalType, SignalStatus

logger = logging.getLogger('DeepSeekQuant.SignalValidator')


class SignalValidator:
    """信号验证器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.signal_config = config.get('signal', {})
        self._market_state = {}  # 市场状态缓存
        self._signal_history = []  # 信号历史
    
    def validate_market_data(self, market_data: Dict) -> bool:
        """验证市场数据"""
        required_keys = ['timestamp', 'symbols', 'prices']
        
        if not all(key in market_data for key in required_keys):
            logger.warning("市场数据缺少必要字段")
            return False
        
        if not market_data['symbols'] or not market_data['prices']:
            logger.warning("市场数据为空")
            return False
        
        # 检查数据完整性
        for symbol in market_data['symbols']:
            if symbol not in market_data['prices']:
                logger.warning(f"缺少价格数据: {symbol}")
                return False
            
            price_data = market_data['prices'][symbol]
            if not isinstance(price_data, dict):
                logger.warning(f"价格数据格式错误: {symbol}")
                return False
        
        return True
    
    def validate_signals(self, signals: Dict[str, List[TradingSignal]]) -> Dict[str, List[TradingSignal]]:
        """验证信号有效性"""
        validated_signals = {}
        
        for symbol, symbol_signals in signals.items():
            valid_signals = []
            
            for signal in symbol_signals:
                try:
                    if self._validate_single_signal(signal):
                        signal.status = SignalStatus.VALIDATED
                        valid_signals.append(signal)
                    else:
                        signal.status = SignalStatus.REJECTED
                        self._record_signal(signal)
                
                except Exception as e:
                    logger.error(f"信号验证失败 {signal.id}: {e}")
                    signal.status = SignalStatus.REJECTED
                    signal.reason = f"验证错误: {str(e)}"
                    self._record_signal(signal)
            
            if valid_signals:
                validated_signals[symbol] = valid_signals
        
        return validated_signals
    
    def _validate_single_signal(self, signal: TradingSignal) -> bool:
        """验证单个信号"""
        validation_rules = self.signal_config.get('validation_rules', {})
        
        # 1. 基本验证
        if not all([signal.id, signal.symbol, signal.signal_type, signal.price > 0]):
            signal.reason = "基本验证失败"
            return False
        
        # 2. 置信度验证
        min_confidence = validation_rules.get('min_confidence', 0.3)
        if signal.metadata.confidence < min_confidence:
            signal.reason = f"置信度过低: {signal.metadata.confidence:.2f}"
            return False
        
        # 3. 风险评分验证
        max_risk_score = validation_rules.get('max_risk_score', 0.7)
        if signal.risk_score > max_risk_score:
            signal.reason = f"风险评分过高: {signal.risk_score:.2f}"
            return False
        
        # 4. 时间有效性验证
        if signal.metadata.expiration:
            try:
                expiration_time = datetime.fromisoformat(signal.metadata.expiration)
                if expiration_time < datetime.now():
                    signal.reason = "信号已过期"
                    return False
            except Exception:
                pass  # 忽略时间解析错误
        
        # 5. 信号频率验证
        max_signals = validation_rules.get('max_signals_per_period', 5)
        recent_signals = self._get_recent_signals(signal.symbol, signal.signal_type)
        if len(recent_signals) >= max_signals:
            signal.reason = f"信号频率过高: {len(recent_signals)}"
            return False
        
        # 6. 止损止盈验证
        if signal.stop_loss and signal.take_profit:
            if signal.signal_type == SignalType.BUY:
                if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
                    signal.reason = "止损止盈设置不合理"
                    return False
            elif signal.signal_type == SignalType.SELL:
                if signal.stop_loss <= signal.price or signal.take_profit >= signal.price:
                    signal.reason = "止损止盈设置不合理"
                    return False
        
        # 7. 成交量验证
        min_volume_ratio = validation_rules.get('min_volume_ratio', 0.8)
        if signal.volume_ratio < min_volume_ratio:
            signal.reason = f"成交量比率不足: {signal.volume_ratio:.2f}"
            return False
        
        # 所有验证通过
        return True
    
    def _get_recent_signals(self, symbol: str, signal_type: SignalType) -> List[TradingSignal]:
        """获取近期信号用于频率验证"""
        # 简化实现：返回空列表
        # TODO: 实现真实的信号历史查询
        return []
    
    def _record_signal(self, signal: TradingSignal):
        """记录验证结果"""
        # 简化实现：添加到历史记录
        self._signal_history.append(signal)
        
        # 限制历史记录大小
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]
