"""
信号验证器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 验证信号质量、检查信号有效性
"""

from typing import Dict, List, Optional
import logging

from .signal_models import TradingSignal

logger = logging.getLogger('DeepSeekQuant.SignalValidator')


class SignalValidator:
    """信号验证器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
        """验证市场数据"""
        required_keys = ['timestamp', 'symbols', 'prices', 'volumes']

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
            if not all(key in price_data for key in ['open', 'high', 'low', 'close']):
                logger.warning(f"价格数据不完整: {symbol}")
                return False

        return True


        """验证信号有效性"""
        validated_signals = {}

        for symbol, symbol_signals in signals.items():
            valid_signals = []

            for signal in symbol_signals:
                try:
                    if self._validate_single_signal(signal):
                        valid_signals.append(signal)
                    else:
                        signal.status = SignalStatus.REJECTED
                        self._record_signal(signal)  # 记录被拒绝的信号

                except Exception as e:
                    logger.error(f"信号验证失败 {signal.id}: {e}")
                    signal.status = SignalStatus.REJECTED
                    signal.reason = f"验证错误: {str(e)}"
                    self._record_signal(signal)

            if valid_signals:
                validated_signals[symbol] = valid_signals

        return validated_signals


        """验证单个信号"""
        validation_rules = self.signal_config.get('validation_rules', {})

        # 1. 基本验证
        if not all([signal.id, signal.symbol, signal.signal_type, signal.price > 0]):
            signal.reason = "基本验证失败"
            return False

        # 2. 价格合理性验证
        current_price = self._market_state[signal.symbol]['price']
        price_deviation = abs(signal.price - current_price) / current_price

        if price_deviation > validation_rules.get('max_price_deviation', 0.1):
            signal.reason = f"价格偏差过大: {price_deviation:.2%}"
            return False

        # 3. 波动率验证
        volatility = self._market_state[signal.symbol]['volatility']
        if volatility > validation_rules.get('max_volatility', 0.5):
            signal.reason = f"波动率过高: {volatility:.2%}"
            return False

        # 4. 流动性验证
        liquidity = self._market_state[signal.symbol]['liquidity']
        if liquidity < validation_rules.get('min_liquidity', 0.1):
            signal.reason = f"流动性不足: {liquidity:.2f}"
            return False

        # 5. 置信度验证
        if signal.metadata.confidence < validation_rules.get('min_confidence', 0.3):
            signal.reason = f"置信度过低: {signal.metadata.confidence:.2f}"
            return False

        # 6. 风险评分验证
        if signal.risk_score > validation_rules.get('max_risk_score', 0.7):
            signal.reason = f"风险评分过高: {signal.risk_score:.2f}"
            return False

        # 7. 时间有效性验证
        if signal.metadata.expiration and datetime.fromisoformat(signal.metadata.expiration) < datetime.now():
            signal.reason = "信号已过期"
            return False

        # 8. 信号频率验证
        recent_signals = self._get_recent_signals(signal.symbol, signal.signal_type)
        if len(recent_signals) >= validation_rules.get('max_signals_per_period', 5):
            signal.reason = f"信号频率过高: {len(recent_signals)}"
            return False

        # 9. 止损止盈验证
        if signal.stop_loss and signal.take_profit:
            if signal.signal_type == SignalType.BUY:
                if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
                    signal.reason = "止损止盈设置不合理"
                    return False
            elif signal.signal_type == SignalType.SELL:
                if signal.stop_loss <= signal.price or signal.take_profit >= signal.price:
                    signal.reason = "止损止盈设置不合理"
                    return False

        # 10. 成交量验证
        if signal.volume_ratio < validation_rules.get('min_volume_ratio', 0.8):
            signal.reason = f"成交量比率不足: {signal.volume_ratio:.2f}"
            return False

        # 11. 市场状态验证
        market_trend = self._market_state[signal.symbol]['trend']
        if (signal.signal_type == SignalType.BUY and market_trend == 'downtrend' and
                not validation_rules.get('allow_counter_trend', False)):
            signal.reason = "不允许逆势交易"
            return False

        # 12. 相关性验证
        if signal.correlation:
            max_correlation = max(signal.correlation.values())
            if max_correlation > validation_rules.get('max_correlation', 0.9):
                signal.reason = f"相关性过高: {max_correlation:.2f}"
                return False

        # 所有验证通过
        signal.status = SignalStatus.VALIDATED
        return True


