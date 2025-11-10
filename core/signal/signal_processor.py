from typing import Any, Dict
from datetime import datetime
from core.base_processor import BaseProcessor
from core.signal.indicators import TechnicalIndicators
from common import TradingSignal, SignalType, SignalStatus, SignalMetadata

class SignalProcessor(BaseProcessor):
    def _initialize_core(self) -> bool:
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        symbol = kwargs.get('symbol', 'TEST')
        price = kwargs.get('price', 0.0)
        prices_history = kwargs.get('prices', [])
        ts = kwargs.get('timestamp', datetime.now().isoformat())
        
        # 计算技术指标（可选）
        indicators: Dict[str, Any] = {}
        if prices_history:
            indicators['sma_20'] = TechnicalIndicators.sma(prices_history, 20)
            indicators['ema_20'] = TechnicalIndicators.ema(prices_history, 20)
            indicators['rsi_14'] = TechnicalIndicators.rsi(prices_history, 14)
            indicators['macd'] = TechnicalIndicators.macd(prices_history)
            indicators['bb'] = TechnicalIndicators.bollinger_bands(prices_history)
            # 双均线
            ema_fast = TechnicalIndicators.ema(prices_history, 12)
            ema_slow = TechnicalIndicators.ema(prices_history, 26)
            indicators['ema_fast_12'] = ema_fast
            indicators['ema_slow_26'] = ema_slow
        
        # 简易信号：基于 RSI 判断超买/超卖
        signal_type = SignalType.HOLD
        reason = "No clear signal"
        rsi_val = indicators.get('rsi_14')
        ema_fast = indicators.get('ema_fast_12')
        ema_slow = indicators.get('ema_slow_26')
        if rsi_val is not None:
            if rsi_val < 30:
                signal_type = SignalType.BUY
                reason = f"RSI oversold: {rsi_val:.2f}"
            elif rsi_val > 70:
                signal_type = SignalType.SELL
                reason = f"RSI overbought: {rsi_val:.2f}"
        # 次级规则：双均线金叉/死叉
        if signal_type == SignalType.HOLD and (ema_fast is not None and ema_slow is not None):
            if ema_fast > ema_slow:
                signal_type = SignalType.BUY
                reason = "EMA12 > EMA26 crossover"
            elif ema_fast < ema_slow:
                signal_type = SignalType.SELL
                reason = "EMA12 < EMA26 crossover"
        
        metadata = SignalMetadata(generated_at=ts, parameters=indicators)
        signal = TradingSignal(
            id=f"{symbol}-{ts}",
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            timestamp=ts,
            metadata=metadata,
            reason=reason
        )
        return {'status': 'success', 'signal': signal.to_dict(), 'indicators': indicators}

    def _cleanup_core(self):
        pass
