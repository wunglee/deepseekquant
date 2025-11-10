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
        params = kwargs.get('params', {})
        ts = kwargs.get('timestamp', datetime.now().isoformat())
        
        # 计算技术指标（可选）
        indicators: Dict[str, Any] = {}
        if prices_history:
            # 参数化周期
            ema_fast_p = int(params.get('ema_fast', 12))
            ema_slow_p = int(params.get('ema_slow', 26))
            indicators['sma_20'] = TechnicalIndicators.sma(prices_history, int(params.get('sma_period', 20)))
            indicators['ema_20'] = TechnicalIndicators.ema(prices_history, int(params.get('ema_period', 20)))
            indicators['rsi_14'] = TechnicalIndicators.rsi(prices_history, int(params.get('rsi_period', 14)))
            indicators['macd'] = TechnicalIndicators.macd(prices_history)
            indicators['bb'] = TechnicalIndicators.bollinger_bands(prices_history)
            # 双均线
            ema_fast = TechnicalIndicators.ema(prices_history, ema_fast_p)
            ema_slow = TechnicalIndicators.ema(prices_history, ema_slow_p)
            indicators['ema_fast'] = ema_fast
            indicators['ema_slow'] = ema_slow
        
        # 简易信号：基于 RSI 判断超买/超卖
        signal_type = SignalType.HOLD
        reason = "No clear signal"
        details: list[str] = []
        rsi_val = indicators.get('rsi_14')
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        buy_thr = float(params.get('rsi_buy', 30))
        sell_thr = float(params.get('rsi_sell', 70))
        if rsi_val is not None:
            if rsi_val < buy_thr:
                signal_type = SignalType.BUY
                reason = f"RSI oversold: {rsi_val:.2f} < {buy_thr:.2f}"
                details.append(f"RSI={rsi_val:.2f} below rsi_buy={buy_thr:.2f}")
            elif rsi_val > sell_thr:
                signal_type = SignalType.SELL
                reason = f"RSI overbought: {rsi_val:.2f} > {sell_thr:.2f}"
                details.append(f"RSI={rsi_val:.2f} above rsi_sell={sell_thr:.2f}")
        # 次级规则：双均线金叉/死叉（当RSI未触发时）
        if signal_type == SignalType.HOLD and (ema_fast is not None and ema_slow is not None):
            if ema_fast > ema_slow:
                signal_type = SignalType.BUY
                reason = "EMA fast > EMA slow"
                details.append(f"EMA_fast={ema_fast:.4f} > EMA_slow={ema_slow:.4f}")
            elif ema_fast < ema_slow:
                signal_type = SignalType.SELL
                reason = "EMA fast < EMA slow"
                details.append(f"EMA_fast={ema_fast:.4f} < EMA_slow={ema_slow:.4f}")
        
        # 组合信号评分（可选）
        weight_rsi = float(params.get('weight_rsi', 0.5))
        weight_ema = float(params.get('weight_ema', 0.5))
        composite_buy_thr = float(params.get('composite_buy_thr', 0.5))
        composite_sell_thr = float(params.get('composite_sell_thr', 0.5))
        rsi_signal = 0.0
        if rsi_val is not None:
            if rsi_val < buy_thr:
                rsi_signal = (buy_thr - rsi_val) / max(buy_thr, 1e-9)
            elif rsi_val > sell_thr:
                rsi_signal = - (rsi_val - sell_thr) / max(sell_thr, 1e-9)
        ema_signal = 0.0
        if ema_fast is not None and ema_slow is not None:
            ema_signal = 1.0 if ema_fast > ema_slow else (-1.0 if ema_fast < ema_slow else 0.0)
        composite_score = round(weight_rsi * rsi_signal + weight_ema * ema_signal, 6)
        indicators['composite_score'] = composite_score
        # 若启用组合阈值则根据score决定信号
        if params.get('use_composite', False):
            if composite_score >= composite_buy_thr:
                signal_type = SignalType.BUY
                reason = f"Composite score {composite_score:.3f} >= {composite_buy_thr:.3f}"
                details.append(f"Composite={composite_score:.3f} >= composite_buy_thr={composite_buy_thr:.3f}")
            elif composite_score <= -composite_sell_thr:
                signal_type = SignalType.SELL
                reason = f"Composite score {composite_score:.3f} <= -{composite_sell_thr:.3f}"
                details.append(f"Composite={composite_score:.3f} <= -composite_sell_thr={composite_sell_thr:.3f}")
        
        if details:
            try:
                self.logger.info("信号触发详情: " + "; ".join(details))
            except Exception:
                pass
        
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
