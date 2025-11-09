from typing import Any, Dict
from core.base_processor import BaseProcessor
from common import TradingSignal, SignalType, SignalStatus, SignalMetadata

class SignalProcessor(BaseProcessor):
    def _initialize_core(self) -> bool:
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        symbol = kwargs.get('symbol', 'TEST')
        price = kwargs.get('price', 0.0)
        ts = kwargs.get('timestamp', getattr(self, 'startup_time', ''))
        metadata = SignalMetadata()
        signal = TradingSignal(
            id=f"{symbol}-{ts}",
            symbol=symbol,
            signal_type=SignalType.BUY,
            price=price,
            timestamp=ts,
            metadata=metadata
        )
        return {'status': 'success', 'signal': signal.to_dict()}

    def _cleanup_core(self):
        pass
