from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from infrastructure.interfaces import InfrastructureProvider

@dataclass
class MarketData:
    symbol: str
    price: float
    timestamp: str
    source: str = "primary"
    extra: Dict[str, Any] = None

class DataValidator:
    def validate(self, data: MarketData) -> Dict[str, Any]:
        issues: List[str] = []
        if not data.symbol:
            issues.append("missing symbol")
        if data.price is None:
            issues.append("missing price")
        return {"is_valid": len(issues) == 0, "issues": issues}

class DataFetcher:
    def __init__(self):
        self.logger = InfrastructureProvider.get('logging').get_logger('DeepSeekQuant.DataFetcher')
        self.cache = InfrastructureProvider.get('cache')
        self.event_bus = InfrastructureProvider.get('event_bus')
        self.validator = DataValidator()

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        # 简化实现：优先读缓存，否则生成模拟数据
        cached = None
        try:
            cached = self.cache.process(op='get', key=f"md:{symbol}")
        except Exception:
            cached = None
        if cached and cached.get('value'):
            md = cached['value']
            self.logger.info(f"缓存命中: {symbol}")
            return {"status": "success", "data": md}

        md = MarketData(symbol=symbol, price=0.0, timestamp=datetime.now().isoformat())
        result = self.validator.validate(md)
        if not result["is_valid"]:
            self.logger.warning(f"数据校验失败: {result['issues']}")
            return {"status": "error", "issues": result['issues']}

        # 写入缓存并发布事件
        try:
            self.cache.process(op='set', key=f"md:{symbol}", value=md.__dict__)
        except Exception:
            pass
        try:
            self.event_bus.publish('market_data', md.__dict__)
        except Exception:
            pass
        return {"status": "success", "data": md.__dict__}
