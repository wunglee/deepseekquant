from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from infrastructure.interfaces import InfrastructureProvider

@dataclass
class MarketData:
    symbol: str
    price: float
    timestamp: str
    source: str = "primary"
    volume: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

class DataValidator:
    def validate(self, data: MarketData) -> Dict[str, Any]:
        issues: List[str] = []
        if not data.symbol:
            issues.append("缺少必需字段: symbol")
        if data.price is None:
            issues.append("缺少必需字段: price")
        elif data.price < 0:
            issues.append(f"价格为负: {data.price}")
        return {"is_valid": len(issues) == 0, "issues": issues}

class DataFetcher:
    def __init__(self):
        self.logger = InfrastructureProvider.get('logging').get_logger('DeepSeekQuant.DataFetcher')
        self.cache = InfrastructureProvider.get('cache')
        self.event_bus = InfrastructureProvider.get('event_bus')
        self.validator = DataValidator()

    def get_market_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        # 优先读缓存
        if use_cache:
            cached = None
            try:
                cached = self.cache.process(op='get', key=f"md:{symbol}")
            except Exception:
                cached = None
            if cached and cached.get('value'):
                md = cached['value']
                self.logger.info(f"缓存命中: {symbol}")
                return {"status": "success", "data": md, "from_cache": True}

        # 模拟抓取
        md = MarketData(symbol=symbol, price=100.0, timestamp=datetime.now().isoformat())
        result = self.validator.validate(md)
        if not result["is_valid"]:
            msg = "; ".join(result['issues'])
            self.logger.error(f"数据校验失败: {msg}")
            return {"status": "error", "issues": result['issues'], "message": msg}

        # 写入缓存并发布事件
        try:
            self.cache.process(op='set', key=f"md:{symbol}", value=md.__dict__)
        except Exception:
            pass
        try:
            self.event_bus.publish('market.data', md.__dict__)
        except Exception:
            pass
        return {"status": "success", "data": md.__dict__, "from_cache": False}

    def get_history(self, symbol: str, lookback: int = 30) -> List[Dict[str, Any]]:
        """生成模拟历史OHLCV数据并缓存"""
        history_key = f"mdhist:{symbol}:{lookback}"
        cached = None
        try:
            cached = self.cache.process(op='get', key=history_key)
        except Exception:
            cached = None
        if cached and cached.get('value'):
            return cached['value']
        
        import random
        base_price = 100.0
        series: List[Dict[str, Any]] = []
        for i in range(lookback):
            ts = datetime.now().isoformat()
            change = random.uniform(-1.0, 1.5)
            open_p = base_price
            close_p = max(0.01, base_price + change)
            high_p = max(open_p, close_p) + random.uniform(0.0, 0.8)
            low_p = min(open_p, close_p) - random.uniform(0.0, 0.8)
            volume = random.uniform(1000, 10000)
            series.append({
                'timestamp': ts,
                'open': round(open_p, 4),
                'high': round(high_p, 4),
                'low': round(low_p, 4),
                'close': round(close_p, 4),
                'volume': round(volume, 2)
            })
            base_price = close_p
        try:
            self.cache.process(op='set', key=history_key, value=series)
        except Exception:
            pass
        return series

    def compute_volatility(self, closes: List[float]) -> float:
        """基于收盘价计算年化波动率（简单标准差）"""
        if not closes or len(closes) < 2:
            return 0.0
        returns = []
        for i in range(1, len(closes)):
            prev = closes[i-1]
            curr = closes[i]
            if prev <= 0:
                continue
            returns.append((curr - prev) / prev)
        if not returns:
            return 0.0
        avg = sum(returns) / len(returns)
        var = sum((r - avg) ** 2 for r in returns) / len(returns)
        import math
        std = math.sqrt(var)
        return round(std * math.sqrt(252), 6)

    def compute_pairwise_correlation(self, series_a: List[float], series_b: List[float]) -> float:
        """基于收益率的皮尔逊相关系数（简化）"""
        if not series_a or not series_b or len(series_a) != len(series_b):
            return 0.0
        returns_a = []
        returns_b = []
        for i in range(1, len(series_a)):
            pa, pb = series_a[i-1], series_b[i-1]
            ca, cb = series_a[i], series_b[i]
            if pa <= 0 or pb <= 0:
                continue
            returns_a.append((ca - pa) / pa)
            returns_b.append((cb - pb) / pb)
        if not returns_a or not returns_b or len(returns_a) != len(returns_b):
            return 0.0
        avg_a = sum(returns_a) / len(returns_a)
        avg_b = sum(returns_b) / len(returns_b)
        cov = sum((ra - avg_a) * (rb - avg_b) for ra, rb in zip(returns_a, returns_b)) / len(returns_a)
        var_a = sum((ra - avg_a) ** 2 for ra in returns_a) / len(returns_a)
        var_b = sum((rb - avg_b) ** 2 for rb in returns_b) / len(returns_b)
        import math
        denom = math.sqrt(var_a) * math.sqrt(var_b)
        if denom == 0:
            return 0.0
        return round(cov / denom, 6)

    def compute_max_drawdown(self, closes: List[float]) -> float:
        """计算最大回撤（正数，表示比例）"""
        if not closes:
            return 0.0
        peak = closes[0]
        max_dd = 0.0
        for c in closes:
            if c > peak:
                peak = c
            dd = (peak - c) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 6)
