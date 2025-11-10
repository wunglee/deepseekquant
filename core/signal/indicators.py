"""
技术指标计算模块
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class IndicatorResult:
    name: str
    value: float
    timestamp: str
    metadata: Dict[str, Any]

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def sma(prices: List[float], period: int = 20) -> Optional[float]:
        """简单移动平均线"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    @staticmethod
    def ema(prices: List[float], period: int = 20) -> Optional[float]:
        """指数移动平均线"""
        if len(prices) < period:
            return None
        
        # 计算初始 SMA
        sma = sum(prices[:period]) / period
        multiplier = 2 / (period + 1)
        ema_val = sma
        
        # 计算 EMA
        for price in prices[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        
        return ema_val
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """相对强弱指标"""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))
        
        return rsi_val
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, float]]:
        """MACD 指标"""
        if len(prices) < slow:
            return None
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # 简化：signal line 使用固定值
        signal_line = macd_line * 0.9
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
        """布林带"""
        if len(prices) < period:
            return None
        
        sma = TechnicalIndicators.sma(prices, period)
        if sma is None:
            return None
        
        # 计算标准差
        recent_prices = prices[-period:]
        variance = sum((p - sma) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """平均真实波幅"""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        return sum(true_ranges[-period:]) / period
    
    @staticmethod
    def momentum(prices: List[float], period: int = 10) -> Optional[float]:
        """动量指标"""
        if len(prices) < period + 1:
            return None
        return prices[-1] - prices[-(period + 1)]
    
    @staticmethod
    def rate_of_change(prices: List[float], period: int = 10) -> Optional[float]:
        """变化率"""
        if len(prices) < period + 1:
            return None
        old_price = prices[-(period + 1)]
        if old_price == 0:
            return None
        return ((prices[-1] - old_price) / old_price) * 100

    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        # TODO：补充了ADX占位实现，待确认
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None
        # 简化近似：使用高低价变化的平均幅度作为趋势强度代理
        dm_plus = 0.0
        dm_minus = 0.0
        for i in range(1, len(highs)):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            dm_plus += max(up, 0)
            dm_minus += max(down, 0)
        denom = dm_plus + dm_minus if (dm_plus + dm_minus) > 0 else 1.0
        dx = abs(dm_plus - dm_minus) / denom
        return round(dx, 6)

    @staticmethod
    def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Optional[Dict[str, float]]:
        # TODO：补充了随机指标占位实现，待确认
        if len(closes) < k_period:
            return None
        recent_high = max(highs[-k_period:])
        recent_low = min(lows[-k_period:])
        denom = recent_high - recent_low
        if denom <= 0:
            return None
        k = (closes[-1] - recent_low) / denom * 100
        # 简化D为K的平滑
        d = k
        return {'%K': round(k, 3), '%D': round(d, 3)}

    @staticmethod
    def williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        # TODO：补充了Williams %R占位实现，待确认
        if len(closes) < period:
            return None
        recent_high = max(highs[-period:])
        recent_low = min(lows[-period:])
        denom = recent_high - recent_low
        if denom <= 0:
            return None
        wr = (recent_high - closes[-1]) / denom * -100
        return round(wr, 3)

    @staticmethod
    def vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
        # TODO：补充了VWAP占位实现，待确认
        if not prices or not volumes or len(prices) != len(volumes):
            return None
        total_vol = sum(volumes)
        if total_vol <= 0:
            return None
        return round(sum(p*v for p, v in zip(prices, volumes)) / total_vol, 6)

    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        # TODO：补充了枢轴点占位实现，待确认
        p = (high + low + close) / 3
        r1 = 2*p - low
        s1 = 2*p - high
        return {'pivot': round(p, 6), 'r1': round(r1, 6), 's1': round(s1, 6)}
