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
