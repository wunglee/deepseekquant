"""
技术指标业务层封装

职责：将基础设施层的纯数学算法映射为量化交易领域的具体指标
- 定义业务概念（MACD、RSI、布林带等金融术语）
- 管理市场参数（A股、美股不同参数）
- 提供业务友好的接口

架构原则：
- 依赖基础设施层的TimeSeriesCalculator
- 只包含业务逻辑，不包含算法实现
- 参数配置化，支持不同市场和策略
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from typing import Tuple, Dict, Any
from infrastructure.technical_indicators import TimeSeriesCalculator


# 市场参数配置（业务概念）
MARKET_PARAMS = {
    'CN': {  # A股市场参数
        'macd': {'fast': 8, 'slow': 21, 'signal': 8},
        'rsi': {'period': 14},
        'bollinger': {'period': 20, 'std': 2.0},
        'atr': {'period': 14},
        'stochastic': {'period': 14, 'smooth': 3},
    },
    'US': {  # 美股市场参数
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi': {'period': 14},
        'bollinger': {'period': 20, 'std': 2.0},
        'atr': {'period': 14},
        'stochastic': {'period': 14, 'smooth': 3},
    }
}


class TechnicalIndicators:
    """
    技术指标业务层
    
    将基础设施层的通用算法映射为量化交易领域的具体指标
    """
    
    def __init__(self, market: str = 'CN'):
        """
        Args:
            market: 市场类型 'CN'(A股) 或 'US'(美股)
        """
        self.market = market
        self.params = MARKET_PARAMS.get(market, MARKET_PARAMS['US'])
        self.calculator = TimeSeriesCalculator
    
    def calculate_macd(self, 
                      prices: pd.Series,
                      fast: int = None,
                      slow: int = None,
                      signal: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算MACD指标（Moving Average Convergence Divergence）
        
        业务含义：
        - 趋势跟踪动量指标
        - MACD线穿越信号线产生买卖信号
        - 柱状图表示两线差值
        
        Args:
            prices: 价格序列
            fast: 快线周期（默认使用市场参数）
            slow: 慢线周期（默认使用市场参数）
            signal: 信号线周期（默认使用市场参数）
        
        Returns:
            (macd_line, signal_line, histogram)
        """
        params = self.params['macd']
        fast = fast or params['fast']
        slow = slow or params['slow']
        signal = signal or params['signal']
        
        return self.calculator.calculate_dual_ema_oscillator(
            prices, fast, slow, signal
        )
    
    def calculate_rsi(self, 
                     prices: pd.Series,
                     period: int = None) -> pd.Series:
        """
        计算RSI指标（Relative Strength Index）
        
        业务含义：
        - 相对强弱指标，范围0-100
        - RSI > 70 表示超买（可能回调）
        - RSI < 30 表示超卖（可能反弹）
        
        Args:
            prices: 价格序列
            period: 周期（默认使用市场参数）
        
        Returns:
            RSI值序列（0-100）
        """
        period = period or self.params['rsi']['period']
        return self.calculator.calculate_momentum_index(prices, period)
    
    def calculate_bollinger_bands(self,
                                  prices: pd.Series,
                                  period: int = None,
                                  std_multiplier: float = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算布林带（Bollinger Bands）
        
        业务含义：
        - 价格波动通道
        - 价格触及上轨可能超买
        - 价格触及下轨可能超卖
        - 通道收窄预示波动率降低
        
        Args:
            prices: 价格序列
            period: 周期（默认使用市场参数）
            std_multiplier: 标准差倍数（默认使用市场参数）
        
        Returns:
            (upper_band, middle_band, lower_band)
        """
        params = self.params['bollinger']
        period = period or params['period']
        std_multiplier = std_multiplier or params['std']
        
        return self.calculator.calculate_volatility_bands(
            prices, period, std_multiplier
        )
    
    def calculate_atr(self,
                     high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = None) -> pd.Series:
        """
        计算ATR指标（Average True Range）
        
        业务含义：
        - 平均真实波动幅度
        - 用于设置止损位
        - 用于仓位管理（波动越大仓位越小）
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期（默认使用市场参数）
        
        Returns:
            ATR值序列
        """
        period = period or self.params['atr']['period']
        previous_close = close.shift()
        
        return self.calculator.calculate_true_range_average(
            high, low, previous_close, period
        )
    
    def calculate_kdj(self,
                     high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = None,
                     smooth: int = None) -> Tuple[pd.Series, pd.Series]:
        """
        计算KDJ指标（Stochastic Oscillator）
        
        业务含义：
        - 随机指标，反映价格在区间中的相对位置
        - K值 > 80 超买，< 20 超卖
        - D值是K值的平滑
        - K线穿越D线产生信号
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期（默认使用市场参数）
            smooth: 平滑周期（默认使用市场参数）
        
        Returns:
            (k_line, d_line)
        """
        params = self.params['stochastic']
        period = period or params['period']
        smooth = smooth or params['smooth']
        
        return self.calculator.calculate_range_position(
            high, low, close, period, smooth
        )
    
    def calculate_obv(self,
                     close: pd.Series,
                     volume: pd.Series) -> pd.Series:
        """
        计算OBV指标（On-Balance Volume）
        
        业务含义：
        - 能量潮指标
        - 价格上涨累加成交量，下跌累减
        - 确认价格趋势的有效性
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        
        Returns:
            OBV值序列
        """
        return self.calculator.calculate_directional_volume(close, volume)
    
    def calculate_adx(self,
                     high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算ADX指标（Average Directional Index）
        
        业务含义：
        - 平均趋向指数，衡量趋势强度
        - +DI > -DI 表示上升趋势
        - +DI < -DI 表示下降趋势
        - ADX > 25 表示强趋势，< 20 表示无趋势
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期（默认14）
        
        Returns:
            (+DI, -DI, ADX)
        """
        period = period or 14
        return self.calculator.calculate_directional_indicators(
            high, low, close, period
        )
    
    # 便捷方法：计算所有常用指标
    def calculate_all_indicators(self, 
                                data: pd.DataFrame) -> Dict[str, Any]:
        """
        批量计算所有常用指标
        
        Args:
            data: OHLCV数据，包含列['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            包含所有指标的字典
        """
        results = {}
        
        # 趋势指标
        macd, signal, hist = self.calculate_macd(data['close'])
        results['macd'] = macd
        results['macd_signal'] = signal
        results['macd_hist'] = hist
        
        # 动量指标
        results['rsi'] = self.calculate_rsi(data['close'])
        
        # 波动率指标
        upper, middle, lower = self.calculate_bollinger_bands(data['close'])
        results['bb_upper'] = upper
        results['bb_middle'] = middle
        results['bb_lower'] = lower
        
        results['atr'] = self.calculate_atr(
            data['high'], data['low'], data['close']
        )
        
        # 随机指标
        k, d = self.calculate_kdj(
            data['high'], data['low'], data['close']
        )
        results['kdj_k'] = k
        results['kdj_d'] = d
        
        # 成交量指标
        if 'volume' in data.columns:
            results['obv'] = self.calculate_obv(
                data['close'], data['volume']
            )
        
        # 趋势强度
        plus_di, minus_di, adx = self.calculate_adx(
            data['high'], data['low'], data['close']
        )
        results['adx_plus'] = plus_di
        results['adx_minus'] = minus_di
        results['adx'] = adx
        
        return results
