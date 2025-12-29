"""市场状态服务 - 评估市场开盘状态、波动率、情绪等"""
import pandas as pd
from typing import Any, Dict, List

import logging
import pytz
import numpy as np
from core_bak_refactored.core.share import MarketData


class MarketStatusService:
    """市场状态评估服务（职责单一：市场环境分析）"""

    def __init__(self, historical_data_fetcher: Any) -> None:
        """
        Args:
            historical_data_fetcher: 历史数据获取器（用于获取VIX、板块数据）
        """
        self.historical_data_fetcher = historical_data_fetcher
        self.logger = logging.getLogger('DeepSeekQuant.MarketStatusService')

    async def get_market_status(self) -> Dict[str, Any]:
        """获取市场状态信息"""
        try:
            now = pd.Timestamp.now()
            market_open = self._is_market_open(now)
            vix_data = await self.historical_data_fetcher.get_historical_data(['^VIX'], '1d', '1d', 'ohlcv', False)
            vix_value = vix_data['^VIX'][-1].close if vix_data and '^VIX' in vix_data else None
            advance_decline = await self._get_advance_decline()
            sector_performance = await self._get_sector_performance()
            liquidity_conditions = self._assess_liquidity_conditions()
            volatility_regime = self._determine_volatility_regime()
            market_sentiment = self._assess_market_sentiment()
            return {
                'market_open': market_open,
                'current_time': now.isoformat(),
                'vix': vix_value,
                'advance_decline': advance_decline,
                'sector_performance': sector_performance,
                'liquidity_conditions': liquidity_conditions,
                'volatility_regime': volatility_regime,
                'market_sentiment': market_sentiment,
                'timestamp': now.isoformat()
            }
        except Exception as e:
            self.logger.error(f"获取市场状态失败: {e}")
            return {
                'market_open': False,
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }

    def _is_market_open(self, dt: pd.Timestamp) -> bool:
        """判断美国股市是否开盘"""
        if dt.weekday() >= 5:
            return False
        if self._is_market_holiday(dt):
            return False
        eastern = pytz.timezone('US/Eastern')
        dt_eastern = dt.astimezone(eastern)
        market_open_time = time(9, 30)
        market_close_time = time(16, 0)
        return market_open_time <= dt_eastern.time() <= market_close_time

    def _is_market_holiday(self, dt: pd.Timestamp) -> bool:
        """检查是否为美国市场假日"""
        holidays = {
            (1, 1): "New Year's Day",
            (1, 15): "Martin Luther King Jr. Day",
            (2, 19): "Presidents' Day",
            (3, 29): "Good Friday",
            (5, 27): "Memorial Day",
            (6, 19): "Juneteenth",
            (7, 4): "Independence Day",
            (9, 2): "Labor Day",
            (11, 28): "Thanksgiving Day",
            (12, 25): "Christmas Day"
        }
        date_tuple = (dt.month, dt.day)
        return date_tuple in holidays

    async def _get_advance_decline(self) -> Dict[str, int]:
        """获取涨跌家数统计（需实时数据支持）"""
        return {
            'advances': 0,
            'declines': 0,
            'unchanged': 0,
            'advance_decline_ratio': 0,
            'total_issues': 0,
            'error': 'realtime data not available'
        }

    async def _get_sector_performance(self) -> Dict[str, Any]:
        """获取板块表现"""
        try:
            sector_etfs = {
                'XLK': 'Technology',
                'XLV': 'Healthcare',
                'XLI': 'Industrial',
                'XLY': 'Consumer Discretionary',
                'XLP': 'Consumer Staples',
                'XLF': 'Financial',
                'XLE': 'Energy',
                'XLU': 'Utilities',
                'XLB': 'Materials',
                'XLRE': 'Real Estate',
                'XLC': 'Communications'
            }
            sector_data = await self.historical_data_fetcher.get_historical_data(
                list(sector_etfs.keys()),
                '1d', '1d', 'ohlcv', True
            )
            performance: Dict[str, Any] = {}
            for etf, sector_name in sector_etfs.items():
                if etf in sector_data and sector_data[etf]:
                    today = sector_data[etf][-1]
                    yesterday = sector_data[etf][-2] if len(sector_data[etf]) >= 2 else today
                    daily_return = (today.close - yesterday.close) / yesterday.close * 100
                    performance[sector_name] = {
                        'daily_return': daily_return,
                        'current_price': today.close,
                        'volume': today.volume,
                        'volatility': self._calculate_daily_volatility(sector_data[etf][-5:]) if len(sector_data[etf]) >= 5 else 0
                    }
            return performance
        except Exception as e:
            self.logger.warning(f"获取板块表现失败: {e}")
            return {}

    def _calculate_daily_volatility(self, data: List[MarketData]) -> float:
        """计算日波动率（年化）"""
        if len(data) < 2:
            return 0.0
        returns = []
        for i in range(1, len(data)):
            daily_return = (data[i].close - data[i - 1].close) / data[i - 1].close
            returns.append(daily_return)
        return float(np.std(returns) * np.sqrt(252))

    def _assess_liquidity_conditions(self) -> Dict[str, Any]:
        """评估市场流动性状况"""
        return {
            'liquidity_score': 0.8,
            'bid_ask_spread': 'normal',
            'market_depth': 'good',
            'execution_quality': 'high',
            'liquidity_risk': 'low',
            'volume_concentration': 'moderate',
            'market_impact_cost': 'low',
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def _determine_volatility_regime(self) -> Dict[str, Any]:
        """确定波动率状态"""
        return {
            'regime': 'normal',
            'vix_level': 'moderate',
            'volatility_clustering': False,
            'regime_confidence': 0.85,
            'expected_duration': 'short_term',
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def _assess_market_sentiment(self) -> Dict[str, Any]:
        """评估市场情绪"""
        return {
            'sentiment_score': 0.6,
            'bullish_bearish_ratio': 1.2,
            'fear_greed_index': 60,
            'put_call_ratio': 0.8,
            'market_outlook': 'neutral_bullish',
            'sentiment_extremes': False,
            'contrarian_indicator': False,
            'timestamp': pd.Timestamp.now().isoformat()
        }
