"""
板块表现模块（从 DataFetcher._get_sector_performance 和 _calculate_daily_volatility 迁移而来）

职责：
1. 获取板块表现数据
2. 计算板块日收益
3. 计算板块波动率
4. 提供板块对比分析
"""
from typing import Dict, List, Any
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def get_sector_performance(fetcher: Any) -> Dict[str, Dict[str, Any]]:
    """
    获取板块表现数据（从 DataFetcher._get_sector_performance 迁移而来）。
    
    使用板块ETF作为板块表现的代理指标。
    
    Args:
        fetcher: DataFetcher实例，包含get_historical_data方法
    
    Returns:
        板块表现字典，键为板块名称，值包含：
        - daily_return: 日收益率(%)
        - current_price: 当前价格
        - volume: 交易量
        - volatility: 年化波动率
    
    Example:
        >>> performance = await get_sector_performance(fetcher)
        >>> # {'Technology': {'daily_return': 1.5, 'current_price': 150.2, ...}, ...}
    """
    try:
        # 板块ETF映射
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

        # 获取板块ETF历史数据（最近2天，用于计算日收益）
        sector_data = await fetcher.get_historical_data(
            list(sector_etfs.keys()),
            period='5d',  # 获取5天数据用于波动率计算
            interval='1d',
            data_type='ohlcv',
            adjustments=True
        )

        performance = {}
        
        for etf, sector_name in sector_etfs.items():
            if etf not in sector_data or not sector_data[etf]:
                logger.warning(f"未获取到板块ETF {etf} ({sector_name}) 的数据")
                continue

            data_list = sector_data[etf]
            
            if len(data_list) < 1:
                continue

            # 获取最新和前一天的数据
            today = data_list[-1]
            yesterday = data_list[-2] if len(data_list) >= 2 else today

            # 计算日收益率
            if hasattr(today, 'close') and hasattr(yesterday, 'close'):
                today_close = today.close
                yesterday_close = yesterday.close
            elif isinstance(today, dict) and isinstance(yesterday, dict):
                today_close = today.get('close', 0)
                yesterday_close = yesterday.get('close', 0)
            else:
                logger.warning(f"无法解析 {etf} 的收盘价数据")
                continue

            if yesterday_close > 0:
                daily_return = (today_close - yesterday_close) / yesterday_close * 100
            else:
                daily_return = 0

            # 提取volume
            if hasattr(today, 'volume'):
                volume = today.volume
            elif isinstance(today, dict):
                volume = today.get('volume', 0)
            else:
                volume = 0

            # 计算波动率（需要至少5天数据）
            if len(data_list) >= 5:
                volatility = calculate_daily_volatility(data_list[-5:])
            else:
                volatility = 0

            # 构造板块表现数据
            performance[sector_name] = {
                'daily_return': daily_return,
                'current_price': today_close,
                'volume': volume,
                'volatility': volatility,
                'etf': etf
            }

        logger.info(f"成功获取 {len(performance)} 个板块的表现数据")
        return performance

    except Exception as e:
        logger.warning(f"获取板块表现失败: {e}")
        return {}


def calculate_daily_volatility(data: List[Any]) -> float:
    """
    计算日波动率（从 DataFetcher._calculate_daily_volatility 迁移而来）。
    
    使用对数收益率计算历史波动率，并年化（假设252个交易日）。
    
    Args:
        data: MarketData对象列表或数据字典列表
    
    Returns:
        年化波动率（小数形式，如0.25表示25%）
    
    Example:
        >>> volatility = calculate_daily_volatility(market_data_list[-20:])
        >>> # 0.18  # 18%年化波动率
    """
    if len(data) < 2:
        logger.warning("数据不足，无法计算波动率（至少需要2个数据点）")
        return 0.0

    try:
        # 提取收盘价
        closes = []
        for item in data:
            if hasattr(item, 'close'):
                closes.append(item.close)
            elif isinstance(item, dict):
                closes.append(item.get('close', 0))
            else:
                logger.warning(f"无法从数据中提取收盘价: {type(item)}")
                return 0.0

        if len(closes) < 2:
            return 0.0

        # 计算对数收益率
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                daily_return = (closes[i] - closes[i - 1]) / closes[i - 1]
                returns.append(daily_return)

        if len(returns) < 1:
            return 0.0

        # 计算标准差
        std_dev = np.std(returns)

        # 年化（假设252个交易日）
        annualized_volatility = std_dev * np.sqrt(252)

        return annualized_volatility

    except Exception as e:
        logger.error(f"计算波动率失败: {e}")
        return 0.0
