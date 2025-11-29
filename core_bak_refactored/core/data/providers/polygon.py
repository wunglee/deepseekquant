"""
Polygon.io数据提供者

职责：
1. 从Polygon.io获取历史OHLCV数据
2. 支持股票、期权、外汇等多种资产类型
3. 处理API认证和限流
4. 支持聚合数据（Aggregates）API
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urlencode
import logging

logger = logging.getLogger(__name__)


async def fetch_polygon_data(
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[List[Dict]]:
    """
    从Polygon.io获取数据。
    
    Args:
        symbol: 股票代码
        period: 数据期间
        interval: 数据间隔
        data_type: 数据类型
        adjustments: 是否调整价格
        api_credentials: API凭证
        aiohttp_session: aiohttp会话
    
    Returns:
        数据字典列表，失败返回None
    """
    try:
        # 验证API密钥
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("Polygon.io API密钥未配置")
        
        # 映射时间间隔
        multiplier, timespan = _map_interval_to_polygon(interval)
        
        # 计算时间范围
        end_date = datetime.now()
        start_date = _calculate_start_date(end_date, period)
        
        # 构建API URL
        base_url = api_credentials.get('base_url', 'https://api.polygon.io')
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            'adjusted': 'true' if adjustments else 'false',
            'sort': 'asc',
            'limit': 50000,
            'apiKey': api_credentials['api_key']
        }
        
        url = f"{base_url}{endpoint}?{urlencode(params)}"
        
        # 发送HTTP请求
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"API请求失败，HTTP状态码: {response.status}")
            
            data = await response.json()
            
            # 检查响应状态
            if data.get('status') != 'OK':
                error_msg = data.get('error', '未知错误')
                raise ValueError(f"API返回错误: {error_msg}")
            
            # 解析结果
            results = data.get('results', [])
            if not results:
                logger.warning(f"Polygon.io未返回 {symbol} 的数据")
                return None
            
            # 转换为标准格式
            market_data_list = []
            for item in results:
                # Polygon返回时间戳是毫秒
                timestamp = datetime.fromtimestamp(item['t'] / 1000)
                
                data_point = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(item['o']),
                    'high': float(item['h']),
                    'low': float(item['l']),
                    'close': float(item['c']),
                    'volume': float(item['v']),
                    'vwap': float(item.get('vw', 0)),  # 成交量加权平均价
                    'trades': int(item.get('n', 0)),  # 交易笔数
                    'metadata': {
                        'data_source': 'polygon',
                        'data_type': data_type,
                        'interval': interval,
                        'adjustments': adjustments
                    }
                }
                market_data_list.append(data_point)
            
            logger.info(f"Polygon.io成功获取 {symbol} 数据，共 {len(market_data_list)} 条记录")
            return market_data_list
            
    except Exception as e:
        logger.error(f"Polygon.io数据获取失败 ({symbol}): {e}")
        return None


def _map_interval_to_polygon(interval: str) -> tuple:
    """
    映射时间间隔到Polygon格式。
    
    Args:
        interval: 时间间隔字符串
    
    Returns:
        (multiplier, timespan) 元组
    """
    mapping = {
        '1m': (1, 'minute'),
        '5m': (5, 'minute'),
        '15m': (15, 'minute'),
        '30m': (30, 'minute'),
        '1h': (1, 'hour'),
        '1d': (1, 'day'),
        '1wk': (1, 'week'),
        '1mo': (1, 'month')
    }
    
    if interval in mapping:
        return mapping[interval]
    else:
        # 默认返回日线
        logger.warning(f"不支持的间隔 {interval}，使用默认值: 1天")
        return (1, 'day')


def _calculate_start_date(end_date: datetime, period: str) -> datetime:
    """
    根据期间计算开始日期。
    
    Args:
        end_date: 结束日期
        period: 期间字符串
    
    Returns:
        开始日期
    """
    period_mapping = {
        '1d': timedelta(days=1),
        '5d': timedelta(days=5),
        '1mo': timedelta(days=30),
        '3mo': timedelta(days=90),
        '6mo': timedelta(days=180),
        '1y': timedelta(days=365),
        '2y': timedelta(days=730),
        '5y': timedelta(days=1825),
        '10y': timedelta(days=3650)
    }
    
    delta = period_mapping.get(period, timedelta(days=365))
    return end_date - delta
