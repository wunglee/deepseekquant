"""
Finnhub数据提供者

职责：
1. 从Finnhub获取股票数据
2. 支持实时报价和历史数据
3. 提供基本面数据和财务指标
4. 处理API限流和认证
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


async def fetch_finnhub_data(
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[List[Dict]]:
    """
    从Finnhub获取K线数据。
    
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
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("Finnhub API密钥未配置")
        
        # 映射时间间隔到Finnhub格式
        resolution = _map_interval_to_finnhub(interval)
        
        # 计算时间范围（Unix时间戳）
        end_time = datetime.now()
        start_time = _calculate_start_time(end_time, period)
        
        from_ts = int(start_time.timestamp())
        to_ts = int(end_time.timestamp())
        
        # 构建API URL
        base_url = api_credentials.get('base_url', 'https://finnhub.io/api/v1')
        api_key = api_credentials['api_key']
        
        url = f"{base_url}/stock/candle?symbol={symbol}&resolution={resolution}&from={from_ts}&to={to_ts}&token={api_key}"
        
        # 发送HTTP请求
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"API请求失败，HTTP状态码: {response.status}, 错误: {error_text}")
            
            data = await response.json()
            
            # 检查响应状态
            if data.get('s') != 'ok':
                logger.warning(f"Finnhub未返回 {symbol} 的数据，状态: {data.get('s')}")
                return None
            
            # 解析K线数据
            timestamps = data.get('t', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            volumes = data.get('v', [])
            
            if not timestamps:
                return None
            
            # 转换为标准格式
            market_data_list = []
            for i in range(len(timestamps)):
                timestamp = datetime.fromtimestamp(timestamps[i])
                
                data_point = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(opens[i]) if i < len(opens) else 0,
                    'high': float(highs[i]) if i < len(highs) else 0,
                    'low': float(lows[i]) if i < len(lows) else 0,
                    'close': float(closes[i]) if i < len(closes) else 0,
                    'volume': float(volumes[i]) if i < len(volumes) else 0,
                    'metadata': {
                        'data_source': 'finnhub',
                        'data_type': data_type,
                        'interval': interval,
                        'resolution': resolution,
                        'adjustments': adjustments
                    }
                }
                market_data_list.append(data_point)
            
            logger.info(f"Finnhub成功获取 {symbol} 数据，共 {len(market_data_list)} 条记录")
            return market_data_list
            
    except Exception as e:
        logger.error(f"Finnhub数据获取失败 ({symbol}): {e}")
        return None


async def fetch_finnhub_quote(
    symbol: str,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[Dict]:
    """
    从Finnhub获取实时报价。
    
    Args:
        symbol: 股票代码
        api_credentials: API凭证
        aiohttp_session: aiohttp会话
    
    Returns:
        报价字典，失败返回None
    """
    try:
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("Finnhub API密钥未配置")
        
        base_url = api_credentials.get('base_url', 'https://finnhub.io/api/v1')
        api_key = api_credentials['api_key']
        
        url = f"{base_url}/quote?symbol={symbol}&token={api_key}"
        
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                return None
            
            quote = await response.json()
            
            return {
                'symbol': symbol,
                'current_price': quote.get('c'),
                'change': quote.get('d'),
                'percent_change': quote.get('dp'),
                'high': quote.get('h'),
                'low': quote.get('l'),
                'open': quote.get('o'),
                'previous_close': quote.get('pc'),
                'timestamp': datetime.fromtimestamp(quote.get('t', 0))
            }
            
    except Exception as e:
        logger.error(f"Finnhub报价获取失败 ({symbol}): {e}")
        return None


def _map_interval_to_finnhub(interval: str) -> str:
    """
    映射时间间隔到Finnhub resolution。
    
    Finnhub支持的resolution:
    - 1, 5, 15, 30, 60 (分钟)
    - D (日)
    - W (周)
    - M (月)
    
    Args:
        interval: 时间间隔字符串
    
    Returns:
        Finnhub resolution字符串
    """
    mapping = {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '1d': 'D',
        '1wk': 'W',
        '1mo': 'M'
    }
    
    return mapping.get(interval, 'D')


def _calculate_start_time(end_time: datetime, period: str) -> datetime:
    """
    根据期间计算开始时间。
    
    Args:
        end_time: 结束时间
        period: 期间字符串
    
    Returns:
        开始时间
    """
    period_mapping = {
        '1d': timedelta(days=1),
        '5d': timedelta(days=5),
        '1mo': timedelta(days=30),
        '3mo': timedelta(days=90),
        '6mo': timedelta(days=180),
        '1y': timedelta(days=365),
        '2y': timedelta(days=730),
        '5y': timedelta(days=1825)
    }
    
    delta = period_mapping.get(period, timedelta(days=365))
    return end_time - delta
