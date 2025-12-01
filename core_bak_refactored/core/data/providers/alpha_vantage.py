"""Alpha Vantage数据源适配器

职责：
- 从Alpha Vantage API获取市场数据
- 支持依赖注入以便测试覆盖
- 返回标准化的MarketData对象列表

设计模式：
- 依赖注入：支持注入fetch_fn进行测试
- 接口优先：实现统一的数据获取接口
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from urllib.parse import urlencode

import aiohttp

from core_bak_refactored.core.share import MarketData

logger = logging.getLogger('DeepSeekQuant.AlphaVantageProvider')


class AlphaVantageProvider:
    """Alpha Vantage数据源适配器
    
    职责单一：从Alpha Vantage API获取OHLCV数据
    
    设计特性：
    - 支持依赖注入fetch_fn用于测试覆盖
    - 自动管理aiohttp session生命周期
    - 标准化输出为MarketData对象列表
    
    Args:
        api_credentials: API凭证字典，必须包含'api_key'
        aiohttp_session: 复用的aiohttp会话（可选）
        fetch_fn: 注入的fetch函数，用于测试（可选）
    """

    def __init__(
        self,
        api_credentials: Optional[Dict[str, Any]] = None,
        aiohttp_session: Optional[Any] = None,
        fetch_fn: Optional[Callable[[str, str, str, str, bool], Any]] = None
    ) -> None:
        """初始化Alpha Vantage提供者
        
        Args:
            api_credentials: API凭证（包含api_key和base_url）
            aiohttp_session: 复用的HTTP会话
            fetch_fn: 测试用的mock函数
        """
        self._api_credentials = api_credentials or {}
        self._session = aiohttp_session
        self._fetch_fn = fetch_fn
        
        if not self._api_credentials.get('api_key'):
            logger.warning("AlphaVantage provider初始化时未提供API key")

    async def fetch(
        self,
        symbol: str,
        period: str,
        interval: str,
        data_type: str,
        adjustments: bool
    ) -> Optional[List[MarketData]]:
        """获取市场数据
        
        Args:
            symbol: 股票代码（如'AAPL'）
            period: 时间周期（'1y', '5y', 'max'等）
            interval: 数据间隔（'daily', '1m', '5m'等）
            data_type: 数据类型（'ohlcv'）
            adjustments: 是否进行价格调整
        
        Returns:
            MarketData对象列表，按时间升序排序；获取失败返回None
        
        Raises:
            不抛出异常，所有错误通过返回None处理
        """
        # 如果注入了测试函数，直接使用
        if self._fetch_fn is not None:
            logger.debug(f"使用注入的fetch_fn获取{symbol}数据")
            return await self._fetch_fn(symbol, period, interval, data_type, adjustments)
        
        # 验证凭证
        credentials = self._api_credentials
        if not credentials or not credentials.get('api_key'):
            logger.error("缺少API key，无法请求Alpha Vantage")
            return None
        # 映射interval到Alpha Vantage的function参数
        interval_param = None
        if interval in ['1m', '5m', '15m', '30m', '60m']:
            function = 'TIME_SERIES_INTRADAY'
            interval_param = interval
        elif interval == 'daily':
            function = 'TIME_SERIES_DAILY'
        elif interval == 'weekly':
            function = 'TIME_SERIES_WEEKLY'
        elif interval == 'monthly':
            function = 'TIME_SERIES_MONTHLY'
        else:
            logger.warning(f"不支持的interval: {interval}")
            return None
        # 构建请求参数
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': credentials['api_key'],
            'outputsize': 'full' if period in ['max', '10y', '5y'] else 'compact',
            'datatype': 'json'
        }
        if interval_param:
            params['interval'] = interval_param
        if adjustments:
            params['adjust'] = 'true'
        
        base_url = credentials.get('base_url', 'https://www.alphavantage.co/query')
        url = f"{base_url}?{urlencode(params)}"
        logger.info(f"请求Alpha Vantage: {symbol}, function={function}, interval={interval}")
        # 发送HTTP请求
        session = self._session or aiohttp.ClientSession()
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Alpha Vantage HTTP请求失败: status={response.status}")
                    return None
                data = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Alpha Vantage网络请求异常: {e}")
            return None
        except Exception as e:
            logger.error(f"Alpha Vantage数据解析异常: {e}")
            return None
        finally:
            if self._session is None:
                await session.close()
        # 检查API错误响应
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage API错误: {data['Error Message']}")
            return None
        if 'Note' in data:
            logger.warning(f"Alpha Vantage API限流: {data['Note']}")
            return None
        
        # 查找时间序列键
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error(f"响应中未找到时间序列数据: {list(data.keys())}")
            return None
        
        time_series = data[time_series_key]
        logger.debug(f"成功获取{len(time_series)}条时间序列数据")
        # 解析时间序列数据
        market_data_list: List[MarketData] = []
        parse_errors = 0
        
        for timestamp_str, values in time_series.items():
            # 尝试解析时间戳（支持日期+时间或仅日期）
            timestamp = None
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d')
                except ValueError:
                    parse_errors += 1
                    continue
            # 构造MarketData对象
            try:
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(values.get('1. open', 0)),
                    high=float(values.get('2. high', 0)),
                    low=float(values.get('3. low', 0)),
                    close=float(values.get('4. close', 0)),
                    volume=float(values.get('5. volume', 0)),
                    metadata={
                        'data_source': 'alpha_vantage',
                        'data_type': data_type,
                        'function': function,
                        'interval': interval
                    }
                )
                market_data_list.append(market_data)
            except (ValueError, TypeError) as e:
                parse_errors += 1
                logger.debug(f"数据解析错误: {timestamp_str}, {e}")
                continue
        
        if parse_errors > 0:
            logger.warning(f"Alpha Vantage数据解析时跳过了{parse_errors}条记录")
        
        if not market_data_list:
            logger.error("没有成功解析任何数据")
            return None
        
        logger.info(f"成功获取{len(market_data_list)}条{symbol}的数据")
        return sorted(market_data_list, key=lambda x: x.timestamp)
