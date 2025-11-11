"""
数据获取器核心 - 业务层
从 core_bak/data_fetcher.py 拆分
职责: 数据获取主流程、数据源管理
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .data_models import DataSourceType, DataFrequency, MarketData
from ...infrastructure.data_adapters import TushareAdapter, AKShareAdapter, YFinanceAdapter

logger = logging.getLogger('DeepSeekQuant.DataFetcher')


class DataFetcher:
    """完整生产级数据获取器 - 支持多数据源和实时数据"""

    def __init__(self, config: Dict):
        self.config = config
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_duration = config.get('cache_duration', 300)
        self.primary_source = config.get('primary', DataSourceType.YAHOO_FINANCE.value)
        self.fallback_sources = config.get('fallback_sources', [])
        self.max_retries = config.get('max_retries', 3)
        self.request_timeout = config.get('request_timeout', 30)

        # 初始化缓存系统
        self._setup_caching()

        # 初始化API密钥和认证
        self._setup_api_credentials()

        # 初始化连接池和会话
        self._setup_http_client()

        # 初始化Redis缓存（如果配置）
        self._setup_redis_cache()

        # 初始化数据源连接
        self.data_sources = self._initialize_data_sources()

        # 性能监控
        self.performance_metrics = {
            'requests_total': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0,
            'data_points_processed': 0,
            'last_update': datetime.now().isoformat()
        }

        logger.info(f"数据获取器初始化完成，主数据源: {self.primary_source}")

    def _setup_caching(self):
        """设置缓存系统"""
        # 内存缓存
        self.memory_cache = TTLCache(
            maxsize=self.config.get('memory_cache_size', 1000),
            ttl=self.cache_duration
        )

        # LRU缓存用于频繁访问的数据
        self.lru_cache = LRUCache(maxsize=self.config.get('lru_cache_size', 500))

        # 缓存统计
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0,
            'evictions': 0
        }

    def _setup_api_credentials(self):
        """设置API认证信息"""
        self.api_credentials = {}
        sources_config = self.config.get('sources', {})

        for source_type in DataSourceType:
            source_config = sources_config.get(source_type.value, {})
            if source_config.get('enabled', False):
                self.api_credentials[source_type.value] = {
                    'api_key': source_config.get('api_key', ''),
                    'secret_key': source_config.get('secret_key', ''),
                    'access_token': source_config.get('access_token', ''),
                    'base_url': source_config.get('base_url', ''),
                    'rate_limit': source_config.get('rate_limit', {}),
                    'authentication_type': source_config.get('authentication_type', 'api_key')
                }

    def _setup_http_client(self):
        """设置HTTP客户端"""
        # 异步HTTP客户端
        self.aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
            headers={'User-Agent': 'DeepSeekQuant/1.0.0'}
        )

        # 同步HTTP会话
        self.requests_session = requests.Session()
        self.requests_session.mount('https://', requests.adapters.HTTPAdapter(
            pool_connections=100, pool_maxsize=100, max_retries=3
        ))

    def _setup_redis_cache(self):
        """设置Redis缓存"""
        redis_config = self.config.get('redis', {})
        if redis_config.get('enabled', False):
            try:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    decode_responses=False,
                    socket_timeout=redis_config.get('socket_timeout', 5),
                    retry_on_timeout=True
                )
                # 测试连接
                self.redis_client.ping()
                logger.info("Redis缓存连接成功")
            except Exception as e:
                logger.error(f"Redis连接失败: {e}")
                self.redis_client = None
        else:
            self.redis_client = None

    def _initialize_data_sources(self) -> Dict:
        """初始化数据源连接"""
        sources = {}

        # 注册所有支持的数据源
        sources[DataSourceType.YAHOO_FINANCE.value] = self._fetch_yahoo_data
        sources[DataSourceType.ALPHA_VANTAGE.value] = self._fetch_alpha_vantage_data
        sources[DataSourceType.IEX_CLOUD.value] = self._fetch_iex_cloud_data
        sources[DataSourceType.POLYGON.value] = self._fetch_polygon_data
        sources[DataSourceType.TWELVE_DATA.value] = self._fetch_twelve_data
        sources[DataSourceType.FINNHUB.value] = self._fetch_finnhub_data
        sources[DataSourceType.TIINGO.value] = self._fetch_tiingo_data
        sources[DataSourceType.QUANDL.value] = self._fetch_quandl_data
        sources[DataSourceType.INTRINIO.value] = self._fetch_intrinio_data
        sources[DataSourceType.EOD_HISTORICAL.value] = self._fetch_eod_historical_data
        sources[DataSourceType.CUSTOM_API.value] = self._fetch_custom_api_data
        sources[DataSourceType.DATABASE.value] = self._fetch_database_data
        sources[DataSourceType.BROKER_API.value] = self._fetch_broker_api_data

        return sources

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, requests.RequestException))
    )
    async def get_historical_data(self, symbols: List[str],
                                  period: str = '1y',
                                  interval: str = '1d',
                                  data_type: str = 'ohlcv',
                                  adjustments: bool = True) -> Dict[str, List[MarketData]]:
        """
        获取历史数据 - 完整生产实现

        Args:
            symbols: 股票代码列表
            period: 数据期间 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            data_type: 数据类型 (ohlcv, dividends, splits, all)
            adjustments: 是否调整价格（分红和拆股）

        Returns:
            包含市场数据的字典
        """
        start_time = time.time()
        results = {}
        failed_symbols = []

        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(symbols, period, interval, data_type, adjustments)

            # 检查缓存
            cached_data = await self._get_cached_data(cache_key)
            if cached_data and self.cache_enabled:
                self.cache_stats['hits'] += 1
                self.performance_metrics['cache_hits'] += 1
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data

            self.cache_stats['misses'] += 1
            self.performance_metrics['cache_misses'] += 1

            # 并发获取所有符号的数据
            tasks = []
            for symbol in symbols:
                task = self._fetch_symbol_data(symbol, period, interval, data_type, adjustments)
                tasks.append(task)

            # 等待所有任务完成
            symbol_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(symbol_results):
                symbol = symbols[i]

                if isinstance(result, Exception):
                    logger.error(f"获取 {symbol} 数据失败: {result}")
                    failed_symbols.append(symbol)
                    continue

                if result:
                    results[symbol] = result
                    self.performance_metrics['data_points_processed'] += len(result)

            # 如果主数据源失败，尝试备用数据源
            if failed_symbols and self.fallback_sources:
                logger.warning(f"主数据源失败，尝试备用数据源: {failed_symbols}")
                fallback_results = await self._try_fallback_sources(failed_symbols, period, interval, data_type,
                                                                    adjustments)
                results.update(fallback_results)

            # 缓存结果
            if results and self.cache_enabled:
                await self._cache_data(cache_key, results)

            # 更新性能指标
            duration = time.time() - start_time
            self.performance_metrics['requests_total'] += len(symbols)
            self.performance_metrics['requests_failed'] += len(failed_symbols)
            self.performance_metrics['avg_response_time'] = (
                    self.performance_metrics['avg_response_time'] * 0.9 + duration * 0.1
            )
            self.performance_metrics['last_update'] = datetime.now().isoformat()

            logger.info(f"历史数据获取完成: {len(results)} 成功, {len(failed_symbols)} 失败, 耗时: {duration:.2f}s")

            return results

        except Exception as e:
            logger.error(f"历史数据获取失败: {e}")
            raise

    async def _fetch_symbol_data(self, symbol: str, period: str, interval: str,
                                 data_type: str, adjustments: bool) -> Optional[List[MarketData]]:
        """获取单个符号的数据"""
        try:
            # 首先尝试主数据源
            primary_source_func = self.data_sources.get(self.primary_source)
            if primary_source_func:
                data = await primary_source_func(symbol, period, interval, data_type, adjustments)
                if data:
                    return data

            # 如果主数据源失败，按优先级尝试备用数据源
            for fallback_source in self.fallback_sources:
                if fallback_source in self.data_sources:
                    fallback_func = self.data_sources[fallback_source]
                    try:
                        data = await fallback_func(symbol, period, interval, data_type, adjustments)
                        if data:
                            logger.info(f"备用数据源 {fallback_source} 成功获取 {symbol} 数据")
                            return data
                    except Exception as e:
                        logger.warning(f"备用数据源 {fallback_source} 获取 {symbol} 失败: {e}")
                        continue

            return None

        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None

    async def _fetch_yahoo_data(self, symbol: str, period: str, interval: str,
                                data_type: str, adjustments: bool) -> Optional[List[MarketData]]:
        """从Yahoo Finance获取数据 - 完整实现"""
        try:
            ticker = yf.Ticker(symbol)

            # 根据数据类型获取不同的数据
            if data_type == 'ohlcv':
                hist = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=adjustments,
                    actions=False
                )
            elif data_type == 'dividends':
                hist = ticker.dividends
            elif data_type == 'splits':
                hist = ticker.splits
            elif data_type == 'all':
                hist = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=adjustments,
                    actions=True
                )
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")

            if hist.empty:
                return None

            # 转换为MarketData对象列表
            market_data_list = []
            for idx, row in hist.iterrows():
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                    open=row.get('Open', float('nan')),
                    high=row.get('High', float('nan')),
                    low=row.get('Low', float('nan')),
                    close=row.get('Close', float('nan')),
                    volume=row.get('Volume', 0),
                    adj_close=row.get('Adj Close', float('nan')),
                    dividends=row.get('Dividends', 0) if 'Dividends' in row else 0,
                    splits=row.get('Stock Splits', 1) if 'Stock Splits' in row else 1,
                    metadata={
                        'data_source': 'yahoo',
                        'data_type': data_type,
                        'period': period,
                        'interval': interval,
                        'adjustments': adjustments
                    }
                )
                market_data_list.append(market_data)

            return market_data_list

        except Exception as e:
            logger.error(f"Yahoo Finance数据获取失败: {e}")
            return None

    async def _fetch_alpha_vantage_data(self, symbol: str, period: str, interval: str,
                                        data_type: str, adjustments: bool) -> Optional[List[MarketData]]:
        """从Alpha Vantage获取数据 - 完整实现"""
        try:
            credentials = self.api_credentials.get(DataSourceType.ALPHA_VANTAGE.value)
            if not credentials or not credentials.get('api_key'):
                raise ValueError("Alpha Vantage API密钥未配置")

            # 根据时间间隔选择API函数
            if interval in ['1m', '5m', '15m', '30m', '60m']:
                function = 'TIME_SERIES_INTRADAY'
                interval_param = interval
            elif interval == 'daily':
                function = 'TIME_SERIES_DAILY'
                interval_param = None
            elif interval == 'weekly':
                function = 'TIME_SERIES_WEEKLY'
                interval_param = None
            elif interval == 'monthly':
                function = 'TIME_SERIES_MONTHLY'
                interval_param = None
            else:
                raise ValueError(f"不支持的间隔: {interval}")

            # 构建API URL
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

            url = f"{credentials.get('base_url', 'https://www.alphavantage.co/query')}?{urlencode(params)}"

            # 发送请求
            async with self.aiohttp_session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"API请求失败: {response.status}")

                data = await response.json()

                # 解析数据
                time_series_key = None
                for key in data.keys():
                    if 'Time Series' in key:
                        time_series_key = key
                        break

                if not time_series_key:
                    raise ValueError("未找到时间序列数据")

                time_series = data[time_series_key]
                market_data_list = []

                for timestamp_str, values in time_series.items():
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                    market_data = MarketData(
