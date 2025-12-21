"""
Yahoo Finance数据提供者 - 整合版
实现HistoricalDataProvider接口

职责：
- 通过yfinance API获取全球市场历史数据
- 支持指数、个股、波动率等多种数据类型
- 数据标准化和质量验证
- 实现统一的HistoricalDataProvider接口

依赖：
pip install yfinance

优势：
- 全球市场覆盖广泛
- 免费使用（有速率限制）
- 数据质量较高
"""

import logging
import random
import time
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
# 导入新的数据结构
from core_bak_refactored.core.data.providers.protocols import PriceData
from core_bak_refactored.core.share.config_manager import ConfigManager
# 导入 HTTP/2 补丁
from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance

logger = logging.getLogger('DeepSeekQuant.YahooFinance')


@dataclass
class YahooFinanceConfig:
    """Yahoo Finance配置数据类"""
    test_symbol: str = "^GSPC"  # 默认测试符号
    timeout: int = 30  # 请求超时（秒）
    max_retries: int = 3  # 最大重试次数


class YahooFinanceDataProvider(BaseDataProvider):
    """Yahoo Finance数据提供者"""
    
    def __init__(self):
        """
        初始化Yahoo Finance数据提供者
        
        Note:
            proxy 从配置文件读取，不通过参数传递
        """
        # 创建自定义 Session（官方推荐，避免 429 限流）
        super().__init__()
        self._session = self._create_session()
        
        # 请求限速器（避免 429）
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 每个请求之间至少间隔 0.5 秒
        self.init_yfinance()

    def init_yfinance(self):
        # 延迟导入yfinance（避免环境依赖问题）
        try:
            import yfinance as yf
            self.yf = yf
            try:
                # 从 ConfigManager 读取 providers 配置
                provider_config = self.config_manager.get_provider_config()
                # 查找 akshare provider 的 use_proxy 配置
                use_proxy = False
                for provider in provider_config.providers:
                    if provider.get('id') == "akshare":
                        use_proxy = provider.get('use_proxy', False)
                        break
                import os
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
                if not use_proxy:
                    logger.info("🚫 akshare配置为不使用代理，清除环境变量中的代理设置")
                    # 清除环境变量中的代理
                    for var in proxy_vars:
                        if var in os.environ:
                            logger.info(f"  清除环境变量: {var} = {os.environ[var]}")
                            del os.environ[var]
                    # 设置 requests 会话不使用代理
                    import requests
                    requests.Session().trust_env = False

                    logger.info("✅ 代理已禁用")
                else:
                    for var in proxy_vars:
                        if var in os.environ:
                            self.proxy = os.environ[var]
                            break
                    logger.info("🌐yahoo配置为使用代理")
            except Exception as e:
                logger.warning(f"配置代理时出错: {e}，将使用默认设置")
            if self.proxy:
                patch_yfinance(proxy_url=self.proxy)
                # 如果提供了代理，也配置 yfinance 原生代理（双保险）
                self.yf.set_config(proxy=self.proxy)
                logger.info(f"YahooFinanceDataProvider initialized with proxy: {self.proxy}")
            else:
                logger.info("YahooFinanceDataProvider initialized with custom session (anti-429)")
        except ImportError:
            logger.error("yfinance not installed. Please run: pip install yfinance")
            self.yf = None
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize yfinance: {e}")
            self.yf = None
            self.available = False

    def _create_session(self) -> requests.Session:
        """
        创建自定义 Session（官方推荐，避免 429 限流）
        
        根据 yfinance 官方文档和最佳实践：
        1. 使用真实的 User-Agent（模拟浏览器）
        2. 保持 Session 重用（保留 cookies）
        3. 设置合理的超时时间
        
        Returns:
            requests.Session: 配置好的 Session
        """
        session = requests.Session()
        
        # 设置真实的 User-Agent（关键！Yahoo 会检测默认的 User-Agent）
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        logger.info("Created custom session with browser-like headers (anti-429)")
        return session
    
    def _throttle_request(self):
        """
        请求限速（避免 429）
        
        确保两个请求之间有足够的时间间隔
        """
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            logger.debug(f"Throttling request: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def get_test_symbol(self) -> str:
        """获取测试符号"""
        return "^GSPC"  # 标普500指数
    
    def _fetch_with_retry(self, trade_record: str, start_date: pd.Timestamp, end_date: pd.Timestamp, period: str = 'daily', max_retries: int = 3) -> pd.DataFrame:
        """
        带重试机制的数据获取方法
        
        Note: 
        - yfinance 已经通过 patch 修复了 "Too Many Requests" bug
        - 使用指数退避策略处理速率限制
        - 使用自定义 Session 和请求限速避免 429
        
        Args:
            trade_record: 股票或指数代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 ('daily', 'weekly', 'monthly')
            max_retries: 最大重试次数
            
        Returns:
            DataFrame: 获取到的数据
        """
        if self.yf is None:
            raise RuntimeError("yfinance not available")
            
        for attempt in range(max_retries + 1):
            try:
                # 指数退避：第1次 5s, 第2次 10s, 第3次 20s, 第4次 40s
                if attempt > 0:
                    delay = 5 * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for {trade_record}, waiting {delay:.1f}s before retry (exponential backoff)")
                    time.sleep(delay)
                
                # 💚 请求限速（关键！避免 429）
                self._throttle_request()
                
                # 🔧 将 period 转换为 yfinance 的 interval 参数
                interval_map = {
                    'daily': '1d',
                    'weekly': '1wk',
                    'monthly': '1mo'
                }
                interval = interval_map.get(period, '1d')
                
                # 使用自定义 Session（关键！避免 429）
                ticker_obj = self.yf.Ticker(trade_record, session=self._session)
                data = ticker_obj.history(start=start_date, end=end_date, interval=interval)
                
                # 检查数据是否有效
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {len(data)} rows for {trade_record}")
                    return data
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {trade_record}: {e}")
                
                # 特殊处理速率限制错误
                if "Too Many Requests" in error_msg or "429" in error_msg or "Rate limited" in error_msg:
                    if attempt < max_retries:
                        logger.info(f"Rate limit hit, will retry with exponential backoff")
                        continue
                    else:
                        # 最后一次尝试失败，提供友好的错误信息
                        raise ValueError(
                            f"Yahoo Finance 速率限制 ({trade_record})\n"
                            f"建议: 1) 等待 5-10 分钟后重试\n"
                            f"      2) 或使用其他数据源 (AKShare/Tushare)\n"
                            f"      3) 或在 data_provider.yml 中启用代理: yahoo_finance.use_proxy: true"
                        )
                
                if attempt == max_retries:
                    raise
                continue
                
        # 如果所有重试都失败了，抛出异常
        raise RuntimeError(f"Failed to fetch data for {trade_record} after {max_retries + 1} attempts")
    
    def _inter_get_index_prices(
        self,
        index_id: str,
        start_date:pd.Timestamp,
        end_date: pd.Timestamp,
        period: str = 'daily'
    ) -> PriceData:
        """
        获取指数历史价格数据
        
        Args:
            index_id: 指数ID（如 "^GSPC"）
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 ('daily', 'weekly', 'monthly')
            
        Returns:
            PriceData: 标准化的价格数据
            
        Raises:
            ValueError: 当无法获取有效数据时
        """
        if self.yf is None:
            raise RuntimeError("yfinance not available")
            
        logger.info(f"Fetching index data for {index_id} from {start_date} to {end_date}, period={period}")
        
        try:
            # 使用带重试机制的方法获取数据
            data = self._fetch_with_retry(index_id, start_date, end_date, period)
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for {index_id}")
                
            # 标准化数据格式
            from core_bak_refactored.core.share.market.market_utils import MarketUtils
            standardized_data = MarketUtils.standardize_format_to_price_data(data, index_id)
            
            logger.info(f"Successfully fetched {len(standardized_data.records)} records for {index_id}")
            return standardized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {index_id}: {e}")
            raise ValueError(f"Failed to fetch data for {index_id}: {str(e)}")
    
    def _inter_get_stock_prices(
        self,
        stock_id: str,
        start_date: pd.Timestamp,
        end_date:pd.Timestamp,
        period: str = 'daily'
    ) -> PriceData:
        """
        获取个股历史价格数据
        
        Args:
            stock_id: 股票ID（如 "AAPL"）
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 ('daily', 'weekly', 'monthly')
        Returns:
            PriceData: 标准化的价格数据
            
        Raises:
            ValueError: 当无法获取有效数据时
        """
        if self.yf is None:
            raise RuntimeError("yfinance not available")
            
        logger.info(f"Fetching stock data for {stock_id} from {start_date} to {end_date}, period={period}")
        
        try:
            # 使用带重试机制的方法获取数据
            data = self._fetch_with_retry(stock_id, start_date, end_date, period)
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for {stock_id}")
                
            # 标准化数据格式
            from core_bak_refactored.core.share.market.market_utils import MarketUtils
            standardized_data = MarketUtils.standardize_format_to_price_data(data, stock_id)
            
            logger.info(f"Successfully fetched {len(standardized_data.records)} records for {stock_id}")
            return standardized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {stock_id}: {e}")
            raise ValueError(f"Failed to fetch data for {stock_id}: {str(e)}")
    
    def _fetch_from_external_api(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, period: str = 'daily') -> PriceData:
        """
        从 Yahoo Finance API 获取数据（实现基类抽象方法）
        
        💚 此方法由 BaseDataProvider._get_with_cache() 调用
        💚 三层缓存逻辑在基类中已实现，子类只需实现外部API调用
        
        Args:
            symbol: 证券代码（如 "^GSPC", "AAPL"）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            period: 周期 ('daily', 'weekly', 'monthly')
        
        Returns:
            PriceData: 标准化的价格数据
        """
        # 判断是指数还是个股（以 ^ 开头的是指数）
        if symbol.startswith('^'):
            return self._inter_get_index_prices(symbol, start_date, end_date, period)
        else:
            return self._inter_get_stock_prices(symbol, start_date, end_date, period)
    
    # _standardize_format method has been moved to MarketUtils.standardize_format_to_price_data
    
    # validate_data_quality方法已迁移到data_quality_utils.py
    # 请使用: from core_bak_refactored.core.data.quality.data_quality_utils import validate_data_quality