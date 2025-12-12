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
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
# 导入新的数据结构
from core_bak_refactored.core.data.providers.protocols import PriceData, OHLCVRecord, HistoricalDataProvider
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


class YahooFinanceDataProvider(BaseDataProvider, HistoricalDataProvider):
    """Yahoo Finance数据提供者"""
    
    def __init__(self):
        """
        初始化Yahoo Finance数据提供者
        
        Note:
            proxy 从配置文件读取，不通过参数传递
        """
        # 创建自定义 Session（官方推荐，避免 429 限流）
        self._session = self._create_session()
        
        # 请求限速器（避免 429）
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 每个请求之间至少间隔 0.5 秒
        
        # 从配置文件中获取代理配置
        config_manager = ConfigManager()
        
        # 检查是否为 Yahoo Finance 启用代理（从 data.yml 读取）
        use_proxy = config_manager.get('data.data_providers.yahoo_finance.use_proxy', default=False)
        
        if use_proxy:
            # 使用统一的 get_proxies_from_config() 方法（从 system.yml 读取）
            proxy_config = config_manager.get_proxies_from_config()
            if proxy_config:
                # 使用HTTP代理（如果有配置的话）
                self.proxy = proxy_config.get('http') or proxy_config.get('socks5')
                logger.info(f"Yahoo Finance: 将尝试使用代理 {self.proxy}")
            else:
                self.proxy = None
                logger.warning(
                    "Yahoo Finance: use_proxy=true 但未配置代理地址\n"
                    "请在 config/dev/system.yml 中配置 proxies.http"
                )
        else:
            self.proxy = None
            logger.info(
                "Yahoo Finance: 不使用代理（直连）\n"
                "使用自定义 Session 和请求限速来避免 429 限流"
            )
        
        # 延迟导入yfinance（避免环境依赖问题）
        try:
            import yfinance as yf
            self.yf = yf
            
            # 应用 HTTP/2 补丁，传入代理配置
            # 补丁内部会测试代理是否可用
            patch_yfinance(proxy_url=self.proxy)
            
            # 如果提供了代理，也配置 yfinance 原生代理（双保险）
            if self.proxy:
                yf.set_config(proxy=self.proxy)
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
    
    def _fetch_with_retry(self, ticker: str, start_date: Union[str, datetime], end_date: Union[str, datetime], max_retries: int = 3) -> pd.DataFrame:
        """
        带重试机制的数据获取方法
        
        Note: 
        - yfinance 已经通过 patch 修复了 "Too Many Requests" bug
        - 使用指数退避策略处理速率限制
        - 使用自定义 Session 和请求限速避免 429
        
        Args:
            ticker: 股票或指数代码
            start_date: 开始日期
            end_date: 结束日期
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
                    logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for {ticker}, waiting {delay:.1f}s before retry (exponential backoff)")
                    time.sleep(delay)
                
                # 💚 请求限速（关键！避免 429）
                self._throttle_request()
                
                # 使用自定义 Session（关键！避免 429）
                ticker_obj = self.yf.Ticker(ticker, session=self._session)
                data = ticker_obj.history(start=start_date, end=end_date)
                
                # 检查数据是否有效
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
                    return data
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                
                # 特殊处理速率限制错误
                if "Too Many Requests" in error_msg or "429" in error_msg or "Rate limited" in error_msg:
                    if attempt < max_retries:
                        logger.info(f"Rate limit hit, will retry with exponential backoff")
                        continue
                    else:
                        # 最后一次尝试失败，提供友好的错误信息
                        raise ValueError(
                            f"Yahoo Finance 速率限制 ({ticker})\n"
                            f"建议: 1) 等待 5-10 分钟后重试\n"
                            f"      2) 或使用其他数据源 (AKShare/Tushare)\n"
                            f"      3) 或在 data.yml 中启用代理: yahoo_finance.use_proxy: true"
                        )
                
                if attempt == max_retries:
                    raise
                continue
                
        # 如果所有重试都失败了，抛出异常
        raise RuntimeError(f"Failed to fetch data for {ticker} after {max_retries + 1} attempts")
    
    def get_index_prices(
        self,
        index_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> PriceData:
        """
        获取指数历史价格数据
        
        Args:
            index_id: 指数ID（如 "^GSPC"）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            PriceData: 标准化的价格数据
            
        Raises:
            ValueError: 当无法获取有效数据时
        """
        if self.yf is None:
            raise RuntimeError("yfinance not available")
            
        logger.info(f"Fetching index data for {index_id} from {start_date} to {end_date}")
        
        try:
            # 使用带重试机制的方法获取数据
            data = self._fetch_with_retry(index_id, start_date, end_date)
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for {index_id}")
                
            # 标准化数据格式
            standardized_data = self._standardize_format(data, index_id)
            
            logger.info(f"Successfully fetched {len(standardized_data.records)} records for {index_id}")
            return standardized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {index_id}: {e}")
            raise ValueError(f"Failed to fetch data for {index_id}: {str(e)}")
    
    def get_stock_prices(
        self,
        stock_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> PriceData:
        """
        获取个股历史价格数据
        
        Args:
            stock_id: 股票ID（如 "AAPL"）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            PriceData: 标准化的价格数据
            
        Raises:
            ValueError: 当无法获取有效数据时
        """
        if self.yf is None:
            raise RuntimeError("yfinance not available")
            
        logger.info(f"Fetching stock data for {stock_id} from {start_date} to {end_date}")
        
        try:
            # 使用带重试机制的方法获取数据
            data = self._fetch_with_retry(stock_id, start_date, end_date)
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for {stock_id}")
                
            # 标准化数据格式
            standardized_data = self._standardize_format(data, stock_id)
            
            logger.info(f"Successfully fetched {len(standardized_data.records)} records for {stock_id}")
            return standardized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {stock_id}: {e}")
            raise ValueError(f"Failed to fetch data for {stock_id}: {str(e)}")
    
    def _standardize_format(self, data: pd.DataFrame, symbol: str) -> PriceData:
        """
        标准化Yahoo Finance返回的数据格式
        
        Args:
            data: Yahoo Finance返回的原始数据
            symbol: 证券代码
            
        Returns:
            PriceData: 标准化后的数据
        """
        if data is None or data.empty:
            # 空数据返回空的PriceData
            return PriceData(
                symbol=symbol, 
                records=[], 
                start_date=pd.Timestamp.now(),
                end_date=pd.Timestamp.now(),
                count=0
            )
        
        # 处理MultiIndex列结构
        if isinstance(data.columns, pd.MultiIndex):
            # 对于MultiIndex，我们需要提取正确的列
            # 通常格式是 [('Close', 'AAPL'), ('High', 'AAPL'), ...]
            close_col = ('Close', symbol) if ('Close', symbol) in data.columns else 'Close'
            high_col = ('High', symbol) if ('High', symbol) in data.columns else 'High'
            low_col = ('Low', symbol) if ('Low', symbol) in data.columns else 'Low'
            open_col = ('Open', symbol) if ('Open', symbol) in data.columns else 'Open'
            volume_col = ('Volume', symbol) if ('Volume', symbol) in data.columns else 'Volume'
        else:
            # 普通列名
            close_col = 'Close'
            high_col = 'High'
            low_col = 'Low'
            open_col = 'Open'
            volume_col = 'Volume'
        
        # 确保必要的列存在
        required_columns = [close_col, high_col, low_col, open_col, volume_col]
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"Missing column {col} in data for {symbol}")
                # 如果缺少某些列，使用默认值填充
                if col not in data.columns:
                    data[col] = np.nan
        
        records = []
        for timestamp, row in data.iterrows():
            try:
                record = OHLCVRecord(
                    date=pd.to_datetime(timestamp),  # 使用date而非timestamp
                    open=float(row.get(open_col, np.nan)),
                    high=float(row.get(high_col, np.nan)),
                    low=float(row.get(low_col, np.nan)),
                    close=float(row.get(close_col, np.nan)),
                    volume=int(row.get(volume_col, 0)) if not np.isnan(row.get(volume_col, np.nan)) else 0
                )
                records.append(record)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row for {symbol} at {timestamp}: {e}")
                continue
        
        # 计算start_date和end_date
        start_date = records[0].date if records else pd.Timestamp.now()
        end_date = records[-1].date if records else pd.Timestamp.now()
        
        return PriceData(
            symbol=symbol, 
            records=records,
            start_date=start_date,
            end_date=end_date,
            count=len(records)
        )
    
    # validate_data_quality方法已迁移到data_quality_utils.py
    # 请使用: from core_bak_refactored.core.data.quality.data_quality_utils import validate_data_quality