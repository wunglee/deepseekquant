"""
Yahoo Finance数据提供者 - 整合版
实现HistoricalDataProvider接口

职责：
- 通过yfinance API获取全球市场历史数据
- 支持指数、个股、波动率等多种数据类型
- 数据标准化和质量验证
- 实现统一的HistoricalDataProvider接口
- 代理配置和会话管理

Note: 
- 反爬虫和请求限流逻辑由 yfinance_patch 处理
- YahooFinanceDataProvider 仅负责代理配置和会话创建
- 避免在两个地方重复实现相同的反爬虫逻辑

依赖：
pip install yfinance

优势：
- 全球市场覆盖广泛
- 免费使用（有速率限制）
- 数据质量较高
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf
from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
# 导入新的数据结构
from core_bak_refactored.core.data.providers.protocols import (PriceData, IntradayData, IntradayTickRecord,
                                                               OrderBookLevel)
# 导入 HTTP/2 补丁
from core_bak_refactored.core.data.providers.yfinance_patch import patch_yfinance, _CURL_SESSION
from core_bak_refactored.core.share.market.market_time_utils import MarketTimeUtils
from core_bak_refactored.core.share.market.market_utils import MarketUtils

logger = logging.getLogger('DeepSeekQuant.YahooFinanceDataProvider')


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
        super().__init__()
        self.initialize()

    def initialize(self, **kwargs):
        # 延迟导入yfinance（避免环境依赖问题）
        try:
            import yfinance as yf
            self.yf = yf
            try:
                # 从 ConfigManager 读取 providers 配置
                provider_config = self.config_manager.get_provider_config()
                # 查找 yahoo provider 的 use_proxy 配置
                use_proxy = False
                for provider in provider_config.providers:
                    if provider.get('id') == "yahoo":
                        use_proxy = provider.get('use_proxy', False)
                        break
                import os
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
                if not use_proxy:
                    logger.info("🚫 yahoo配置为不使用代理，将使用无代理的网络请求")
                    # 不再清除环境变量，而是通过自定义会话控制代理
                    self.proxy = None
                    logger.info("✅ Yahoo 代理已禁用（通过自定义会话）")
                else:
                    # 查找可用的代理设置
                    for var in proxy_vars:
                        if var in os.environ:
                            self.proxy = os.environ[var]
                            logger.info(f"✅ 使用代理: {var} = {self.proxy}")
                            break
                    else:
                        self.proxy = None
                        logger.info("🌐 未找到代理环境变量，将使用直连")
                    if self.proxy:
                        logger.info("✅ Yahoo 代理已设置")
                    else:
                        logger.info("🌐 Yahoo 配置为使用直连")
            except Exception as e:
                logger.warning(f"配置代理时出错: {e}，将使用默认设置")

            patch_yfinance(proxy_url=self.proxy)
            logger.info("✅ YahooFinanceDataProvider initialized with Browser Simulation patch (anti-429)")
        except ImportError:
            logger.error("yfinance not installed. Please run: pip install yfinance")
            self.yf = None
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize yfinance: {e}")
            self.yf = None
            self.available = False

    def get_test_symbol(self) -> str:
        """获取测试符号"""
        return "^GSPC.US"  # 标普500指数

    def _fetch_history_kline_from_external_api(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                               period: str = 'daily') -> PriceData:
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
        if self.yf is None:
            raise RuntimeError("yfinance not available")

        logger.info(f"Fetching stock data for {symbol} from {start_date} to {end_date}, period={period}")
        # 使用带重试机制的方法获取数据
        if self.yf is None:
            raise RuntimeError("yfinance not available")

        try:
            # Note: 请求限流和重试逻辑由 yfinance_patch 处理
            # 🔧 将 period 转换为 yfinance 的 interval 参数
            interval_map = {
                'daily': '1d',
                'weekly': '1wk',
                'monthly': '1mo'
            }
            interval = interval_map.get(period, '1d')
            # Note: yfinance_patch 补丁会拦截所有 yfinance 内部请求
            ticker_obj = self.yf.Ticker(self._map_to_yahoo(symbol), session=_CURL_SESSION)
            start_date = MarketTimeUtils.to_market_time_by_symbol(start_date, symbol)
            end_date = MarketTimeUtils.to_market_time_by_symbol(end_date, symbol)
            data = ticker_obj.history(start=start_date, end=end_date, interval=interval)
        except Exception as e:
            logger.warning(f"Yahoo API调用失败 {symbol}: {e}")
            raise
        # 检查数据是否有效
        if data is None or data.empty:
            standardized_data = MarketUtils.standardize_format_to_price_data(data, symbol)
            logger.info(f"Yahoo 返回空数据：{symbol}")
            return standardized_data
        else:
            try:
                standardized_data = MarketUtils.standardize_format_to_price_data(data, symbol)
                logger.info(f"Successfully fetched {len(standardized_data.records)} records for {symbol}")
                return standardized_data
            except Exception as e:
                logger.error(f"Failed to standardized data for {symbol}: {e}")
                raise ValueError(f"Failed to standardized data for {symbol}: {str(e)}")

    def _fetch_real_intraday_from_external_api(self, symbol: str, start_time_str: str,
                                               end_time_str: str) -> pd.DataFrame:
        """
        从数据源获取分时数据（子类必须实现）

        Args:
            symbol: 证券代码
            start_time_str: 开始时间
            end_time_str: 结束时间

        Returns:
            pd.DataFrame: 分时数据 DataFrame
        """
        logger.info(f"时间范围: {start_time_str} ~ {end_time_str}")

        # 使用 yfinance 获取 1分钟数据
        # Note: yfinance_patch 补丁会拦截所有 yfinance 内部请求
        ticker_obj = self.yf.Ticker(self._map_to_yahoo(symbol), session=_CURL_SESSION)

        # Yahoo Finance 的 1m 数据最多只能获取 7 天
        # 如果时间范围超过 7 天，使用 5m 数据
        start_time = pd.Timestamp(start_time_str)
        end_time = pd.Timestamp(end_time_str)
        time_diff = (start_time - end_time).days
        if time_diff > 7:
            interval = '5m'
            period = '7d'  # 5分钟数据最多7天
            logger.info("时间范围超过 1 天，使用 5分钟数据")
        else:
            interval = '1m'
            period = '1d'  # 1分钟数据最多1天
            logger.info("使用 1分钟数据")

        # 获取数据
        df = ticker_obj.history(period=period, interval=interval)

        if df is None or df.empty:
            logger.warning(f"⚠️ Yahoo Finance 返回空数据: {symbol}")
            return df
        start_time = MarketTimeUtils.to_market_time_by_symbol(start_time,symbol)
        end_time = MarketTimeUtils.to_market_time_by_symbol(end_time,symbol)
        df = df[(df.index >= start_time) & (df.index <= end_time)]
        return df

    def _map_to_yahoo(self, symbol: str) -> str:
        return symbol[:-3]

    def _to_IntradayData(self, df: pd.DataFrame, symbol: str, trade_date: pd.Timestamp,
                         interpolate_func=None) -> IntradayData:
        """
        将 Yahoo Finance 的 DataFrame 转换为 IntradayData
        
        Args:
            df: Yahoo Finance 返回的 DataFrame（index 为时间戳）
            symbol: 证券代码
            trade_date: 交易日期
        
        Returns:
            IntradayData: 分时数据对象
        """

        is_index = MarketUtils.is_index(symbol)

        # 获取昨收价（使用第一个开盘价作为近似值）
        yesterday_close = float(df['Open'].iloc[0]) if not df.empty else 0.0

        # 获取当前价格（最后一条数据的收盘价）
        current_price = float(df['Close'].iloc[-1]) if not df.empty else 0.0

        # 计算涨跌
        change = current_price - yesterday_close
        change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0.0

        # 转换 ticks
        ticks = []
        total_volume = 0
        total_amount = 0.0

        for idx, row in df.iterrows():
            price = float(row['Close'])
            volume = int(row['Volume']) if pd.notna(row['Volume']) else 0

            total_volume += volume
            total_amount += price * volume
            avg_price = total_amount / total_volume if total_volume > 0 else price

            # 将时间索引转换为当地时间的H:M:S格式
            if hasattr(idx, 'strftime'):
                time_str = idx.strftime('%H:%M:%S')
            elif hasattr(idx, 'time'):
                time_str = str(idx.time())
            else:
                time_str = str(idx)

            ticks.append(IntradayTickRecord(
                time=time_str,
                price=round(price, 2),
                volume=volume,
                avg_price=round(avg_price, 2)
            ))

        return IntradayData(
            symbol=symbol,
            name=symbol,  # Yahoo Finance 不提供中文名称
            current_price=round(current_price, 2),
            yesterday_close=round(yesterday_close, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            ticks=ticks,
            order_book_bids=[],  # 由调用方设置
            order_book_asks=[],  # 由调用方设置
            trade_records=[],  # 由调用方设置
            trade_date=trade_date,
            order_book_message='',  # 由调用方设置
            trade_records_message='',  # 由调用方设置
            is_index=is_index,
            should_poll=False
        )

    def _fetch_realtime_order_book_from_external_api(self, symbol: str) -> Optional[tuple[list, list]]:
        """
        获取实时盘口数据（买卖五档）

        Args:
            symbol: 证券代码（带后缀，如000300.SH）

        Returns:
            (order_book_bids, order_book_asks): 买盘列表, 卖盘列表

        注意：
        - 只在交易时间内调用此方法
        - 非交易时间返回空列表
        """
        # 获取实时盘口数据（一档买卖盘）
        is_index = MarketUtils.is_index(symbol)
        order_book_bids = []
        order_book_asks = []
        if not is_index:
            try:
                ticker = yf.Ticker(self._map_to_yahoo(symbol), session=_CURL_SESSION)
                if not ticker.info:
                    logger.info(f"⚠️ Yahoo 返回空的盘口数据：{symbol}")
                else:
                    info = ticker.info
                    bid_price = info.get('bid')
                    ask_price = info.get('ask')
                    bid_size = info.get('bidSize')
                    ask_size = info.get('askSize')

                    if bid_price and bid_size:
                        order_book_bids.append(OrderBookLevel(
                            price=round(float(bid_price), 2),
                            volume=int(bid_size)
                        ))

                    if ask_price and ask_size:
                        order_book_asks.append(OrderBookLevel(
                            price=round(float(ask_price), 2),
                            volume=int(ask_size)
                        ))

                    logger.debug(
                        f"📊 Yahoo Finance 盘口: {symbol} bid={bid_price}x{bid_size} ask={ask_price}x{ask_size}")
            except Exception as e:
                logger.warning(f"⚠️ 获取 Yahoo Finance 盘口数据失败: {symbol}, {e}")
            return order_book_bids, order_book_asks
        else:
            return [], []

    def _fetch_realtime_trade_records_from_external_api(self, symbol: str):
        """
        获取实时成交明细（逐笔成交）

        Args:
            symbol: 证券代码（带后缀，如000300.SH）

        Returns:
            trade_records: 成交明细列表

        注意：
        - 只在交易时间内调用此方法
        - 非交易时间返回空列表
        """
        # Yahoo Finance API不提供逐笔成交数据，返回空列表
        logger.info(f"⚠️ Yahoo Finance 不支持逐笔成交数据: {symbol}")
        return []

    # _standardize_format method has been moved to MarketUtils.standardize_format_to_price_data

    # validate_data_quality方法已迁移到data_quality_utils.py
    # 请使用: from core_bak_refactored.core.data.quality.data_quality_utils import validate_data_quality
