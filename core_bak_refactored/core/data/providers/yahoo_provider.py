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
- 反爬虫和请求限流逻辑由 yfinance_http2_patch 处理
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
import random
import time
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import requests

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
# 导入新的数据结构
from core_bak_refactored.core.data.providers.protocols import (PriceData, TickRange, IntradayData, IntradayTickRecord,
                                                               OrderBookLevel)
# 导入 HTTP/2 补丁
from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
from core_bak_refactored.core.share.market.market_time_utils import MarketTimeUtils
from core_bak_refactored.core.share.market.market_enums import TradingPhase
from core_bak_refactored.core.share.market.market_utils import MarketUtils
import yfinance as yf

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

        # 代理配置（初始化为None）
        self.proxy = None

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
        return "^GSPC"  # 标普500指数

    def _fetch_history_prices(self, trade_record: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                              period: str = 'daily', max_retries: int = 3) -> pd.DataFrame:
        """
        带重试机制的数据获取方法
        
        Note: 
        - yfinance 已经通过 patch 修复了 "Too Many Requests" bug
        - 使用指数退避策略处理速率限制
        - 实际的请求限流逻辑在 yfinance_http2_patch 中处理
        
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

        try:
            # Note: 请求限流和重试逻辑由 yfinance_http2_patch 处理
            # 🔧 将 period 转换为 yfinance 的 interval 参数
            interval_map = {
                'daily': '1d',
                'weekly': '1wk',
                'monthly': '1mo'
            }
            interval = interval_map.get(period, '1d')

            # Note: yfinance_http2_patch 补丁会拦截所有 yfinance 内部请求
            ticker_obj = self.yf.Ticker(trade_record)
            data = ticker_obj.history(start=start_date, end=end_date, interval=interval)

            # 检查数据是否有效
            if data is not None and not data.empty:
                logger.info(f"Successfully fetched {len(data)} rows for {trade_record}")
                return data

        except Exception as e:
            logger.warning(f"Failed to fetch data for {trade_record}: {e}")
            raise

        # 如果所有重试都失败了，抛出异常
        raise RuntimeError(f"Failed to fetch data for {trade_record} after {max_retries + 1} attempts")

    def _inter_get_index_prices(
            self,
            index_id: str,
            start_date: pd.Timestamp,
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
            data = self._fetch_history_prices(index_id, start_date, end_date, period)

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
            end_date: pd.Timestamp,
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
            data = self._fetch_history_prices(stock_id, start_date, end_date, period)

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

    def _fetch_from_external_api(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                 period: str = 'daily') -> PriceData:
        self.data_ = """
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

    def get_intraday_data(self, symbol: str, tick_range: TickRange = None,
                          market_local_time: pd.Timestamp = None) -> IntradayData:
        """
        获取分时数据（通过 Yahoo Finance API）
        
        Args:
            symbol: 证券代码
            tick_range: 时间范围（可选）
            market_local_time: 市场本地时间（必须带正确的市场时区，由API层传入）
        
        Returns:
            IntradayData: 分时数据对象
        """
        # 使用传入的市场本地时间或当前系统UTC时间转换为本地时间
        market_code = MarketUtils.infer_market_from_symbol(symbol)
        if market_local_time is None:
            market_local_time = MarketTimeUtils.get_market_time_now(symbol)

        # market_local_time 已经是市场本地时间，直接使用
        trade_date = market_local_time.date()

        logger.info(f"🌍 使用市场本地时间: {market_local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # 使用市场本地时间判断交易时段
        trading_phase = MarketTimeUtils.determine_trading_phase(market_code, market_local_time)

        logger.info(f"识别市场: {symbol} -> {market_code.value}, 交易时段: {trading_phase.value}")

        # 判断是否为盘前时段，盘前返回空数据
        if trading_phase == TradingPhase.BEFORE_OPEN:
            logger.info("集合竞价时段，返回空数据")
            return self._generate_empty_intraday_data(symbol, trade_date, should_poll=True)

        # 🔧 关键修复：盘后时段获取最近一个交易日的数据
        if trading_phase == TradingPhase.AFTER_CLOSE:
            logger.info("🌃 盘后时段，获取最近一个交易日的数据")
            last_trade_date = MarketTimeUtils.get_last_trade_date(market_code, market_local_time)
            trade_date = last_trade_date.date()
            logger.info(f"📅 最近交易日: {trade_date}")

        try:
            # 计算时间范围
            if tick_range is not None:
                # 假设tick_range的时间已经是无时区的市场本地时间
                start_time = tick_range.start_time
                end_time = tick_range.end_time
            else:
                # 默认获取当日数据
                trading_hours = self.config_manager.get_trading_hours(market_code.value)
                # 直接创建无时区的时间戳
                start_time = pd.Timestamp(f"{trade_date} {trading_hours['open']}")

                if trading_phase == TradingPhase.AFTER_CLOSE:
                    # 盘后获取全天数据
                    end_time = pd.Timestamp(f"{trade_date} {trading_hours['close']}")
                elif trading_phase == TradingPhase.NOON_BREAK:
                    # 午休获取上午数据
                    end_time = pd.Timestamp(f"{trade_date} {trading_hours['lunch_start']}")
                else:
                    # 盘中获取到当前时间的数据，假设已经是无时区的市场本地时间
                    end_time = market_local_time

            logger.info(f"时间范围: {start_time} ~ {end_time}")

            # Note: 请求限流和重试逻辑由 yfinance_http2_patch 处理

            # 使用 yfinance 获取 1分钟数据
            # Note: yfinance_http2_patch 补丁会拦截所有 yfinance 内部请求
            ticker_obj = self.yf.Ticker(symbol)

            # Yahoo Finance 的 1m 数据最多只能获取 7 天
            # 如果时间范围超过 7 天，使用 5m 数据
            time_diff = (end_time - start_time).days
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
                # 返回空数据
                logger.info(f"返回空数据对象（可能是盘后或节假日）")
                return self._generate_empty_intraday_data(symbol, trade_date,
                                                          should_poll=(trading_phase != TradingPhase.AFTER_CLOSE))

            # 🔧 修复时区问题：确保时间范围过滤时时间戳具有一致的时区信息
            # start_time和end_time都是无时区的市场本地时间，所以需要将df的时间索引也转换为无时区的市场本地时间
            if df.index.tz is not None:
                # 将df索引转换为无时区的本地时间（保持为市场本地时间）
                df.index = df.index.tz_localize(None)

            df = df[(df.index >= start_time) & (df.index <= end_time)]

            if df.empty:
                logger.warning(f"⚠️ 过滤后数据为空: {symbol}")
                return self._generate_empty_intraday_data(symbol, trade_date,
                                                          should_poll=(trading_phase != TradingPhase.AFTER_CLOSE))

            # 转换为 IntradayData
            try:
                intraday_data = self._convert_yahoo_df_to_intraday(df, symbol, trade_date)

                # 设置 should_poll
                intraday_data.should_poll = trading_phase in [TradingPhase.BEFORE_OPEN, TradingPhase.TRADING]

                logger.info(f"✅ 成功获取 {len(intraday_data.ticks)} 条分时数据")
                return intraday_data
            except Exception as convert_error:
                logger.error(f"转换Yahoo Finance数据为分时数据失败: {convert_error}", exc_info=True)
                return self._generate_empty_intraday_data(symbol, trade_date, should_poll=False)

        except Exception as e:
            error_msg = str(e)

            # 检查是否是速率限制错误
            if "Rate limit" in error_msg or "Too Many Requests" in error_msg or "429" in error_msg:
                logger.warning(f"⚠️ Yahoo Finance 速率限制: {symbol}, 返回空数据")
            
            # 其他错误：记录并返回空数据（重试逻辑由补丁处理）
            logger.error(f"获取Yahoo Finance分时数据失败: {e}", exc_info=True)
            return self._generate_empty_intraday_data(symbol, trade_date, should_poll=False)

    def _generate_empty_intraday_data(self, symbol: str, trade_date, should_poll: bool = False) -> IntradayData:
        """
        生成空的分时数据对象
        
        Args:
            symbol: 证券代码
            trade_date: 交易日期
            should_poll: 是否需要轮询
        
        Returns:
            IntradayData: 空的分时数据对象
        """
        from core_bak_refactored.core.data.providers.protocols import IntradayData
        from core_bak_refactored.core.share.market.market_utils import MarketUtils

        is_index = MarketUtils.is_index(symbol)

        return IntradayData(
            symbol=symbol,
            name=symbol,  # Yahoo Finance 不提供中文名称
            current_price=0.0,
            yesterday_close=0.0,
            change=0.0,
            change_percent=0.0,
            ticks=[],
            order_book_bids=[],
            order_book_asks=[],
            trade_records=[],
            trade_date=trade_date,
            order_book_message='实时盘口数据仅在交易时段可用' if not is_index else '指数不可交易',
            trade_records_message='Yahoo Finance 仅提供 K 线数据，不提供逐笔成交明细' if not is_index else '指数无成交明细',
            is_index=is_index,
            should_poll=should_poll
        )

    def _convert_yahoo_df_to_intraday(self, df: pd.DataFrame, symbol: str, trade_date) -> IntradayData:
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

        # 获取实时盘口数据（一档买卖盘）
        order_book_bids = []
        order_book_asks = []
        if not is_index:
            try:
                ticker = yf.Ticker(symbol)
                if ticker.info:
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

                    logger.debug(f"📊 Yahoo Finance 盘口: {symbol} bid={bid_price}x{bid_size} ask={ask_price}x{ask_size}")
            except Exception as e:
                logger.warning(f"⚠️ 获取 Yahoo Finance 盘口数据失败: {symbol}, {e}")

        return IntradayData(
            symbol=symbol,
            name=symbol,  # Yahoo Finance 不提供中文名称
            current_price=round(current_price, 2),
            yesterday_close=round(yesterday_close, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            ticks=ticks,
            order_book_bids=order_book_bids,  # Yahoo Finance 提供一档买盘
            order_book_asks=order_book_asks,  # Yahoo Finance 提供一档卖盘
            trade_records=[],  # Yahoo Finance 不提供逐笔成交明细
            trade_date=trade_date,
            order_book_message='' if order_book_bids or order_book_asks else (
                '实时盘口数据仅在交易时段可用' if not is_index else '指数不可交易'),
            trade_records_message='Yahoo Finance 仅提供 K 线数据，不提供逐笔成交明细' if not is_index else '指数无成交明细',
            is_index=is_index,
            should_poll=False
        )

    # _standardize_format method has been moved to MarketUtils.standardize_format_to_price_data

    # validate_data_quality方法已迁移到data_quality_utils.py
    # 请使用: from core_bak_refactored.core.data.quality.data_quality_utils import validate_data_quality