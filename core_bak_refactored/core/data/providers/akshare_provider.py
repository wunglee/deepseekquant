"""
AKShare数据提供者 - 全市场数据源
实现HistoricalDataProvider接口

职责：
- 通过AKShare API获取全市场历史数据
- 支持A股、港股、美股指数和个股
- 数据标准化和质量验证
- 实现统一的HistoricalDataProvider接口

依赖：
pip install akshare

优势：
- 完全免费，无需注册
- 无调用限制
- 数据覆盖全面（A股、港股、美股、期货、基金等）
- 更新及时，数据质量高

未来扩展计划：
- 实现全球指数名称智能映射（基于AKShare index_global_name_table）
- 添加指数名称缓存机制
- 支持更多市场和数据类型
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import akshare as ak
import pandas as pd

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
from core_bak_refactored.core.data.providers.protocols import (PriceData,
                                                               IntradayData, IntradayTickRecord,
                                                               OrderBookLevel, TradeDetailRecord)
from core_bak_refactored.core.data.providers.protocols import TickRange
from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.core.share.market import MarketUtils
from core_bak_refactored.core.share.market.market_enums import TradingPhase

logger = logging.getLogger(__name__)


class AKShareDataProvider(BaseDataProvider):
    """
    基于AKShare的数据提供者

    支持多市场数据获取：
    - A股：个股、指数
    - 港股：个股、指数
    - 美股：个股、指数

    设计特点：
    - 自动识别代码格式并调用对应API
    - 统一的数据标准化接口
    - 透明失败原则（网络问题直接抛出异常）
    - 备选数据源机制
    """

    def __init__(self):
        """初始化AKShare数据提供者"""
        # 💚 调用基类构造函数（初始化缓存）
        super().__init__()
        
        # 🔧 禁用数据库缓存（避免 iCloud 路径的 disk I/O error）
        self._enable_db_cache = False
        logger.info("💾 已禁用数据库缓存（仅使用内存缓存）")
        
        # 🔧 初始化分时数据专用的内存缓存（与历史数据缓存分开）
        self._enable_memory_cache = True  # 启用分时数据内存缓存
        self._memory_cache = {}  # 分时数据缓存字典

        self.ak = None
        self.available = False
        self._load_us_symbol_mapping()
        self._initialize()
        self.config_manager = ConfigManager()
        
        # 🔧 读取代理配置并处理环境变量
        self._configure_proxy()

    def get_test_symbol(self) -> str:
        """获取测试符号"""
        return '000300.SH'  # 沪深300指数

    def _load_us_symbol_mapping(self):
        """加载美股符号映射配置"""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'config',
                'us_symbol_mapping.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.us_symbol_mapping = config.get('us_symbol_mapping', {})
            logger.info(f"Loaded US symbol mapping with {len(self.us_symbol_mapping)} entries")
        except Exception as e:
            logger.warning(f"Failed to load US symbol mapping: {e}, using empty mapping")
            self.us_symbol_mapping = {}

    def _us_symbol_to_chinese(self, us_symbol: str) -> str:
        """
        将美股代码转换为AKShare全球指数API所需的中文名称

        Args:
            us_symbol: 美股代码（如 '^GSPC'）

        Returns:
            中文指数名称（如 '标普500'）或去掉前缀'^'的原始字符
        """
        # 从配置文件读取映射表
        chinese_name = self.us_symbol_mapping.get(us_symbol)
        if chinese_name:
            return chinese_name

        # 如果配置文件中没有找到映射，返回去掉前缀'^'的原始字符
        if us_symbol.startswith('^'):
            return us_symbol[1:]

        # 其他情况直接返回原始字符
        return us_symbol

    def _initialize(self):
        """初始化AKShare模块"""
        try:
            self.ak = ak
            self.available = True
            logger.info("AKShareDataProvider initialized successfully")
        except ImportError as e:
            logger.warning(f"akshare not installed: {e}. Install with: pip install akshare")
            self.ak = None
            raise

        # 未来扩展：指数名称映射缓存（暂未实现）
        self._index_name_cache = {
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指'
        }
    
    def _configure_proxy(self):
        """配置代理设置
        
        从 data_provider.yml 读取 use_proxy 配置，如果为 false，则禁用代理：
        1. 清除环境变量中的代理设置
        2. 设置 requests 库不使用代理
        """
        try:
            # 从 ConfigManager 读取 providers 配置
            provider_config = self.config_manager.get_provider_config()
            
            # 查找 akshare provider 的 use_proxy 配置
            use_proxy = False
            for provider in provider_config.providers:
                if provider.get('id') == 'akshare':
                    use_proxy = provider.get('use_proxy', False)
                    break
            
            if not use_proxy:
                logger.info("🚫 AKShare 配置为不使用代理，清除环境变量中的代理设置")
                
                # 清除环境变量中的代理
                import os
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
                for var in proxy_vars:
                    if var in os.environ:
                        logger.info(f"  清除环境变量: {var} = {os.environ[var]}")
                        del os.environ[var]
                
                # 设置 requests 会话不使用代理
                import requests
                requests.Session().trust_env = False
                
                logger.info("✅ 代理已禁用")
            else:
                logger.info("🌐 AKShare 配置为使用代理")
                
        except Exception as e:
            logger.warning(f"配置代理时出错: {e}，将使用默认设置")

    def get_index_returns(
            self,
            index_id: str,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp
    ) -> pd.Series:
        """
        获取指数收益率序列（实现HistoricalDataProvider接口）

        Args:
            index_id: 指数代码
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象

        Returns:
            Series with date index and return values
        """
        price_data = self.get_index_prices(index_id, start_date, end_date, pd.Timestamp.now())
        prices = price_data.to_dataframe().set_index('date')
        returns = prices['close'].pct_change().dropna()
        return returns

    def _fetch_from_external_api(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, period: str = 'daily') -> PriceData:
        """
        获取历史价格数据（适用于个股和指数）

        Args:
            symbol: 股票或指数代码（支持市场后缀，如 '000001.SZ' 或 '^GSPC'）
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 ('daily', 'weekly', 'monthly')

        Returns:
            标准化的价格数据

        设计原则：
        - 职责单一：只负责数据获取和标准化
        - 日期处理：统一转换为datetime对象
        - 异常处理：网络失败直接抛出异常，符合透明失败原则
        
        🔧 AKShare 没有周线/月线 API，需要从日线数据转换
        """
        if not self.available or self.ak is None:
            raise RuntimeError("AKShare API不可用，请安装: pip install akshare")

        try:
            logger.info(f"Fetching price data for {symbol} from {start_date} to {end_date}")

            # 🔧 关键修复：传递日期范围参数给 _fetch_by_market
            # 根据代码格式自动判断市场，调用对应的AKShare API
            df = self._fetch_by_market(symbol, start_date=start_date, end_date=end_date)

            # 🔧 关键修复：从 AKShare 获取的数据为空，返回空 PriceData（不是错误）
            # 根据业务场景，这可能是：
            # 1. 无限滚动到头了（before 早于数据源最早日期）
            # 2. 该股票/指数确实没有数据
            if df is None or df.empty:
                logger.warning(f"⚠️ AKShare 返回空数据: {symbol} ({start_date} to {end_date})。可能原因：1.无限滚动到头 2.该股票没有数据")
                # 返回空 PriceData 对象，使用查询的日期范围
                return PriceData(
                    records=[],
                    symbol=symbol,
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                    count=0
                )

            # 🔧 先标准化格式（处理列名差异）
            from core_bak_refactored.core.share.market.market_utils import MarketUtils
            standardized_data = MarketUtils.standardize_format(df)

            # 筛选日期范围（使用标准化后的数据）
            if 'date' not in standardized_data.columns:
                raise ValueError(
                    f"Standardized data missing 'date' column. Columns: {standardized_data.columns.tolist()}")

            standardized_data['date'] = pd.to_datetime(standardized_data['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            standardized_data = standardized_data[
                (standardized_data['date'] >= start_dt) & (standardized_data['date'] <= end_dt)
                ]

            # 🔧 关键修复：筛选后数据为空，返回空 PriceData（不是错误）
            if standardized_data.empty:
                logger.warning(f"⚠️ 日期范围筛选后数据为空: {symbol} ({start_date} to {end_date})。可能原因：1.无限滚动到头 2.该时间段没有数据")
                # 返回空 PriceData 对象，使用查询的日期范围
                return PriceData(
                    records=[],
                    symbol=symbol,
                    start_date=start_dt,  # 已经转换为 pd.Timestamp
                    end_date=end_dt,      # 已经转换为 pd.Timestamp
                    count=0
                )

            logger.info(f"Successfully fetched {len(standardized_data)} rows for {symbol}")
            # 返回PriceData对象而不是原始DataFrame
            price_data = PriceData.from_dataframe(standardized_data, symbol)
            
            # 🔧 AKShare 不支持直接查询周线/月线，需要从日线转换
            if period != 'daily':
                logger.info(f"AKShare 不支持直接查询 {period}，从日线转换（{price_data.count} 条日线数据）")
                price_data = self._convert_period(price_data, period)
            
            return price_data

        except Exception as e:
            logger.error(f"AKShare failed for {symbol}: {e}")
            # 更详细的错误信息，帮助调试
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}") from e


    def _map_to_akshare(self, symbol: str, with_market_prefix: bool = True) -> str:
        """
        映射代码到AKShare格式（基于规则自动转换）

        Args:
            symbol: 统一代码
            with_market_prefix: 是否包含市场前缀（sh/sz）

        Returns:
            AKShare API所需的代码格式

        转换规则：
        1. A股指数/个股：
           - with_market_prefix=True: '000300.SH' → 'sh000300'，'399001.SZ' → 'sz399001'
           - with_market_prefix=False: '000300.SH' → '000300'，'399001.SZ' → '399001'
        2. 港股指数：'HSI' → 'HSI'（保持原样）
        3. 美股指数：'^GSPC' → '标普500'（AKShare全球指数API需要中文名）

        设计原则：
        - 不维护写死的股票代码映射表
        - 基于代码格式规则进行自动转换
        - 快捷映射表仅为性能优化，非必需

        不同API的格式要求：
        - stock_zh_a_hist_min_em（分时数据）：需要纯数字，with_market_prefix=False
        - stock_bid_ask_em（盘口）：需要纯数字，with_market_prefix=False
        - stock_zh_a_tick_tx_js（成交明细）：需要市场前缀，with_market_prefix=True
        - stock_zh_index_daily（指数日线）：需要市场前缀，with_market_prefix=True
        """

        # A股指数/个股：自动转换 .SH/.SZ
        if symbol.endswith('.SH'):
            code = symbol[:-3]  # 移除 .SH 后缀
            return ('sh' + code) if with_market_prefix else code
        if symbol.endswith('.SZ'):
            code = symbol[:-3]  # 移除 .SZ 后缀
            return ('sz' + code) if with_market_prefix else code

        # 美股指数：自动转换 ^开头 → 中文名称
        if symbol.startswith('^'):
            return self._us_symbol_to_chinese(symbol)

        # 其他：直接返回（港股、其他市场）
        return symbol

    def _fetch_by_market(self, symbol_id: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        根据原始代码格式判断市场，调用对应的AKShare API

        Args:
            symbol_id: 原始代码
            start_date: 开始日期（YYYY-MM-DD格式，可选）
            end_date: 结束日期（YYYY-MM-DD格式，可选）

        Returns:
            AKShare返回的原始DataFrame

        设计原则：
        - 职责单一：只负责根据市场调用对应API
        - 代码格式转换已在 _map_to_akshare 完成
        - 市场判断逻辑集中在此方法
        
        🔧 优先使用支持日期范围的API（如index_zh_a_hist）
        """
        # 映射代码
        ak_symbol = self._map_to_akshare(symbol_id)
        logger.info(f"Fetching data for {symbol_id} (mapped to {ak_symbol})")

        # 使用领域层工具推断市场
        from core_bak_refactored.core.share.market import MarketUtils, MarketCode
        market = MarketUtils.infer_market_from_symbol(symbol_id)

        try:
            # 根据市场选择 API
            if market == MarketCode.CN:
                # 🔧 A股指数：优先使用支持日期范围的 index_zh_a_hist API
                # 移除市场前缀（index_zh_a_hist只需要纯数字代码）
                pure_code = ak_symbol.replace('sh', '').replace('sz', '')
                start_date_str=start_date.strftime('%Y%m%d')
                end_date_str=end_date.strftime('%Y%m%d')
                logger.debug(f"调用A股指数API: index_zh_a_hist({pure_code}, period='daily', start_date={start_date_str}, end_date={end_date_str})")
                df = self.ak.index_zh_a_hist(
                    symbol=pure_code,
                    period='daily',
                    start_date=start_date_str,
                    end_date=end_date_str,
                )
                
                # 重命名列（index_zh_a_hist使用中文列名）
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_change',
                    '涨跌额': 'change',
                    '换手率': 'turnover'
                }
                df = df.rename(columns=column_mapping)
                return df

            elif market == MarketCode.HK:
                # 港股指数API
                logger.debug(f"调用港股指数API: stock_hk_index_daily_em({ak_symbol})")
                return self.ak.stock_hk_index_daily_em(symbol=ak_symbol)

            elif market == MarketCode.US:
                # 美股/全球指数API
                logger.debug(f"调用全球指数API: index_global_hist_em({ak_symbol})")
                return self.ak.index_global_hist_em(symbol=ak_symbol)

            else:
                # 默认使用A股指数API
                pure_code = ak_symbol.replace('sh', '').replace('sz', '')
                start_date_str=start_date.strftime('%Y%m%d')
                end_date_str=end_date.strftime('%Y%m%d')
                logger.debug(f"默认调用A股指数API: index_zh_a_hist({pure_code})")
                df = self.ak.index_zh_a_hist(
                    symbol=pure_code,
                    period='daily',
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                column_mapping = {
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume'
                }
                df = df.rename(columns=column_mapping)
                return df
        except Exception as e:
            logger.error(f"AKShare API调用失败 for {symbol_id} (market: {market.value}): {e}")
            # 提供更友好的错误信息
            if "HTTPSConnectionPool" in str(e) or "proxy" in str(e).lower():
                raise ConnectionError(f"网络连接失败，请检查网络设置或代理配置: {str(e)}")
            # 重新抛出异常，让上层处理
            raise

    # _standardize_format method has been moved to MarketUtils.standardize_format

    def get_intraday_data(self, symbol: str, tick_range: TickRange = None,
                          current_time: datetime = None) -> IntradayData:
        """
        获取分时图数据（1分钟级别） - 仅负责真实数据

        实现策略：
        1. 智能日期处理：如果是周末/节假日，自动获取最近交易日数据
        2. 优先从内存缓存获取
        3. 然后尝试真实API（AKShare stock_zh_a_hist_min_em）
        4. API失败则fallback到前一交易日缓存
        5. 最后返回空数据

        Args:
            symbol: 证券代码
            tick_range: 时间范围
            current_time: 当前时间（用于测试，默认使用系统时间）
        注意：
        - 此方法只负责获取真实数据
        - 模拟数据由 MockDataProvider 单独处理
        :param symbol: 证券代码
        :param tick_range: 时间范围
        :param current_time: 当前时间（用于测试）
        """

        logger.info(f"获取真实分时数据: symbol={symbol}")

        # 🔧 根据symbol识别市场
        market_code = MarketUtils.infer_market_from_symbol(symbol)
        logger.info(f"识别市场: {symbol} -> {market_code.value}")

        # 使用传入的时间或当前系统时间
        now = pd.Timestamp.now()
        trade_date = now
        intraday_data = None
        trading_phase = MarketUtils.determine_trading_phase(market_code, now)
        trading_hours = self.config_manager.get_trading_hours(market_code.value)
        if trading_phase == TradingPhase.BEFORE_OPEN:
            # 集合竞价时段（9:00-9:30）：返回空数据，用于清空分时图
            logger.info(f"集合竞价时段（{market_code.value}），返回空数据用于清空分时图")

            # 生成空 DataFrame
            empty_df = self._generate_empty_data(symbol)

            # 构建 IntradayData（集合竞价时段也要尝试获取盘口）
            intraday_data = self._build_intraday_data(
                empty_df, symbol, trade_date,
                fetch_trade_records=True, should_poll=True, enable_cache=False
            )
        elif trading_phase == TradingPhase.AFTER_CLOSE:
            last_trade_date = MarketUtils.get_last_trade_date(market_code, trade_date, now)
            last_trade_date_cache_key = f"intraday_{symbol}_{last_trade_date}_TRADING"
            if self._enable_memory_cache:
                date_cache = self._get_from_memory_cache(last_trade_date_cache_key)
                intraday_data = IntradayData.from_any(date_cache)
            if intraday_data is None:
                # 🔧 尝试从外部获取last_trade_date的分时数据
                logger.info(f"尝试直接从外部获取最后交易日的数据: {last_trade_date}")
                try:
                    # 获取原始DataFrame
                    df = self._fetch_real_intraday_from_akshare(symbol, last_trade_date, tick_range=None)
                    if df is not None:
                        # 构建 IntradayData（盘后不获取实时盘口）
                        intraday_data = self._build_intraday_data(
                            df, symbol, last_trade_date,
                            fetch_trade_records=False, should_poll=False, enable_cache=True
                        )
                except Exception as e:
                    # 其他异常（如网络错误、API错误），记录警告并fallback
                    logger.warning(f"获取最后交易日的真实分时数据失败: {e}")
            else:
                logger.info(f"✅ 盘后缓存命中（来自最后盘中数据）: {last_trade_date_cache_key}")
        elif trading_phase == TradingPhase.NOON_BREAK:
            # 午盘休市时段（11:30-13:00）：返回上午的分时数据 + 最后的盘口
            logger.info(f"午盘休市时段（{market_code.value}），返回上午数据 + 盘口")

            # 构建上午时间范围（9:30-11:30）
            morning_start = trading_hours['open']  # 09:30
            morning_end = trading_hours['lunch_start']  # 11:30
            if tick_range is None:
                tick_range = TickRange(
                    start_time=pd.Timestamp(f"{trade_date} {morning_start}"),
                    end_time=pd.Timestamp(f"{trade_date} {morning_end}"),
                )
            # 获取上午的分时数据
            df = self._fetch_real_intraday_from_akshare(symbol, trade_date, tick_range=tick_range)

            # 构建 IntradayData（午休时段获取上午收盘时的盘口）
            intraday_data = self._build_intraday_data(
                df, symbol, trade_date,
                fetch_trade_records=True, should_poll=False, enable_cache=False
            )

        elif trading_phase == TradingPhase.TRADING:
            # 交易时段（上午或下午）：返回当前时刻之前的数据 + 实时盘口
            logger.info(f"交易时段（{trading_phase.value}），返回实时数据 + 盘口")

            # 🔧 关键：盘中必须有tick_range，如果前端未提供，则自动创建（开盘到当前时刻）
            if tick_range is None:
                # 创建从开盘到当前时刻的tick_range
                tick_range = TickRange(
                    start_time=pd.Timestamp(f"{trade_date} {trading_hours['open']}"),
                    end_time=pd.Timestamp(now),
                    period_seconds=5
                )
                logger.info(f"📅 自动创建 TickRange（盘中首次加载）: {tick_range.start_time} ~ {tick_range.end_time}")

            # 🔧 尝试从 AKShare 获取真实数据（盘中不使用缓存，实时获取）
            logger.info(f"📊 真实数据模式 - 从 AKShare 获取 (phase={trading_phase.value})")
            try:
                # 获取原始DataFrame（传入current_time用于判断时间范围）
                df = self._fetch_real_intraday_from_akshare(symbol, trade_date, tick_range)

                if df is not None:
                    # 构建 IntradayData（交易时段获取实时盘口并缓存）
                    intraday_data = self._build_intraday_data(
                        df, symbol, trade_date,
                        fetch_trade_records=True, should_poll=True, enable_cache=False
                    )
                else:
                    intraday_data = None
            except Exception as e:
                # 其他异常（如网络错误、API错误），记录警告
                logger.warning(f"获取真实分时数据失败: {e}")
                intraday_data = None

        # 如果所有尝试都失败了，抛出异常
        if intraday_data is None:
            error_msg = f"无法获取分时数据: symbol={symbol}, date={trade_date}, phase={trading_phase.value}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return intraday_data

    def _set_to_memory_cache_obj(self, cache_key: str, obj: Any):
        """
        将任意对象写入内存缓存（专门用于IntradayData等非DataFrame对象）

        Args:
            cache_key: 缓存键
            obj: 任意对象
        """
        if not self._enable_memory_cache or obj is None:
            return

        import time
        self._memory_cache[cache_key] = {
            'data': obj,
            'timestamp': time.time()
        }
        logger.debug(f"✅ 写入内存缓存: {cache_key}")
    
    def _get_from_memory_cache(self, cache_key: str) -> Any:
        """
        从内存缓存读取对象

        Args:
            cache_key: 缓存键

        Returns:
            缓存的对象或None
        """
        if not self._enable_memory_cache:
            return None

        cached = self._memory_cache.get(cache_key)
        if cached:
            logger.debug(f"✅ 内存缓存命中: {cache_key}")
            return cached.get('data')
        
        return None

    def _build_intraday_data(self, df, symbol: str, trade_date: pd.Timestamp,
                             fetch_trade_records: bool = True, should_poll: bool = True,
                             enable_cache: bool = False) -> IntradayData:
        """
        构建 IntradayData 对象（统一处理盘口获取和数据转换）

        Args:
            df: AKShare 返回的 DataFrame
            symbol: 证券代码
            trade_date: 交易日期
            fetch_trade_records: 是否获取盘口数据
            enable_cache: 是否缓存结果


        Returns:
            IntradayData 对象
        """

        # 使用 IntradayData 的类方法转换 DataFrame
        intraday_data = IntradayData.from_akshare_df(
            df, symbol, trade_date,
            interpolate_func=self._interpolate_to_5_seconds
        )
        # 获取盘口和成交明细
        if fetch_trade_records:
            order_book_bids, order_book_asks, trade_records, order_book_message, trade_records_message = \
                self._fetch_order_book_and_trades(symbol)
            # 设置盘口和成交明细
            intraday_data.order_book_bids = order_book_bids
            intraday_data.order_book_asks = order_book_asks
            intraday_data.trade_records = trade_records
            intraday_data.order_book_message = order_book_message
            intraday_data.trade_records_message = trade_records_message
        intraday_data.should_poll = should_poll
        # 缓存数据（如果需要）
        if enable_cache and self._enable_memory_cache:
            cache_key = f"intraday_{symbol}_{trade_date}_TRADING"
            self._set_to_memory_cache_obj(cache_key, intraday_data)
            logger.info(f"✅ 数据已缓存: {cache_key}")

        return intraday_data

    def _fetch_real_intraday_from_akshare(self, symbol: str, trade_date: pd.Timestamp, tick_range: TickRange = None):
        """
        从AKShare获取真实分时数据（返回原始DataFrame）

        Args:
            symbol: 证券代码（需转换为AKShare格式）
            trade_date: 交易日期
            tick_range: Tick数据时间范围（可选）

        Returns:
            pandas.DataFrame 或 None（AKShare原始格式）
        """

        if not self.available or self.ak is None:
            raise RuntimeError("AKShare不可用")

        # 转换symbol为AKShare格式（分时数据API不需要市场前缀）
        ak_symbol = self._map_to_akshare(symbol, with_market_prefix=False)

        # 构建查询时间范围
        if tick_range is not None:
            # 如果提供了 tick_range，使用其时间范围（增量获取或盘中首次加载）
            start_time = tick_range.start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_time = tick_range.end_time.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"使用tick_range时间范围: {start_time} ~ {end_time}")
        else:
            # tick_range=None：盘后获取全天数据
            from core_bak_refactored.core.share.market.market_utils import MarketUtils
            from core_bak_refactored.core.share.config_manager import ConfigManager

            # 获取市场代码
            market_code = MarketUtils.infer_market_from_symbol(symbol)
            trading_hours = self.config_manager.get_trading_hours(market_code.value)
            # 构建全天时间范围（从开盘到收盘）
            morning_start = trading_hours['open']
            afternoon_end = trading_hours['close']

            start_time = f"{trade_date} {morning_start}"
            end_time = f"{trade_date} {afternoon_end}"
            logger.info(f"📅 盘后模式，使用市场{market_code.value}的全天范围: {start_time} ~ {end_time}")

        logger.info(f"调用AKShare API: symbol={ak_symbol}, 时间范围: {start_time} ~ {end_time}")

        try:
            # 调用AKShare API获取1分钟数据
            df = self.ak.stock_zh_a_hist_min_em(
                symbol=ak_symbol,
                start_date=start_time,
                end_date=end_time,
                period='1',
                adjust=''
            )

            if df is None:
                return None
            if tick_range is None:
                # 一个完整交易日应该有270分钟的数据（09:30-12:00 = 150分钟，13:00-15:00 = 120分钟）
                expected_ticks = 270
                actual_ticks = len(df)

                logger.info(f"✅ AKShare返回 {actual_ticks} 条分时数据（期望 {expected_ticks} 条）")

                # 🔧 严格模式：如果是盘后且数据不完整（少于80%），抛出异常
                # 盘后应该返回完整的交易日数据，如果不完整说明数据源有问题
                if actual_ticks < expected_ticks * 0.8:
                    error_msg = f"盘后数据不完整：期望{expected_ticks}条，实际仅获取{actual_ticks}条。可能原因：AKShare API限制或数据源问题。"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

            # 返回原始DataFrame，由调用方进行转换
            return df

        except Exception as e:
            logger.error(f"AKShare API调用失败: {e}")
            raise

    def _fetch_realtime_order_book(self, symbol: str):
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
        if not self.available or self.ak is None:
            return [], []

        try:
            # 转换symbol为AKShare格式（盘口API需要纯数字，不需要市场前缀）
            ak_symbol = self._map_to_akshare(symbol, with_market_prefix=False)

            # 调用AKShare API获取实时盘口
            df = self.ak.stock_bid_ask_em(symbol=ak_symbol)

            if df is None or df.empty:
                logger.warning(f"无法获取盘口数据: {symbol}")
                return [], []

            # 🔧 AKShare返回长格式数据（item, value），需要转换
            # 示例：
            #    item      value
            # 0  sell_5    41.41
            # 1  sell_5_vol 13100.00
            # 2  buy_1     41.35
            # 3  buy_1_vol  1000.00

            # 将长格式转换为字典
            data_dict = {}
            for _, row in df.iterrows():
                item_name = str(row['item'])
                item_value = row['value']
                data_dict[item_name] = item_value

            # 解析盘口数据
            order_book_bids = []
            order_book_asks = []

            # 买盘（buy_1是最高价）
            for i in range(1, 6):
                price_key = f'buy_{i}'
                volume_key = f'buy_{i}_vol'  # 🔧 注意：是_vol不是_volume
                if price_key in data_dict and volume_key in data_dict:
                    price = float(data_dict[price_key])
                    volume = int(float(data_dict[volume_key]))
                    order_book_bids.append(OrderBookLevel(
                        price=round(price, 2),
                        volume=volume
                    ))

            # 卖盘（sell_1是最低价）
            for i in range(1, 6):
                price_key = f'sell_{i}'
                volume_key = f'sell_{i}_vol'  # 🔧 注意：是_vol不是_volume
                if price_key in data_dict and volume_key in data_dict:
                    price = float(data_dict[price_key])
                    volume = int(float(data_dict[volume_key]))
                    order_book_asks.append(OrderBookLevel(
                        price=round(price, 2),
                        volume=volume
                    ))

            logger.debug(f"获取盘口数据成功: {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘")
            return order_book_bids, order_book_asks

        except Exception as e:
            logger.warning(f"获取实时盘口失败: {e}")
            return [], []

    def _fetch_order_book_and_trades(self, symbol: str) -> tuple:
        """
        获取盘口和成交明细数据（仅限个股）

        Args:
            symbol: 证券代码

        Returns:
            (order_book_bids, order_book_asks, trade_records, order_book_message, trade_records_message)
        """

        order_book_bids = []
        order_book_asks = []
        trade_records = []
        order_book_message = ''
        trade_records_message = ''

        # 指数不获取盘口和成交明细
        if MarketUtils.is_index(symbol):
            order_book_message = '指数无盘口数据'
            trade_records_message = '指数无成交明细'
            return order_book_bids, order_book_asks, trade_records, order_book_message, trade_records_message

        # 获取实时盘口数据
        try:
            order_book_bids, order_book_asks = self._fetch_realtime_order_book(symbol)
            if not order_book_bids and not order_book_asks:
                order_book_message = '无法获取盘口数据'
            logger.info(f"✅ 获取实时盘口: {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘")
        except Exception as e:
            logger.warning(f"获取实时盘口失败: {e}")
            order_book_message = '无法获取盘口数据'

        # 获取实时成交明细
        try:
            trade_records = self._fetch_realtime_trade_records(symbol)
            if not trade_records:
                trade_records_message = '无法获取成交明细'
            logger.info(f"✅ 获取实时成交: {len(trade_records)}条")
        except Exception as e:
            logger.warning(f"获取实时成交明细失败: {e}")
            trade_records_message = '无法获取成交明细'

        return order_book_bids, order_book_asks, trade_records, order_book_message, trade_records_message

    def _fetch_realtime_trade_records(self, symbol: str):
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
        if not self.available or self.ak is None:
            return []

        try:
            # 转换symbol为AKShare格式（成交明细API需要市场前缀sh/sz）
            ak_symbol = self._map_to_akshare(symbol, with_market_prefix=True)

            # 调用AKShare API获取逐笔成交
            df = self.ak.stock_zh_a_tick_tx_js(symbol=ak_symbol)

            if df is None or df.empty:
                logger.warning(f"无法获取成交明细: {symbol}")
                return []

            # 解析成交明细（字段：成交时间,成交价,价格变动,成交量,成交额,性质）
            trade_records = []

            # 只取最近20条
            for _, row in df.head(20).iterrows():
                time_str = str(row.get('成交时间', ''))  # HH:MM:SS
                price = float(row.get('成交价', 0))
                volume = int(row.get('成交量', 0))
                nature = str(row.get('性质', ''))

                # 性质: '买盘' -> 'buy', '卖盘' -> 'sell', '中性盘' -> 'neutral'
                if '买' in nature:
                    direction = 'buy'
                elif '卖' in nature:
                    direction = 'sell'
                else:
                    direction = 'neutral'

                trade_records.append(TradeDetailRecord(
                    time=time_str,
                    price=round(price, 2),
                    volume=volume,
                    direction=direction
                ))

            logger.debug(f"获取成交明细成功: {len(trade_records)}条")
            return trade_records

        except Exception as e:
            logger.warning(f"获取实时成交明细失败: {e}")
            return []

    def _generate_empty_data(self, symbol: str):
        """
        生成空的DataFrame（与AKShare API返回格式一致，但包含初始化信息）

        Args:
            symbol: 证券代码

        Returns:
            pandas.DataFrame: 包含初始化信息的空DataFrame

        注意：
        - 返回空DataFrame后，会由_convert_akshare_df_to_intraday转换为IntradayData
        - DataFrame虽然为空，但会携带必要的初始化信息供转换使用
        """
        import pandas as pd
        from core_bak_refactored.core.share.market.market_utils import MarketUtils

        # 创建空的DataFrame，列名与AKShare API返回格式一致
        # AKShare返回格式：时间,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        empty_df = pd.DataFrame(columns=[
            '时间', '开盘', '收盘', '最高', '最低',
            '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
        ])

        # 在DataFrame的attrs中保存初始化信息，供_convert_akshare_df_to_intraday使用
        empty_df.attrs['_init_info'] = {
            'name': self._index_name_cache.get(symbol, symbol),
            'is_index': MarketUtils.is_index(symbol),
        }

        return empty_df

    def _interpolate_to_5_seconds(self, ticks: list) -> list:
        """
        将1分钟粒度的tick数据插值为5秒粒度（与模拟数据一致）

        策略：
        - 价格/均价：三次样条插值（平滑过渡）
        - 成交量：平均分配

        Args:
            ticks: 原始1分钟粒度的tick列表

        Returns:
            插值后的5秒粒度tick列表
        """
        if len(ticks) <= 1:
            return ticks

        from datetime import datetime, timedelta
        
        # 尝试导入scipy的三次样条插值
        try:
            from scipy.interpolate import CubicSpline
            use_cubic_spline = True
        except ImportError:
            logger.warning("未安装scipy，降级为线性插值")
            use_cubic_spline = False
            CubicSpline = None  # 定义为None以避免未定义错误

        interpolated_ticks = []

        # 准备数据：收集所有原始数据点
        times = []
        prices = []
        avg_prices = []

        for tick in ticks:
            tick_time = datetime.strptime(tick.time, '%H:%M:%S')
            times.append(tick_time)
            prices.append(tick.price)
            avg_prices.append(tick.avg_price)

        if use_cubic_spline and len(times) >= 3:
            # 使用三次样条插值（需要至少3个点）
            # 将时间转换为秒数（从第一个点开始）
            base_time = times[0]
            x_seconds = [(t - base_time).total_seconds() for t in times]

            # 创建三次样条插值函数
            cs_price = CubicSpline(x_seconds, prices, bc_type='natural')
            cs_avg_price = CubicSpline(x_seconds, avg_prices, bc_type='natural')

            # 生成插值点
            for i in range(len(ticks)):
                current_tick = ticks[i]
                current_time = times[i]
                current_seconds = x_seconds[i]

                # 添加当前分钟的第0秒数据（原始数据点）
                interpolated_ticks.append(current_tick)

                # 如果不是最后一个tick，则生成到下一个tick之间的插值点
                if i < len(ticks) - 1:
                    next_time = times[i + 1]
                    next_seconds = x_seconds[i + 1]
                    time_diff_seconds = (next_time - current_time).total_seconds()

                    # 只对相邻的分钟进行插值（差值 <= 60秒）
                    if 0 < time_diff_seconds <= 60:
                        # 计算需要插值的点数（5秒间隔）
                        num_intervals = int(time_diff_seconds / 5)

                        # 成交量平均分配
                        volume_per_interval = current_tick.volume / num_intervals if num_intervals > 0 else 0

                        # 生成中间的5秒间隔数据点
                        for j in range(1, num_intervals):
                            interpolated_seconds = current_seconds + j * 5
                            interpolated_time = base_time + timedelta(seconds=interpolated_seconds)

                            # 使用三次样条插值计算价格
                            interpolated_price = float(cs_price(interpolated_seconds))
                            interpolated_avg_price = float(cs_avg_price(interpolated_seconds))

                            interpolated_ticks.append(IntradayTickRecord(
                                time=interpolated_time.strftime('%H:%M:%S'),
                                price=round(interpolated_price, 2),
                                volume=int(volume_per_interval),
                                avg_price=round(interpolated_avg_price, 2)
                            ))
        else:
            # 降级为线性插值（scipy不可用或数据点太少）
            for i in range(len(ticks)):
                current_tick = ticks[i]
                current_time = datetime.strptime(current_tick.time, '%H:%M:%S')

                # 添加当前分钟的第0秒数据（原始数据点）
                interpolated_ticks.append(current_tick)

                # 如果不是最后一个tick，则生成到下一个tick之间的插值点
                if i < len(ticks) - 1:
                    next_tick = ticks[i + 1]
                    next_time = datetime.strptime(next_tick.time, '%H:%M:%S')

                    time_diff_seconds = (next_time - current_time).total_seconds()

                    # 只对相邻的分钟进行插值（差值 <= 60秒）
                    if 0 < time_diff_seconds <= 60:
                        # 计算需要插值的点数（5秒间隔）
                        num_intervals = int(time_diff_seconds / 5)

                        # 成交量平均分配
                        volume_per_interval = current_tick.volume / num_intervals if num_intervals > 0 else 0

                        # 生成中间的5秒间隔数据点
                        for j in range(1, num_intervals):
                            interpolated_time = current_time + timedelta(seconds=j * 5)

                            # 线性插值计算价格
                            ratio = (j * 5) / time_diff_seconds
                            interpolated_price = current_tick.price + (next_tick.price - current_tick.price) * ratio
                            interpolated_avg_price = current_tick.avg_price + (
                                    next_tick.avg_price - current_tick.avg_price) * ratio

                            interpolated_ticks.append(IntradayTickRecord(
                                time=interpolated_time.strftime('%H:%M:%S'),
                                price=round(interpolated_price, 2),
                                volume=int(volume_per_interval),
                                avg_price=round(interpolated_avg_price, 2)
                            ))

        return interpolated_ticks

    def get_realtime_kline(self, symbol: str, current_time: datetime = None) -> dict:
        """
        获取实时K线数据（领域层方法）

        职责：
        1. 自动判断交易时段
        2. 获取当天分时数据
        3. 根据分时数据计算OHLCV
        4. 缓存管理（开盘价、最高价、最低价）
        5. 盘前时段返回集合竞价价格（从盘口获取）
        6. 返回 should_poll 标志

        Args:
            symbol: 证券代码
            current_time: 当前时间（用于测试，默认使用系统时间）

        Returns:
            {
                'date': str,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int,
                'trading_phase': str,  # 交易时段：BEFORE_OPEN, TRADING, AFTER_CLOSE等
                'should_poll': bool  # 服务器根据 trading_phase 决定，前端只依赖此字段控制行为
            }
        """
        # 初始化返回数据（使用类型注解避免类型推断错误）
        from typing import Any, Dict
        kline_data: Dict[str, Any] = {
            'date': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,
            'trading_phase': None,
            'should_poll': False
        }

        # 判断交易时段
        now = pd.Timestamp.now()
        market_code = MarketUtils.infer_market_from_symbol(symbol)
        trading_phase = MarketUtils.determine_trading_phase(market_code, now)
        trade_date = now
        cache_key = f"realtime_kline_{symbol}_{trade_date}"

        if trading_phase == TradingPhase.TRADING:
            # 盘中时段：获取实时K线数据
            try:
                # 1. 尝试从缓存获取
                cached = self._get_from_memory_cache(cache_key)

                # 2. 如果无缓存，从分时数据初始化
                if not cached:
                    intraday_data = self.get_intraday_data(symbol)
                    if intraday_data.ticks and len(intraday_data.ticks) > 0:
                        prices = [tick.price for tick in intraday_data.ticks]
                        volumes = [tick.volume for tick in intraday_data.ticks]

                        cached = {
                            'date': trade_date,
                            'open': prices[0],
                            'high': max(prices),
                            'low': min(prices),
                            'close': prices[-1],
                            'volume': sum(volumes),
                            'trading_phase': trading_phase.name,
                            'should_poll': True
                        }
                        self._set_to_memory_cache_obj(cache_key, cached)
                    else:
                        # 分时数据为空时返回空K线
                        kline_data['date'] = trade_date
                        kline_data['trading_phase'] = trading_phase.name
                        kline_data['should_poll'] = True
                        return kline_data

                # 3. 使用akshare获取最新分钟数据更新K线
                ak_symbol = self._map_to_akshare(symbol, with_market_prefix=False)
                current_time_str = now.strftime('%Y-%m-%d %H:%M:%S')

                try:
                    df = self.ak.stock_zh_a_hist_min_em(
                        symbol=ak_symbol,
                        start_date=current_time_str,
                        end_date=current_time_str,
                        period='1',
                        adjust=''
                    )

                    if df is not None and not df.empty:
                        # 更新高低点和收盘价
                        latest_high = float(df['最高'].iloc[-1])
                        latest_low = float(df['最低'].iloc[-1])
                        latest_close = float(df['收盘'].iloc[-1])
                        latest_volume = int(df['成交量'].iloc[-1])

                        kline_data = {
                            'date': trade_date,
                            'open': cached['open'],  # 开盘价不变
                            'high': max(cached['high'], latest_high),
                            'low': min(cached['low'], latest_low),
                            'close': latest_close,
                            'volume': cached['volume'] + latest_volume,
                            'trading_phase': trading_phase.name,
                            'should_poll': True
                        }

                        # 更新缓存
                        self._set_to_memory_cache_obj(cache_key, kline_data)
                    else:
                        # akshare无数据时使用缓存
                        kline_data = cached.copy()
                        kline_data['trading_phase'] = trading_phase.name
                        kline_data['should_poll'] = True

                except Exception as e:
                    logger.warning(f"获取akshare分钟数据失败: {e}，使用缓存数据")
                    # API失败时使用缓存
                    kline_data = cached.copy()
                    kline_data['trading_phase'] = trading_phase.name
                    kline_data['should_poll'] = True

            except Exception as e:
                logger.error(f"获取实时K线失败: {e}")
                kline_data['date'] = trade_date
                kline_data['trading_phase'] = trading_phase.name
                kline_data['should_poll'] = True

        elif trading_phase == TradingPhase.BEFORE_OPEN:
            # 盘前时段：使用集合竞价价格（从盘口获取）
            try:
                order_book_bids, order_book_asks = self._fetch_realtime_order_book(symbol)
                auction_price = None

                if order_book_bids and len(order_book_bids) > 0:
                    # 使用买一价作为集合竞价参考价格
                    auction_price = order_book_bids[0].price

                kline_data = {
                    'date': trade_date,
                    'open': auction_price,
                    'high': auction_price,
                    'low': auction_price,
                    'close': auction_price,
                    'volume': 0,
                    'trading_phase': trading_phase.name,
                    'should_poll': True
                }
            except Exception as e:
                logger.error(f"获取集合竞价价格失败: {e}")
                kline_data['date'] = trade_date
                kline_data['trading_phase'] = trading_phase.name
                kline_data['should_poll'] = True

        # 其他时段（AFTER_CLOSE等）返回默认数据，不轮询
        kline_data['trading_phase'] = trading_phase.name
        return kline_data
