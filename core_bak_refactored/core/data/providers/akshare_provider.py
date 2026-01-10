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
from typing import Any, Optional

import akshare as ak
import pandas as pd

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
from core_bak_refactored.core.data.providers.protocols import (PriceData,
                                                               IntradayData,
                                                               OrderBookLevel, TradeDetailRecord, IntradayTickRecord)
from core_bak_refactored.core.data.providers.protocols import TickRange
from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.core.share.market import MarketUtils
from core_bak_refactored.core.share.market.market_time_utils import MarketTimeUtils
from core_bak_refactored.core.share.market.market_enums import MarketCode, TradingPhase

logger = logging.getLogger(__name__)


def _clear_proxy(original_get, original_post, requests):
    # 恢复原始的 requests 方法
    if original_get is not None:
        requests.get = original_get
    if 'original_post' in locals():
        requests.post = original_post
    logger.info("🔧 已恢复原始 requests 方法")


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
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                'config',
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

    def _configure_proxy(self):
        """配置代理设置
        
        从 data_provider.yml 读取 use_proxy 配置，如果为 false，则禁用代理：
        1. 不再清除环境变量中的代理设置（避免影响其他组件）
        2. 设置 AKShare 不使用代理（通过 akshare 自身的配置）
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
            self._akshare_proxy_config = None  # 默认设置为 None
            if not use_proxy:
                logger.info("🚫 AKShare 配置为不使用代理（通过参数控制）")
            else:
                logger.info("🌐 AKShare 配置为使用代理")
                # 查找可用的代理设置
                import os
                proxy_vars = [
                    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy',
                    'https_proxy', 'ALL_PROXY', 'all_proxy'
                ]
                for var in proxy_vars:
                    if var in os.environ:
                        self._akshare_proxy_config = os.environ[var]
                        logger.info(f"  使用代理: {var} = {self._akshare_proxy_config}")
                        break
                if self._akshare_proxy_config:
                    logger.info("✅ AKShare 代理已设置")
                else:
                    logger.info("⚠️ 未找到代理环境变量，将使用直连")

        except Exception as e:
            logger.warning(f"配置代理时出错: {e}，将使用默认设置")

    def _fetch_history_kline_from_external_api(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                               period: str = 'daily') -> PriceData:
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
            df = self._fetch_history_df_from_external_api(symbol, start_date=start_date, end_date=end_date)

            # 🔧 关键修复：从 AKShare 获取的数据为空，返回空 PriceData（不是错误）
            # 根据业务场景，这可能是：
            # 1. 无限滚动到头了（before 早于数据源最早日期）
            # 2. 该股票/指数确实没有数据
            if df is None or df.empty:
                logger.warning(
                    f"⚠️ AKShare 返回空数据: {symbol} ({start_date} to {end_date})。可能原因：1.无限滚动到头 2.该股票没有数据")
                # 返回空 PriceData 对象，使用查询的日期范围
                return PriceData(
                    records=[],
                    symbol=symbol,
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                    count=0
                )

            # 🔧 先标准化格式（处理列名差异）
            # 使用文件顶部的全局导入
            standardized_data = MarketUtils.standardize_format(df)

            # 筛选日期范围（使用标准化后的数据）
            if 'date' not in standardized_data.columns:
                raise ValueError(
                    f"Standardized data missing 'date' column. Columns: {standardized_data.columns.tolist()}")

            standardized_data['date'] = pd.to_datetime(standardized_data['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            standardized_data = standardized_data[
                (standardized_data['date'].values >= start_dt.to_datetime64()) & (standardized_data['date'].values
                                                                                  <= end_dt.to_datetime64())]

            # 🔧 关键修复：筛选后数据为空，返回空 PriceData（不是错误）
            if standardized_data.empty:
                logger.warning(
                    f"⚠️ 日期范围筛选后数据为空: {symbol} ({start_date} to {end_date})。可能原因：1.无限滚动到头 2.该时间段没有数据")
                # 返回空 PriceData 对象，使用查询的日期范围
                return PriceData(
                    records=[],
                    symbol=symbol,
                    start_date=start_dt,  # 已经转换为 pd.Timestamp
                    end_date=end_dt,  # 已经转换为 pd.Timestamp
                    count=0
                )

            logger.info(f"Successfully fetched {len(standardized_data)} rows for {symbol}")
            # 返回PriceData对象而不是原始DataFrame
            price_data = PriceData.from_dataframe(standardized_data, symbol)

            # 🔧 AKShare 不支持直接查询周线/月线，需要从日线转换
            if period != 'daily':
                logger.info(f"AKShare 不支持直接查询 {period}，从日线转换（{price_data.count} 条日线数据）")
                # 推断市场代码（使用文件顶部的全局导入）
                market_code = MarketUtils.infer_market_from_symbol(symbol)
                price_data = self._convert_period(price_data, period, market_code)

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
        if symbol.endswith('.US') and symbol.startswith('^'):
            return self._us_symbol_to_chinese(symbol)

        # 其他：直接返回（港股、其他市场）
        return symbol

    def _fetch_history_df_from_external_api(self,
                                            symbol_id: str,
                                            start_date: pd.Timestamp = None,
                                            end_date: pd.Timestamp = None) -> pd.DataFrame:
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

        # 使用领域层工具推断市场（使用文件顶部的全局导入）
        market = MarketUtils.infer_market_from_symbol(symbol_id)

        # 检查 start_date 和 end_date 是否为 None，如果是，则抛出异常
        if start_date is None or end_date is None:
            raise ValueError(f"start_date 和 end_date 不能为 None: start_date={start_date}, end_date={end_date}")

        original_get, original_post, requests = self._config_proxy()
        df = pd.DataFrame()
        try:
            # 根据市场选择 API
            if market == MarketCode.CN:
                # 🔧 A股指数：优先使用支持日期范围的 index_zh_a_hist API
                # 移除市场前缀（index_zh_a_hist只需要纯数字代码）
                pure_code = ak_symbol.replace('sh', '').replace('sz', '')
                is_index = MarketUtils.is_index(pure_code)
                if is_index:
                    pure_code = pure_code[1:]
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')
                logger.debug(
                    f"调用A股指数API: index_zh_a_hist({pure_code}, period='daily',"
                    f" start_date={start_date_str}, end_date={end_date_str})")
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
            elif market == MarketCode.HK:
                # 港股指数API
                logger.debug(f"调用港股指数API: stock_hk_index_daily_em({ak_symbol})")
                df = self.ak.stock_hk_index_daily_em(symbol=ak_symbol)
            elif market == MarketCode.US:
                # 美股/全球指数API
                logger.debug(f"调用全球指数API: index_global_hist_em({ak_symbol})")
                df = self.ak.index_global_hist_em(symbol=ak_symbol)
            else:
                raise Exception(f"不支持的市场类型{market}")
        except Exception as e:
            logger.error(f"AKShare API调用失败 for {symbol_id} (market: {market.value}): {e}")
            # 提供更友好的错误信息
            if "HTTPSConnectionPool" in str(e) or "proxy" in str(e).lower():
                raise ConnectionError(f"网络连接失败，请检查网络设置或代理配置: {str(e)}")
            # 重新抛出异常，让上层处理
            raise
        finally:
            _clear_proxy(original_get, original_post, requests)
            return df


    def _config_proxy(self):
        import requests
        session = requests.Session()
        # 根据代理配置创建自定义session
        if self._akshare_proxy_config:
            # 使用代理
            session.proxies = {
                'http': self._akshare_proxy_config,
                'https': self._akshare_proxy_config
            }
            logger.info(f"🔧 使用代理: {self._akshare_proxy_config}")
        else:
            # 清除可能存在的代理设置
            session.trust_env = False  # 不信任环境变量中的代理设置
            logger.info("🔧 不使用代理（禁用环境代理）")

        original_get = requests.get
        original_post = requests.post

        # 替换方法
        requests.get = session.get
        requests.post = session.post

        return original_get, original_post, requests

    # _standardize_format method has been moved to MarketUtils.standardize_format

    def _fetch_today_k_column_from_external_api(self, market_local_time, symbol):
        ak_symbol = self._map_to_akshare(symbol, with_market_prefix=False)
        current_time_str = market_local_time.strftime('%Y-%m-%d %H:%M:%S')
        df = self.ak.stock_zh_a_hist_min_em(
            symbol=ak_symbol,
            start_date=current_time_str,
            end_date=current_time_str,
            period='1',
            adjust=''
        )
        return df

    def _fetch_real_intraday_from_external_api(self, symbol: str, start_time_str: str, end_time_str: str):
        """
        从AKShare获取真实分时数据（返回原始DataFrame）

        Args:
            symbol: 证券代码（需转换为AKShare格式）
            start_time_str: 开始时间
            end_time_str: 结束时间

        Returns:
            pandas.DataFrame 或 None（AKShare原始格式）
        """

        original_get, original_post, requests = self._config_proxy()
        if not self.available or self.ak is None:
            raise RuntimeError("AKShare不可用")
        # 转换symbol为AKShare格式（分时数据API不需要市场前缀）
        ak_symbol = self._map_to_akshare(symbol, with_market_prefix=False)
        logger.info(f"调用AKShare API: symbol={ak_symbol}, 时间范围: {start_time_str} ~ {end_time_str}")

        # 🔧 判断是个股还是指数，使用不同的 API
        is_index = MarketUtils.is_index(symbol)

        try:
            if is_index:
                # 指数使用 index_zh_a_hist_min_em
                logger.info(f"调用指数分时API: index_zh_a_hist_min_em({ak_symbol})")
                df = self.ak.index_zh_a_hist_min_em(
                    symbol=ak_symbol[1:],
                    start_date=start_time_str,
                    end_date=end_time_str,
                    period='1'
                )
            else:
                # 个股使用 stock_zh_a_hist_min_em
                logger.info(f"调用个股分时API: stock_zh_a_hist_min_em({ak_symbol})")
                df = self.ak.stock_zh_a_hist_min_em(
                    symbol=ak_symbol,
                    start_date=start_time_str,
                    end_date=end_time_str,
                    period='1',
                    adjust=''
                )

            if df is None or df.empty:
                logger.warning(f"⚠️ AKShare 返回空数据: {symbol}")
                return df
            # 返回原始DataFrame，由调用方进行转换
            return df

        except Exception as e:
            logger.error(f"AKShare API调用失败{symbol}:{e}")
            raise
        finally:
            _clear_proxy(original_get, original_post, requests)

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
        if not self.available or self.ak is None:
            return [], []

        original_get, original_post, requests = self._config_proxy()

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
        finally:
            _clear_proxy(original_get, original_post, requests)

    def _fetch_realtime_trade_records_from_external_api(self, symbol: str)-> list[TradeDetailRecord]:
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
        trade_records: list[TradeDetailRecord] = []
        if not self.available or self.ak is None:
            return trade_records
        original_get, original_post, requests = self._config_proxy()
        try:
            # 转换symbol为AKShare格式（成交明细API需要市场前缀sh/sz）
            ak_symbol = self._map_to_akshare(symbol, with_market_prefix=True)

            # 调用AKShare API获取逐笔成交
            df = self.ak.stock_zh_a_tick_tx_js(symbol=ak_symbol)

            if df is None or df.empty:
                logger.warning(f"无法获取成交明细: {symbol}")
                return []

            # 🔧 调试：输出AKShare返回的实际列名
            logger.info(f"🔍 AKShare成交明细列名: {df.columns.tolist()}")
            if not df.empty:
                logger.info(f"🔍 第一条数据示例: {df.head(1).to_dict('records')}")

            # 解析成交明细
            # 🔧 AKShare 返回的字段名：成交时间, 成交价格, 价格变动, 成交量, 成交额, 性质

            # 只取最近20条
            for _, row in df.head(20).iterrows():
                # 成交时间
                time_str = str(row.get('成交时间', ''))

                # 成交价格（注意是"成交价格"而不是"成交价"）
                price = float(row.get('成交价格', 0))

                # 成交量（单位：手）
                volume = int(row.get('成交量', 0))

                # 性质：买盘/卖盘/中性盘
                nature = str(row.get('性质', ''))

                # 🔧 调试：如果价格为0，记录警告
                if price == 0:
                    logger.warning(f"⚠️ 成交明细价格为0，原始数据: {row.to_dict()}")

                # 性质: '买盘' -> 'buy', '卖盘' -> 'sell', '中性盘' -> 'neutral'
                if '买盘' in nature:
                    direction = 'buy'
                elif '卖盘' in nature:
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
            return trade_records
        finally:
            _clear_proxy(original_get, original_post, requests)
            return trade_records


    def _to_IntradayData(self, df: pd.DataFrame, symbol: str, trade_date: pd.Timestamp,
                         interpolate_func=None) -> IntradayData:
        """
        从 AKShare 返回的 DataFrame 构建 IntradayData 对象

        AKShare DataFrame 格式：时间,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率

        Args:
            df: AKShare 返回的 DataFrame
            symbol: 证券代码
            trade_date: 交易日期
            interpolate_func: 插值函数（可选，用于将1分钟数据插值为5秒数据）

        Returns:
            IntradayData 对象（不包含盘口和成交明细）

        注意：
        - 本方法仅负责 DataFrame 到 IntradayData 的数据转换
        - 盘口和成交明细为空，由调用方设置
        """
        from core_bak_refactored.core.share.market.market_utils import MarketUtils
        import logging
        logger = logging.getLogger(__name__)

        # 获取股票名称
        name_map = {
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指'
        }
        name = name_map.get(symbol, symbol)

        # 处理空DataFrame（集合竞价时段等）
        if df.empty:
            yesterday_close = 0.0
            ticks = []
            current_price = 0.0
            change = 0.0
            change_percent = 0.0
        else:
            # 获取昨收价（从第一条数据推算）
            if '涨跌额' in df.columns and '收盘' in df.columns:
                first_close = float(df.iloc[0]['收盘'])
                first_change = float(df.iloc[0].get('涨跌额', 0))
                yesterday_close = first_close - first_change
            else:
                yesterday_close = float(df.iloc[0].get('收盘', 0)) * 0.99  # 估算

            # 构建 ticks
            ticks = []
            total_volume = 0
            total_amount = 0

            for _, row in df.iterrows():
                time_str = str(row['时间']).split(' ')[-1]  # 提取时间部分 HH:MM:SS
                # 确保时间格式为 HH:MM:SS
                if len(time_str) == 5:  # HH:MM
                    time_str += ':00'
                elif len(time_str) > 8:  # 可能包含毫秒，截断到秒
                    time_str = time_str[:8]

                price = float(row.get('收盘', row.get('最新价', 0)))
                volume = int(row.get('成交量', 0))

                total_volume += volume
                total_amount += price * volume
                avg_price = total_amount / total_volume if total_volume > 0 else price

                ticks.append(IntradayTickRecord(
                    time=time_str,
                    price=round(price, 2),
                    volume=volume,
                    avg_price=round(avg_price, 2)
                ))

            # 插值：将1分钟数据插值为5秒数据（如果提供了插值函数）
            if interpolate_func and ticks:
                ticks = interpolate_func(ticks)
                logger.info(f"📊 插值完成: {len(ticks)}个5秒粒度数据点")

            # 当前价格
            current_price = ticks[-1].price if ticks else yesterday_close
            change = current_price - yesterday_close
            change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0

        # 判断是否为指数
        is_index = MarketUtils.is_index(symbol)

        return IntradayData(
            symbol=symbol,
            name=name,
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
