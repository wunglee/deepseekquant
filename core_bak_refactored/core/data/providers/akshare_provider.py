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
import os
import time
import logging
from datetime import datetime
from typing import Union, Dict, Any, Optional, List

import akshare as ak
import pandas as pd

from core_bak_refactored.core.data.providers.protocols import (HistoricalDataProvider, PriceData,
                                                                IntradayData, IntradayTickRecord,
                                                                OrderBookLevel, TickerRecord)
from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider

logger = logging.getLogger(__name__)

class AKShareDataProvider(BaseDataProvider, HistoricalDataProvider):
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
        
        self.ak = None
        self.available = False
        self._load_us_symbol_mapping()
        self._initialize()
    
    def get_test_symbol(self) -> str:
        """获取测试符号"""
        return '000300.SH'  # 沪深300指数

    def _load_us_symbol_mapping(self):
        """加载美股符号映射配置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'config', 'us_symbol_mapping.json')
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
            # 💚 AKShare访问国内网站，使用 ConfigManager 管理代理配置
            # 临时禁用代理（避免访问国内数据源时出现问题）
            import akshare as ak
            self.ak = ak
            self.available = True
            logger.info("AKShareDataProvider initialized successfully")
        except ImportError as e:
            logger.warning(f"akshare not installed: {e}. Install with: pip install akshare")
            self.ak = None
            # 不抛出异常，允许优雅降级

        # 未来扩展：指数名称映射缓存（暂未实现）
        self._index_name_cache = None

    def get_prices(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> PriceData:
        """
        获取历史价格数据（适用于个股和指数）
        
        Args:
            symbol: 股票或指数代码（支持市场后缀，如 '000001.SZ' 或 '^GSPC'）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            标准化的价格数据
            
        设计原则：
        - 职责单一：只负责数据获取和标准化
        - 日期处理：统一转换为datetime对象
        - 异常处理：网络失败直接抛出异常，符合透明失败原则
        """
        if not self.available or self.ak is None:
            raise RuntimeError("AKShare API不可用，请安装: pip install akshare")
        
        try:
            logger.info(f"Fetching price data for {symbol} from {start_date} to {end_date}")
            
            # 根据代码格式自动判断市场，调用对应的AKShare API
            df = self._fetch_by_market(symbol)
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # 🔧 先标准化格式（处理列名差异）
            standardized_data = self._standardize_format(df)
            
            # 筛选日期范围（使用标准化后的数据）
            if 'date' not in standardized_data.columns:
                raise ValueError(f"Standardized data missing 'date' column. Columns: {standardized_data.columns.tolist()}")
            
            standardized_data['date'] = pd.to_datetime(standardized_data['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            standardized_data = standardized_data[
                (standardized_data['date'] >= start_dt) & (standardized_data['date'] <= end_dt)
            ]
            
            if standardized_data.empty:
                raise ValueError(f"No data in date range {start_date} to {end_date}")
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {symbol}")
            # 返回PriceData对象而不是原始DataFrame
            return PriceData.from_dataframe(standardized_data, symbol)
            
        except Exception as e:
            logger.error(f"AKShare failed for {symbol}: {e}")
            # 更详细的错误信息，帮助调试
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}") from e

    def get_index_prices(
        self,
        index_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> PriceData:
        """
        获取指数历史价格数据（实现HistoricalDataProvider接口）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            PriceData: 包含标准OHLCV数据的结构化对象
            
        Raises:
            ValueError: 日期格式错误或数据不可用（fallback禁用时）
        """
        # 直接委托给通用方法
        return self.get_prices(index_id, start_date, end_date)

    def get_index_returns(
        self,
        index_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
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
        price_data = self.get_index_prices(index_id, start_date, end_date)
        prices = price_data.to_dataframe().set_index('date')
        returns = prices['close'].pct_change().dropna()
        return returns

    def get_stock_prices(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> PriceData:
        """
        获取个股历史价格数据
        
        💚 注意: 此方法由基类处理缓存，不需覆写
        
        Args:
            symbol: 股票代码（支持市场后缀，如 '000001.SZ'）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            标准化的个股价格数据
        """
        # 💚 由基类自动处理缓存，此处不需实现
        # 这是为了兼容性保留的方法
        return super().get_stock_prices(symbol, start_date, end_date)
    
    def _fetch_from_external_api(self, symbol: str, start_date: str, end_date: str) -> PriceData:
        """
        从 AKShare API 获取数据（实现基类抽象方法）
        
        💚 注意:
        - 此方法仅供内部使用
        - 外部调用者应使用 get_index_prices()
        - 基类已自动处理缓存
        
        Args:
            symbol: 股票/指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            PriceData: 价格数据对象
        """
        # 复用原有的 get_prices 逻辑
        return self.get_prices(symbol, start_date, end_date)

    def _map_to_akshare(self, symbol: str) -> str:
        """
        映射代码到AKShare格式（基于规则自动转换）
        
        Args:
            symbol: 统一代码
            
        Returns:
            AKShare API所需的代码格式
            
        转换规则：
        1. A股指数：'000300.SH' → 'sh000300'，'399001.SZ' → 'sz399001'
        2. 港股指数：'HSI' → 'HSI'（保持原样）
        3. 美股指数：'^GSPC' → '标普500'（AKShare全球指数API需要中文名）
        
        设计原则：
        - 不维护写死的股票代码映射表
        - 基于代码格式规则进行自动转换
        - 快捷映射表仅为性能优化，非必需
        """
        
        # 2. A股指数：自动转换 .SH/.SZ → sh/sz前缀
        if symbol.endswith('.SH'):
            return 'sh' + symbol[:-3]  # 移除 .SH 后缀
        if symbol.endswith('.SZ'):
            return 'sz' + symbol[:-3]  # 移除 .SZ 后缀
        
        # 3. 美股指数：自动转换 ^开头 → 中文名称
        if symbol.startswith('^'):
            return self._us_symbol_to_chinese(symbol)
        
        # 4. 其他：直接返回（港股、其他市场）
        return symbol

    def _fetch_by_market(self, symbol_id: str) -> pd.DataFrame:
        """
        根据原始代码格式判断市场，调用对应的AKShare API

        Args:
            symbol_id: 原始代码
            
        Returns:
            AKShare返回的原始DataFrame
            
        设计原则：
        - 职责单一：只负责根据市场调用对应API
        - 代码格式转换已在 _map_to_akshare 完成
        - 市场判断逻辑集中在此方法
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
                # A股指数API
                logger.debug(f"调用A股指数API: stock_zh_index_daily({ak_symbol})")
                return self.ak.stock_zh_index_daily(symbol=ak_symbol)
            
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
                logger.debug(f"默认调用A股指数API: stock_zh_index_daily({ak_symbol})")
                return self.ak.stock_zh_index_daily(symbol=ak_symbol)
        except Exception as e:
            logger.error(f"AKShare API调用失败 for {symbol_id} (market: {market.value}): {e}")
            # 提供更友好的错误信息
            if "HTTPSConnectionPool" in str(e) or "proxy" in str(e).lower():
                raise ConnectionError(f"网络连接失败，请检查网络设置或代理配置: {str(e)}")
            # 重新抛出异常，让上层处理
            raise
    
    def _standardize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化指数数据格式（处理不同API的列名差异）
        
        Args:
            df: AKShare返回的原始DataFrame
        
        Returns:
            标准化的DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            
        数据标准：
        - date: pd.Timestamp 类型，交易日期时间
        - open: float，开盘价
        - high: float，最高价
        - low: float，最低价
        - close: float，收盘价
        - volume: float，成交量
        """
        # 🔧 处理不同 API 返回的列名差异
        # A股: date, open, close, high, low, volume, amount
        # 港股/美股: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
        
        # 尝试识别日期列
        date_col = None
        for col in ['日期', 'date', 'Date', 'DATE']:
            if col in df.columns:
                date_col = col
                break
        
        # 尝试识别开盘价列
        open_col = None
        for col in ['开盘', 'open', 'Open', 'OPEN']:
            if col in df.columns:
                open_col = col
                break
        
        # 尝试识别最高价列
        high_col = None
        for col in ['最高', 'high', 'High', 'HIGH']:
            if col in df.columns:
                high_col = col
                break
        
        # 尝试识别最低价列
        low_col = None
        for col in ['最低', 'low', 'Low', 'LOW']:
            if col in df.columns:
                low_col = col
                break
        
        # 尝试识别收盘价列
        close_col = None
        for col in ['收盘', 'close', 'Close', 'CLOSE', '收盘价']:
            if col in df.columns:
                close_col = col
                break
        
        # 尝试识别成交量列
        volume_col = None
        for col in ['成交量', 'volume', 'Volume', 'VOLUME']:
            if col in df.columns:
                volume_col = col
                break
        
        if not date_col or not close_col:
            raise ValueError(f"Cannot find date or close columns in DataFrame. Columns: {df.columns.tolist()}")
        
        # 如果缺少OHLC数据，使用收盘价填充
        standardized = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'open': df[open_col].astype(float) if open_col else df[close_col].astype(float),
            'high': df[high_col].astype(float) if high_col else df[close_col].astype(float),
            'low': df[low_col].astype(float) if low_col else df[close_col].astype(float),
            'close': df[close_col].astype(float),
            'volume': df[volume_col].astype(float) if volume_col else 0.0
        })
        
        # 按日期排序
        standardized = standardized.sort_values('date').reset_index(drop=True)
        
        # 数据清洗：移除NaN和异常值
        original_len = len(standardized)
        standardized = standardized.dropna(subset=['close'])
        if len(standardized) < original_len:
            logger.warning(f"Removed {original_len - len(standardized)} rows with missing close prices")
        
        return standardized
    
    def get_intraday_data(self, symbol: str, trade_date: str = None, 
                         batch_indices: Optional[List[int]] = None,
                         timestamps: Optional[List[int]] = None) -> IntradayData:
        """
        获取分时图数据（1分钟级别）
        
        实现策略：
        1. 智能日期处理：如果是周末/节假日，自动获取最近交易日数据
        2. 优先从内存缓存获取
        3. 然后尝试真实API（AKShare stock_zh_a_hist_min_em）
        4. API失败则fallback到前一交易日缓存
        5. 最后生成模拟数据
        
        Args:
            symbol: 证券代码
            trade_date: 交易日期（默认为当前日期）
            batch_indices: 批次序号数组（如 [3, 4, 5]）
            timestamps: 时间戳数组（如 [1234567890, 1234567891, 1234567892]）
        
        批次机制：
        - 模拟数据：每个批次生成12个5秒级节点，不使用timestamps
        - 真实数据：每个批次生成1个节点，时间为timestamps中对应的值
        """
        from datetime import datetime as dt, time as dt_time, timedelta
        import random
        
        logger.info(f"获取分时数据: symbol={symbol}, trade_date={trade_date}, batch_indices={batch_indices}")
        
        # 🔧 智能日期处理：如果未指定日期或指定的是非交易日，获取最近交易日
        if trade_date is None:
            trade_date = self._get_latest_trading_day()
            logger.info(f"未指定日期，使用最近交易日: {trade_date}")
        else:
            # 检查指定日期是否是交易日
            parsed_date = dt.strptime(trade_date, '%Y-%m-%d')
            if parsed_date.weekday() >= 5:  # 周末
                trade_date = self._get_latest_trading_day(trade_date)
                logger.info(f"指定日期为周末，调整为最近交易日: {trade_date}")
        
        # 🔧 1. 尝试从内存缓存获取（复用BaseProvider机制）
        # 注意：对于模拟数据，缓存key需要包含批次信息，避免不同批次返回相同数据
        cache_key = f"intraday_{symbol}_{trade_date}"
        
        # 🔧 只有在没有传递批次序号时（即请求真实数据），才使用缓存
        # 传递了批次序号的请求（模拟数据）不使用缓存
        use_cache = (batch_indices is None or len(batch_indices) == 0)
        
        if use_cache:
            cached_data = self._get_from_memory_cache(cache_key)
            if cached_data is not None:
                # 直接返回IntradayData对象
                logger.info(f"✅ 内存缓存命中: {cache_key}")
                return cached_data
        
        # 🔧 2. 关键修复：模拟模式下直接生成模拟数据，不调用真实API（避免15秒超时等待）
        # 只有当传入batch_indices时，才是模拟模式，直接跳过真实数据获取
        if batch_indices is None or len(batch_indices) == 0:
            # 没有传入batch_indices，尝试获取真实数据
            logger.info(f"📊 模式：真实数据 - 尝试从 AKShare 获取")
            try:
                intraday_data = self._fetch_real_intraday_from_akshare(symbol, trade_date)
                if intraday_data is not None:
                    # 写入内存缓存（直接存储IntradayData对象）
                    if use_cache:
                        self._set_to_memory_cache_obj(cache_key, intraday_data)
                    return intraday_data
            except Exception as e:
                logger.warning(f"获取真实分时数据失败: {e}")
            
            # 🔧 3. fallback: 尝试获取前一交易日缓存
            prev_trade_date = self._get_previous_trading_day(trade_date)
            if prev_trade_date != trade_date:
                logger.info(f"尝试使用前一交易日数据: {prev_trade_date}")
                prev_cache_key = f"intraday_{symbol}_{prev_trade_date}"
                prev_cached = self._get_from_memory_cache(prev_cache_key)
                if prev_cached is not None:
                    logger.info(f"✅ 使用前一交易日缓存: {prev_trade_date}")
                    return prev_cached
        else:
            logger.info(f"🎮 模式：模拟数据 - 批次序号 {batch_indices}")
        
        # 🔧 4. 最后生成模拟数据
        logger.warning(f"无法获取真实数据，生成模拟数据: {symbol}, batch_indices={batch_indices}")
        mock_data = self._generate_mock_intraday_data(symbol, trade_date, batch_indices, timestamps)  # 🔧 传递批次数组
        # 🔧 关键修改：模拟数据不缓存（因为不同批次序号应返回不同数据）
        # 如果后续需要缓存，应使用包含批次信息的key，例如：intraday_{symbol}_{trade_date}_batch_{batch_indices}
        return mock_data
    
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
    
    def _fetch_real_intraday_from_akshare(self, symbol: str, trade_date: str) -> IntradayData:
        """
        从AKShare获取真实分时数据
        
        Args:
            symbol: 证券代码（需转换为AKShare格式）
            trade_date: 交易日期
        
        Returns:
            IntradayData或None
        """
        from datetime import datetime as dt
        
        if not self.available or self.ak is None:
            raise RuntimeError("AKShare不可用")
        
        # 转换symbol为AKShare格式（去掉后缀）
        ak_symbol = symbol.split('.')[0]
        
        # 构建查询时间范围（当日09:30-15:00）
        start_time = f"{trade_date} 09:30:00"
        end_time = f"{trade_date} 15:00:00"
        
        logger.info(f"调用AKShare API: symbol={ak_symbol}, {start_time} ~ {end_time}")
        
        try:
            # 调用AKShare API获取1分钟数据
            df = self.ak.stock_zh_a_hist_min_em(
                symbol=ak_symbol,
                start_date=start_time,
                end_date=end_time,
                period='1',
                adjust=''
            )
            
            if df is None or df.empty:
                logger.warning(f"AKShare返回空数据: {ak_symbol}")
                return None
            
            logger.info(f"✅ AKShare返回 {len(df)} 条分时数据")
            
            # 转换为IntradayData格式（尝试获取实时盘口和成交明细）
            return self._convert_akshare_df_to_intraday(
                df, symbol, trade_date, fetch_realtime_data=True
            )
        
        except Exception as e:
            logger.error(f"AKShare API调用失败: {e}")
            raise
    
    def _convert_akshare_df_to_intraday(self, df, symbol: str, trade_date: str, 
                                         fetch_realtime_data: bool = False) -> IntradayData:
        """
        将AKShare返回的DataFrame转换为IntradayData
        
        AKShare返回格式：时间,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        
        Args:
            df: AKShare返回的DataFrame
            symbol: 证券代码
            trade_date: 交易日期
            fetch_realtime_data: 是否尝试获取实时盘口和成交明细
        
        注意：
        - 盘口和成交明细数据只在实时行情中提供
        - 历史数据不包含盘口和成交明细，这些字段将为空列表
        """
        import random
        from datetime import datetime as dt
        
        # 获取股票名称
        name_map = {
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指'
        }
        name = name_map.get(symbol, symbol)
        
        # 获取昨收价（从第一条数据推算）
        if '涨跌额' in df.columns and '收盘' in df.columns:
            first_close = float(df.iloc[0]['收盘'])
            first_change = float(df.iloc[0].get('涨跌额', 0))
            yesterday_close = first_close - first_change
        else:
            yesterday_close = float(df.iloc[0].get('收盘', 0)) * 0.99  # 估算
        
        # 构建ticks
        ticks = []
        total_volume = 0
        total_amount = 0
        
        for _, row in df.iterrows():
            time_str = str(row['时间']).split(' ')[-1][:5]  # 提取HH:MM
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
        
        # 当前价格
        current_price = ticks[-1].price if ticks else yesterday_close
        change = current_price - yesterday_close
        change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0
        
        # 🔧 关键修改：尝试获取实时盘口和成交明细数据
        order_book_bids = []
        order_book_asks = []
        tickers_list = []
        order_book_message = ''
        tickers_message = ''
        
        if fetch_realtime_data:
            # 只在当天且在交易时间内才尝试获取实时数据
            from datetime import datetime as dt, time as dt_time
            now = dt.now()
            is_today = trade_date == now.strftime('%Y-%m-%d')
            is_trading_hours = (dt_time(9, 30) <= now.time() <= dt_time(15, 0)) and (now.weekday() < 5)
            
            if is_today and is_trading_hours:
                try:
                    # 获取实时盘口数据
                    order_book_bids, order_book_asks = self._fetch_realtime_order_book(symbol)
                    # 获取实时成交明细
                    tickers_list = self._fetch_realtime_tickers(symbol)
                    logger.info(f"✅ 获取实时数据: {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘, {len(tickers_list)}条成交")
                    
                    # 如果获取失败，设置提示信息
                    if not order_book_bids and not order_book_asks:
                        order_book_message = '无法获取盘口数据'
                    if not tickers_list:
                        tickers_message = '无法获取成交明细'
                except Exception as e:
                    logger.warning(f"获取实时盘口/成交明细失败: {e}")
                    order_book_message = '无法获取盘口数据'
                    tickers_message = '无法获取成交明细'
            else:
                # 非交易时间或非交易日
                if now.weekday() >= 5:  # 周末
                    order_book_message = '非交易日'
                    tickers_message = '非交易日'
                elif not is_today:
                    order_book_message = '非当日数据'
                    tickers_message = '非当日数据'
                else:
                    order_book_message = '非交易时间'
                    tickers_message = '非交易时间'
                logger.info(f"非交易时间，盘口和成交明细为空: {order_book_message}")
        else:
            # 不获取实时数据时的默认提示
            order_book_message = '历史数据无盘口信息'
            tickers_message = '历史数据无成交明细'
        
        logger.info(f"分时数据转换完成: {len(ticks)}个tick, {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘")
        
        # 🔧 判断是否可交易：指数不可交易，个股可交易
        is_tradable = not self._is_index(symbol)
        
        return IntradayData(
            symbol=symbol,
            name=name,
            current_price=round(current_price, 2),
            yesterday_close=round(yesterday_close, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            ticks=ticks,
            order_book_bids=order_book_bids,
            order_book_asks=order_book_asks,
            tickers=tickers_list,
            trade_date=trade_date,
            order_book_message=order_book_message,
            tickers_message=tickers_message,
            is_tradable=is_tradable
        )
    
    def _get_previous_trading_day(self, trade_date: str) -> str:
        """
        获取前一交易日（简化实现：前一天，忽略节假日）
        
        TODO: 集成交易日历API
        """
        from datetime import datetime as dt, timedelta
        
        current = dt.strptime(trade_date, '%Y-%m-%d')
        prev = current - timedelta(days=1)
        
        # 跳过周末
        while prev.weekday() >= 5:  # 5=Saturday, 6=Sunday
            prev -= timedelta(days=1)
        
        return prev.strftime('%Y-%m-%d')
    
    def _is_index(self, symbol: str) -> bool:
        """
        判断是否是指数（简化版本，用于模拟测试）
        
        Args:
            symbol: 证券代码
        
        Returns:
            bool: True表示是指数（不可交易），False表示是个股（可交易）
        
        注意：
        - 这是临时的简化判断逻辑，仅用于模拟数据测试
        - 生产环境应该从数据源API获取证券类型信息
        - TODO: 未来接入真实数据源时，应调用数据源的证券信息接口
        
        模拟测试规则（方便测试两种情况）：
        - 指数（不可交易）：000开头（如000001.SH, 000300.SH）或399开头（如399001.SZ）
        - 个股（可交易）：其他代码（如600000.SH, 000001.SZ, 300001.SZ）
        """
        if not symbol:
            return False
        
        # 提取代码部分（去掉市场后缀）
        code = symbol.split('.')[0]
        market = symbol.split('.')[1] if '.' in symbol else ''
        
        # 🔧 模拟测试：简化判断，仅用于测试两种情况
        # 指数判断规则：
        # - 上海指数：000xxx.SH (如 000001.SH上证指数, 000300.SH沪深300)
        # - 深圳指数：399xxx.SZ (如 399001.SZ深证成指, 399006.SZ创业板指)
        # 注意：000001.SZ 是平安银行（个股），不是指数
        if len(code) == 6:
            # 上海市场：000开头是指数
            if market == 'SH' and code.startswith('000'):
                return True  # 上海指数
            # 深圳市场：399开头是指数，000开头是个股
            if market == 'SZ' and code.startswith('399'):
                return True  # 深圳指数
        
        # 其他都视为个股，可交易
        return False
    
    def _get_latest_trading_day(self, from_date: str = None) -> str:
        """
        获取最近的交易日（从指定日期往前推）
        
        Args:
            from_date: 起始日期（默认为今天）
        
        Returns:
            最近的交易日（YYYY-MM-DD）
        
        逻辑：
        - 如果是工作日（周一到周五），返回当天
        - 如果是周六，返回周五
        - 如果是周日，返回周五
        
        TODO: 集成交易日历API处理节假日
        """
        from datetime import datetime as dt, timedelta
        
        if from_date is None:
            current = dt.now()
        else:
            current = dt.strptime(from_date, '%Y-%m-%d')
        
        # 向前推到最近的工作日
        while current.weekday() >= 5:  # 5=周六, 6=周日
            current -= timedelta(days=1)
        
        return current.strftime('%Y-%m-%d')
    
    def _generate_mock_intraday_data(self, symbol: str, trade_date: str, batch_indices: Optional[List[int]] = None,
                                     timestamps: Optional[List[int]] = None) -> IntradayData:
        """
        生成模拟分时数据（fallback方案）
        
        Args:
            symbol: 证券代码
            trade_date: 交易日期
            batch_indices: 批次序号数组（如 [3, 4, 5]）
            timestamps: 时间戳数组（如 [1234567890, 1234567891, 1234567892]）
        
        注意：
        - 只生成分时价格和成交量数据
        - 盘口和成交明细不生成（保持为空列表）
        """
        import random
        from datetime import datetime as dt, time as dt_time, timedelta
        
        # 获取股票名称
        name_map = {
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '^GSPC': 'S&P 500',
            'AAPL': 'Apple Inc.'
        }
        name = name_map.get(symbol, symbol)
        
        now = dt.now()
        # 🔧 关键修复：base_price必须使用固定种子，确保每次请求同一股票同一日期的基准价相同
        random.seed(symbol + trade_date)  # 使用股票代码+日期作为种子
        base_price = 3000 + random.random() * 300
        yesterday_close = base_price
        logger.info(f"💰 生成基准价: {base_price:.2f} (symbol={symbol}, trade_date={trade_date})")
        
        # 🔧 关键修复：使用确定性算法直接计算任意批次的价格，保证连续性
        # 从批次0开始的累积波动总和
        if batch_indices is not None and len(batch_indices) > 0:
            first_batch_idx = batch_indices[0]
            # 计算从批次0到first_batch_idx之前的所有波动
            cumulative_change = 0.0
            for prev_batch in range(0, first_batch_idx):
                random.seed(symbol + trade_date + str(prev_batch))
                for _ in range(12):  # 每批次12个点
                    cumulative_change += (random.random() - 0.5) * 0.5
            current_price = base_price + cumulative_change
        else:
            current_price = base_price
        
        # 解析trade_date
        trade_date_obj = dt.strptime(trade_date, '%Y-%m-%d').date()
        
        # 🔧 关键修改：使用批次序号机制，与系统时间完全解耦
        # 每次请求返回当前批次往前的batch_count个批次
        # 模拟数据：每批次12个点位（5秒级，用于加速观看，1秒间隔获取1批次）
        # 真实数据：每批次1个点位（5秒级，真实环境5秒间隔获取1批次）
        import time
        
        # 计算当前批次序号：每1秒真实时间 = 1个批次（60倍速）
        current_timestamp = time.time()
        current_batch_index = int(current_timestamp) % 330  # 330个批次 = 330分钟交易时间
        
        # 🔧 计算需要生成的批次序号列表
        if batch_indices is None or len(batch_indices) == 0:
            # 🔧 首次加载：返回最近30分钟的数据（30个批次）
            initial_window = 30  # 最近30分钟
            start_batch_index = max(0, current_batch_index - initial_window + 1)
            batch_indices_to_use = list(range(start_batch_index, current_batch_index + 1))
            logger.info(f"🆕 首次加载：返回最近{initial_window}分钟 ({start_batch_index} - {current_batch_index}, 总计{len(batch_indices_to_use)}个批次)")
        else:
            # 🔧 增量更新：使用传入的批次序号数组
            batch_indices_to_use = batch_indices
            logger.info(f"🔢 增量更新：批次序号 {batch_indices_to_use}, 总计{len(batch_indices_to_use)}个批次")
        
        # 生成分时tick数据
        ticks = []
        total_volume = 0
        total_amount = 0
        
        # 🔧 循环生成每个批次的数据（模拟数据每批次12个点位）
        for batch_idx in batch_indices_to_use:
            # 计算该批次对应的虚拟时间范围（每批次1分钟，12个5秒点位）
            mock_start_time = dt.combine(trade_date_obj, dt_time(9, 30))
            
            # 将批次序号转换为交易时段内的分钟偏移
            if batch_idx < 120:  # 上午时段：0-119 -> 09:30-11:29
                batch_start_time = mock_start_time + timedelta(minutes=batch_idx)
            else:  # 下午时段：120-329 -> 13:00-14:59
                afternoon_offset = batch_idx - 120
                batch_start_time = dt.combine(trade_date_obj, dt_time(13, 0)) + timedelta(minutes=afternoon_offset)
            
            batch_end_time = batch_start_time + timedelta(minutes=1)
            
            # 🔧 关键修复：使用批次序号作为种子，但保持价格连续性
            # 每个批次内的随机波动使用固定种子，确保数据一致
            random.seed(symbol + trade_date + str(batch_idx))
            
            # 🔧 模拟数据：每批次生成12个点位（1分钟内，每5秒一个）
            for second in range(0, 60, 5):  # 0, 5, 10, 15, ..., 55
                tick_time = batch_start_time + timedelta(seconds=second)
                
                # 不能超过批次结束时间
                if tick_time >= batch_end_time:
                    break
                
                time_str = tick_time.strftime('%H:%M:%S')
                # 🔧 价格在前一个点位基础上波动（确保连续性）
                current_price += (random.random() - 0.5) * 0.5
                volume = random.randint(500, 2000)
                
                total_volume += volume
                total_amount += current_price * volume
                avg_price = total_amount / total_volume if total_volume > 0 else current_price
                
                ticks.append(IntradayTickRecord(
                    time=time_str,
                    price=round(current_price, 2),
                    volume=volume,
                    avg_price=round(avg_price, 2)
                ))
        
        change = current_price - yesterday_close
        change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0
        
        # 🔧 判断是否可交易：指数不可交易，个股可交易
        is_tradable = not self._is_index(symbol)
        
        # 🔧 关键修改：只有可交易的证券才生成盘口和成交明细
        # 盘口和成交明细基于最新价格生成，每个批次都会更新
        if is_tradable:
            # 使用当前批次的最后价格生成盘口和成交明细
            order_book_bids, order_book_asks = self._generate_mock_order_book(current_price, current_batch_index)
            tickers_list = self._generate_mock_tickers(current_price, current_batch_index, ticks[-12:] if len(ticks) >= 12 else ticks)
            order_book_message = ''
            tickers_message = ''
        else:
            # 指数不可交易，不生成盘口和成交明细
            order_book_bids = []
            order_book_asks = []
            tickers_list = []
            order_book_message = '指数不可交易'
            tickers_message = '指数不可交易'
        
        logger.info(f"生成模拟分时数据: {len(ticks)}个tick, {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘, {len(tickers_list)}条成交, is_tradable={is_tradable}")
        
        return IntradayData(
            symbol=symbol,
            name=name,
            current_price=round(current_price, 2),
            yesterday_close=round(yesterday_close, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            ticks=ticks,
            order_book_bids=order_book_bids,
            order_book_asks=order_book_asks,
            tickers=tickers_list,
            trade_date=trade_date,
            order_book_message=order_book_message,
            tickers_message=tickers_message,
            is_tradable=is_tradable
        )
    
    def _generate_mock_order_book(self, current_price: float, batch_index: int = 0):
        """
        生成模拟盘口数据
        
        Args:
            current_price: 当前价格
            batch_index: 批次序号，用于生成不同的随机数据
        """
        import random
        
        # 🔧 使用批次序号作为种子，确保每个批次生成不同的盘口
        random.seed(batch_index * 1000)
        
        order_book_bids = []
        order_book_asks = []
        
        for i in range(1, 11):
            order_book_bids.append(OrderBookLevel(
                price=round(current_price - i * 0.01, 2),
                volume=random.randint(1000, 10000)
            ))
            order_book_asks.append(OrderBookLevel(
                price=round(current_price + i * 0.01, 2),
                volume=random.randint(1000, 10000)
            ))
        
        return order_book_bids, order_book_asks
    
    def _generate_mock_tickers(self, current_price: float, batch_index: int = 0, recent_ticks: list = None):
        """
        生成模拟成交明细
        
        Args:
            current_price: 当前价格
            batch_index: 批次序号，用于生成不同的随机数据
            recent_ticks: 最近的tick数据，用于生成更真实的成交明细
        """
        import random
        from datetime import datetime as dt, timedelta
        
        # 🔧 使用批次序号作为种子，确保每个批次生成不同的成交明细
        random.seed(batch_index * 2000)
        
        tickers_list = []
        
        # 🔧 如果有最近的tick数据，基于它们生成成交明细
        if recent_ticks and len(recent_ticks) > 0:
            for i, tick in enumerate(reversed(recent_ticks[-20:])):  # 最多20条
                tickers_list.append(TickerRecord(
                    time=tick.time,
                    price=tick.price,
                    volume=random.randint(100, 500),  # 单笔成交量
                    direction=random.choice(['buy', 'sell'])
                ))
        else:
            # 如果没有tick数据，生成随机的成交明细
            now = dt.now()
            for i in range(20):
                tick_time = now - timedelta(seconds=i * 5)
                tickers_list.append(TickerRecord(
                    time=tick_time.strftime('%H:%M:%S'),
                    price=round(current_price + (random.random() - 0.5) * 0.2, 2),
                    volume=random.randint(100, 2000),
                    direction=random.choice(['buy', 'sell'])
                ))
        
        return tickers_list
    
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
            # 转换symbol为AKShare格式（去掉后缀）
            ak_symbol = symbol.split('.')[0]
            
            # 调用AKShare API获取实时盘口
            df = self.ak.stock_bid_ask_em(symbol=ak_symbol)
            
            if df is None or df.empty:
                logger.warning(f"无法获取盘口数据: {symbol}")
                return [], []
            
            # 解析盘口数据（字段：buy_1~buy_5, sell_1~sell_5）
            order_book_bids = []
            order_book_asks = []
            
            # 买盘（buy_1是最高价）
            for i in range(1, 6):
                price_key = f'buy_{i}'
                volume_key = f'buy_{i}_volume'
                if price_key in df.columns and volume_key in df.columns:
                    price = float(df[price_key].iloc[0])
                    volume = int(df[volume_key].iloc[0])
                    order_book_bids.append(OrderBookLevel(
                        price=round(price, 2),
                        volume=volume
                    ))
            
            # 卖盘（sell_1是最低价）
            for i in range(1, 6):
                price_key = f'sell_{i}'
                volume_key = f'sell_{i}_volume'
                if price_key in df.columns and volume_key in df.columns:
                    price = float(df[price_key].iloc[0])
                    volume = int(df[volume_key].iloc[0])
                    order_book_asks.append(OrderBookLevel(
                        price=round(price, 2),
                        volume=volume
                    ))
            
            logger.debug(f"获取盘口数据成功: {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘")
            return order_book_bids, order_book_asks
            
        except Exception as e:
            logger.warning(f"获取实时盘口失败: {e}")
            return [], []
    
    def _fetch_realtime_tickers(self, symbol: str):
        """
        获取实时成交明细（逐笔成交）
        
        Args:
            symbol: 证券代码（带后缀，如000300.SH）
        
        Returns:
            tickers_list: 成交明细列表
        
        注意：
        - 只在交易时间内调用此方法
        - 非交易时间返回空列表
        """
        if not self.available or self.ak is None:
            return []
        
        try:
            # 转换symbol为AKShare格式（需要加市场前缀）
            if symbol.endswith('.SH'):
                ak_symbol = 'sh' + symbol.split('.')[0]
            elif symbol.endswith('.SZ'):
                ak_symbol = 'sz' + symbol.split('.')[0]
            else:
                ak_symbol = symbol
            
            # 调用AKShare API获取逐笔成交
            df = self.ak.stock_zh_a_tick_tx_js(symbol=ak_symbol)
            
            if df is None or df.empty:
                logger.warning(f"无法获取成交明细: {symbol}")
                return []
            
            # 解析成交明细（字段：成交时间,成交价,价格变动,成交量,成交额,性质）
            tickers_list = []
            
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
                
                tickers_list.append(TickerRecord(
                    time=time_str,
                    price=round(price, 2),
                    volume=volume,
                    direction=direction
                ))
            
            logger.debug(f"获取成交明细成功: {len(tickers_list)}条")
            return tickers_list
            
        except Exception as e:
            logger.warning(f"获取实时成交明细失败: {e}")
            return []
