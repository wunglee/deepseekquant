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
from typing import Union, Dict, Any

import akshare as ak
import pandas as pd

from core_bak_refactored.core.data.providers.protocols import HistoricalDataProvider, PriceData
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
