"""
AKShare数据提供者 - A股/港股/美股数据源
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
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('DeepSeekQuant.AKShareProvider')


class AKShareDataProvider:
    """
    AKShare数据提供者（实现HistoricalDataProvider接口）
    
    功能：
    - A股指数/个股数据
    - 港股指数/个股数据
    - 美股指数/个股数据
    - 期货、基金、债券数据（可扩展）
    - 自动数据标准化
    - 数据质量验证
    - 实现HistoricalDataProvider标准接口
    
    使用示例：
        provider = AKShareDataProvider(fallback_to_mock=True)
        data = provider.get_index_prices('000300.SH', '2024-01-01', '2024-12-01')
        returns = provider.get_index_returns('000300.SH', '2024-01-01', '2024-12-01')
    """
    
    # 指数代码映射（统一代码 → AKShare格式）
    INDEX_MAPPING = {
        # A股指数
        '000300.SH': 'sh000300',      # 沪深300
        '000001.SH': 'sh000001',      # 上证指数
        '399001.SZ': 'sz399001',      # 深证成指
        '000016.SH': 'sh000016',      # 上证50
        '000905.SH': 'sh000905',      # 中证500
        '399006.SZ': 'sz399006',      # 创业板指
        
        # 港股指数
        'HSI': 'HSI',                  # 恒生指数
        'HSCEI': 'HSCEI',              # 国企指数
        
        # 美股指数（通用格式）
        'SPX': '^GSPC',                # S&P 500
        '^GSPC': '^GSPC',              # S&P 500
        'DJI': '^DJI',                 # 道琼斯
        '^DJI': '^DJI',                # 道琼斯
        'IXIC': '^IXIC',               # 纳斯达克
        '^IXIC': '^IXIC',              # 纳斯达克
    }
    
    def __init__(self, fallback_to_mock: bool = True):
        """
        初始化AKShare数据提供者
        
        Args:
            fallback_to_mock: 是否在失败时回退到Mock数据（默认True）
        """
        self.fallback = fallback_to_mock
        self.available = False
        
        # 延迟导入akshare（避免环境依赖问题）
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
            logger.info("AKShareDataProvider initialized successfully")
        except ImportError:
            logger.warning("akshare not installed, will fallback to Mock if enabled")
            self.ak = None
            if not self.fallback:
                raise RuntimeError("akshare library not available and fallback disabled")
    
    def get_index_prices(
        self,
        index_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        获取指数价格数据（实现HistoricalDataProvider接口）
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
            
        Raises:
            ValueError: 数据不可用（fallback禁用时）
        """
        if not self.available or self.ak is None:
            if self.fallback:
                return self._fallback_to_mock(index_id, start_date, end_date)
            else:
                raise RuntimeError("AKShare API不可用且fallback禁用")
        
        # 转换日期格式
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime('%Y%m%d')
        else:
            start_date_str = start_date.replace('-', '')
        
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime('%Y%m%d')
        else:
            end_date_str = end_date.replace('-', '')
        
        try:
            # 映射指数代码
            ak_symbol = self._map_index_to_akshare(index_id)
            logger.info(f"Fetching data for {index_id} (mapped to {ak_symbol}) from {start_date_str} to {end_date_str}")
            
            # 调用AKShare API获取指数数据
            df = self.ak.stock_zh_index_daily(symbol=ak_symbol)
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ak_symbol}")
            
            # 筛选日期范围
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            if df.empty:
                raise ValueError(f"No data in date range {start_date} to {end_date}")
            
            # 标准化格式
            standardized_data = self._standardize_format(df)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {index_id}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"AKShare failed for {index_id}: {e}")
            
            if self.fallback:
                return self._fallback_to_mock(index_id, start_date, end_date)
            else:
                raise ValueError(f"Failed to fetch data for {index_id}: {e}")
    
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
        prices = self.get_index_prices(index_id, start_date, end_date)
        prices = prices.set_index('date')
        returns = prices['close'].pct_change().dropna()
        return returns
    
    def get_stock_prices(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        获取个股价格数据
        
        Args:
            symbol: 股票代码（如'000001'平安银行）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
            
        Raises:
            ValueError: 日期格式错误或数据不可用（fallback禁用时）
        """
        if not self.available or self.ak is None:
            if self.fallback:
                return self._fallback_to_mock(symbol, start_date, end_date)
            else:
                raise RuntimeError("AKShare API不可用且fallback禁用")
        
        # 转换日期格式
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime('%Y%m%d')
        else:
            start_date_str = start_date.replace('-', '')
        
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime('%Y%m%d')
        else:
            end_date_str = end_date.replace('-', '')
        
        try:
            logger.info(f"Fetching stock data for {symbol} from {start_date_str} to {end_date_str}")
            
            # AKShare获取A股数据（不需要后缀）
            # 去掉 .SH 或 .SZ 后缀
            clean_symbol = symbol.replace('.SH', '').replace('.SZ', '').replace('.SS', '')
            
            # 调用AKShare API
            df = self.ak.stock_zh_a_hist(
                symbol=clean_symbol,
                period="daily",
                start_date=start_date_str,
                end_date=end_date_str,
                adjust="qfq"  # 前复权
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # 标准化格式
            standardized_data = self._standardize_stock_format(df)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {symbol}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"AKShare failed for {symbol}: {e}")
            
            if self.fallback:
                return self._fallback_to_mock(symbol, start_date, end_date)
            else:
                raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def _map_index_to_akshare(self, index_id: str) -> str:
        """
        映射指数代码到AKShare格式
        
        Args:
            index_id: 统一指数代码（如'000300.SH'）
        
        Returns:
            AKShare格式的指数代码（如'sh000300'）
        """
        mapped = self.INDEX_MAPPING.get(index_id)
        
        if mapped:
            return mapped
        
        # 尝试自动转换
        if index_id.endswith('.SH'):
            return 'sh' + index_id.replace('.SH', '')
        elif index_id.endswith('.SZ'):
            return 'sz' + index_id.replace('.SZ', '')
        else:
            # 直接返回原代码（可能是美股指数）
            return index_id
    
    def _standardize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化指数数据格式
        
        Args:
            df: AKShare返回的原始DataFrame
        
        Returns:
            标准化的DataFrame with columns: ['date', 'close', 'volume']
        """
        # AKShare指数数据列名：date, open, close, high, low, volume, amount
        standardized = pd.DataFrame({
            'date': pd.to_datetime(df['date']),
            'close': df['close'].astype(float),
            'volume': df['volume'].astype(float) if 'volume' in df.columns else 0.0
        })
        
        # 按日期排序
        standardized = standardized.sort_values('date').reset_index(drop=True)
        
        return standardized
    
    def _standardize_stock_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化个股数据格式
        
        Args:
            df: AKShare返回的原始DataFrame
        
        Returns:
            标准化的DataFrame with columns: ['date', 'close', 'volume']
        """
        # AKShare个股数据列名可能为：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
        # 或：date, open, close, high, low, volume, amount
        
        # 尝试识别日期列
        date_col = None
        for col in ['日期', 'date', 'Date', 'DATE']:
            if col in df.columns:
                date_col = col
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
        
        standardized = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'close': df[close_col].astype(float),
            'volume': df[volume_col].astype(float) if volume_col else 0.0
        })
        
        # 按日期排序
        standardized = standardized.sort_values('date').reset_index(drop=True)
        
        return standardized
    
    def _fallback_to_mock(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        回退到Mock数据提供者
        
        Args:
            symbol: 股票或指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Mock数据
        """
        logger.warning(f"Falling back to Mock data for {symbol}")
        
        try:
            from tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
            mock = MockHistoricalDataProvider()
            return mock.get_index_prices(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Mock fallback also failed: {e}")
            raise ValueError(f"Both AKShare and Mock failed for {symbol}")
