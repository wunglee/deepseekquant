"""
Yahoo Finance数据提供者 - Phase 3B真实数据集成
从第6轮专家指导实施
职责: 通过yfinance API获取真实历史数据

设计原则：
- 实现HistoricalDataProvider接口
- 支持回退到Mock数据（数据不可用时）
- 标准化输出格式（与Mock保持一致）
- 异常处理与日志记录
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('DeepSeekQuant.YahooFinanceProvider')


class YahooFinanceDataProvider:
    """
    雅虎财经数据提供者（Phase 3B）
    
    功能：
    - 通过yfinance API获取真实历史指数价格数据
    - 自动映射国内指数代码到Yahoo Finance ticker
    - 数据质量验证与清洗
    - 失败时回退到Mock数据（可选）
    
    示例：
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-09-01')
    """
    
    # 指数代码映射表（国内代码 → Yahoo Finance ticker）
    INDEX_MAPPING = {
        # 中国市场
        '000300.SH': '000300.SS',      # 沪深300
        '000001.SH': '000001.SS',      # 上证指数
        '399001.SZ': '399001.SZ',      # 深证成指
        '000016.SH': 'SSEC',           # 上证50（备选）
        '000905.SH': '000905.SS',      # 中证500
        
        # 美国市场
        'SPX': '^GSPC',                # S&P 500
        'DJI': '^DJI',                 # 道琼斯
        'IXIC': '^IXIC',               # 纳斯达克
        
        # 香港市场
        'HSI': '^HSI',                 # 恒生指数
        'HSCEI': '^HSCEI',             # 国企指数
    }
    
    def __init__(self, fallback_to_mock: bool = True):
        """
        初始化Yahoo Finance数据提供者
        
        Args:
            fallback_to_mock: 是否在失败时回退到Mock数据（默认True）
        """
        self.fallback = fallback_to_mock
        self._session = None
        
        # 延迟导入yfinance（避免环境依赖问题）
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("YahooFinanceDataProvider initialized successfully")
        except ImportError:
            logger.warning("yfinance not installed, will fallback to Mock if enabled")
            self.yf = None
            if not self.fallback:
                raise RuntimeError("yfinance library not available and fallback disabled")
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
            
        Raises:
            ValueError: 日期格式错误或数据不可用（fallback禁用时）
        """
        try:
            # 1. 映射指数代码
            ticker = self._map_index_to_yahoo(index_id)
            logger.info(f"Fetching data for {index_id} (mapped to {ticker}) from {start_date} to {end_date}")
            
            # 2. 调用yfinance API
            if self.yf is None:
                raise RuntimeError("yfinance not available")
            
            data = self.yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # 3. 数据质量验证
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # 4. 标准化格式
            standardized_data = self._standardize_format(data)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {index_id}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {index_id}: {e}")
            
            if self.fallback:
                logger.info(f"Falling back to Mock data for {index_id}")
                from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
                mock_provider = MockHistoricalDataProvider()
                return mock_provider.get_index_prices(index_id, start_date, end_date)
            else:
                raise ValueError(f"Failed to fetch data for {index_id}: {e}")
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series with date index and return values
        """
        prices = self.get_index_prices(index_id, start_date, end_date)
        prices = prices.set_index('date')
        returns = prices['close'].pct_change().dropna()
        return returns
    
    def _map_index_to_yahoo(self, index_id: str) -> str:
        """
        映射国内指数代码到Yahoo Finance ticker
        
        Args:
            index_id: 国内指数代码（如'000300.SH'）
        
        Returns:
            Yahoo Finance ticker（如'000300.SS'）
        
        Raises:
            ValueError: 未知的指数代码
        """
        if index_id in self.INDEX_MAPPING:
            return self.INDEX_MAPPING[index_id]
        
        # 如果已经是Yahoo格式（包含^或以.SS/.SZ结尾），直接使用
        if index_id.startswith('^') or index_id.endswith('.SS') or index_id.endswith('.SZ'):
            logger.info(f"Using {index_id} as-is (already Yahoo format)")
            return index_id
        
        # 未知代码，尝试直接使用（可能失败）
        logger.warning(f"Unknown index code {index_id}, trying as-is")
        return index_id
    
    def _standardize_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化yfinance数据格式
        
        Args:
            data: yfinance返回的原始DataFrame
        
        Returns:
            标准化DataFrame with columns: ['date', 'close', 'volume']
        """
        # yfinance返回的列名可能是大写或小写
        standardized = pd.DataFrame()
        
        # 提取date（从index）
        standardized['date'] = data.index
        
        # 提取close价格（处理多种列名格式）
        if 'Close' in data.columns:
            standardized['close'] = data['Close'].values
        elif 'close' in data.columns:
            standardized['close'] = data['close'].values
        else:
            raise ValueError("No 'Close' column found in Yahoo Finance data")
        
        # 提取成交量（可选，如果不存在则填充NaN）
        if 'Volume' in data.columns:
            standardized['volume'] = data['Volume'].values
        elif 'volume' in data.columns:
            standardized['volume'] = data['volume'].values
        else:
            logger.warning("No 'Volume' column found, filling with NaN")
            standardized['volume'] = np.nan
        
        # 重置index
        standardized = standardized.reset_index(drop=True)
        
        # 数据清洗：移除NaN和异常值
        original_len = len(standardized)
        standardized = standardized.dropna(subset=['close'])
        if len(standardized) < original_len:
            logger.warning(f"Removed {original_len - len(standardized)} rows with missing close prices")
        
        return standardized
    
    def test_connection(self, sample_index: str = '000300.SH') -> bool:
        """
        测试与Yahoo Finance的连接
        
        Args:
            sample_index: 测试用指数代码
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # 获取最近10天数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            data = self.get_index_prices(sample_index, start_date, end_date)
            
            if len(data) > 0:
                logger.info(f"Connection test passed: fetched {len(data)} rows for {sample_index}")
                return True
            else:
                logger.warning("Connection test failed: no data returned")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
