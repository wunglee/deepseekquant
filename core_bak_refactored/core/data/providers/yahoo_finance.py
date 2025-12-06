"""
Yahoo Finance数据提供者 - 整合版

整合来源：
1. yahoo_finance.py (HistoricalDataProvider体系) - 主体
2. yahoo.py (DataFetcher体系) - 增强功能

功能范围：
- 实现HistoricalDataProvider接口（get_index_prices, get_index_returns）
- 支持灵活的时间参数（period/interval 或 start_date/end_date）
- 支持多种数据类型（ohlcv, dividends, splits, all）
- 数据质量验证与标准化
- 异常处理与日志记录

设计原则：
- 接口优先：实现HistoricalDataProvider统一接口
- 功能增强：整合DataFetcher的灵活参数支持
- 健壮性：完整的错误处理和质量验证
- 可扩展：支持指数、个股、波动率等多种数据
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

logger = logging.getLogger('DeepSeekQuant.YahooFinanceProvider')


# TODO: 从quality_types.py迁移到此处,避免外部依赖
@dataclass
class DataQualityReport:
    """数据质量报告"""
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    outliers_detected: int
    total_rows: int
    missing_values: int
    overall_score: float


class YahooFinanceDataProvider:
    """
    雅虎财经数据提供者（实现HistoricalDataProvider接口）
    
    功能：
    - 通过yfinance API获取真实历史指数价格数据
    - 自动映射国内指数代码到Yahoo Finance ticker
    - 数据质量验证与清洗
    - 完整的错误处理与日志记录
    
    示例：
        provider = YahooFinanceDataProvider()
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
    
    def __init__(self):
        """
        初始化Yahoo Finance数据提供者
        """
        self._session = None
        
        # 延迟导入yfinance（避免环境依赖问题）
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("YahooFinanceDataProvider initialized successfully")
        except ImportError:
            logger.error("yfinance not installed. Please run: pip install yfinance")
            self.yf = None
            raise RuntimeError("yfinance library not available. Please install: pip install yfinance")
    
    def get_index_prices(
        self,
        index_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        include_ohlcv: bool = False
    ) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
            include_ohlcv: 是否包含完整OHLCV数据（默认False，仅返回close和volume）
        
        Returns:
            DataFrame with columns:
            - 默认: ['date', 'close', 'volume']
            - include_ohlcv=True: ['date', 'open', 'high', 'low', 'close', 'volume']
            
        Raises:
            ValueError: 日期格式错误或数据不可用
        """
        # Convert datetime objects to string format if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
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
            standardized_data = self._standardize_format(data, include_ohlcv=include_ohlcv)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {index_id}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {index_id}: {e}")
            raise ValueError(f"Failed to fetch data for {index_id}: {e}")
    
    def get_index_returns(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            Series with date index and return values
        """
        # Convert datetime objects to string format if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        prices = self.get_index_prices(index_id, start_date, end_date)
        prices = prices.set_index('date')
        returns = prices['close'].pct_change().dropna()
        return returns
    
    def get_stock_prices(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        获取个股价格数据
        
        Args:
            symbol: 股票代码（如'600036.SS'招商银行）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
            
        Raises:
            ValueError: 日期格式错误或数据不可用
        """
        # Convert datetime objects to string format if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        try:
            logger.info(f"Fetching stock data for {symbol} from {start_date} to {end_date}")
            
            # 1. 调用yfinance API
            if self.yf is None:
                raise RuntimeError("yfinance not available")
            
            data = self.yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # 2. 数据质量验证
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # 3. 标准化格式
            standardized_data = self._standardize_format(data)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {symbol}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def get_volatility_index(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.Series:
        """
        获取波动率指数（如VIX）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            Series with date index and volatility values
        """
        # Convert datetime objects to string format if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # VIX指数在Yahoo Finance中的ticker
        volatility_tickers = {
            'VIX': '^VIX',      # CBOE波动率指数
            'VXFXI': '^VXFXI',  # 中国波指
        }
        
        ticker = volatility_tickers.get(index_id, index_id)
        
        try:
            if self.yf is None:
                raise RuntimeError("yfinance not available")
            
            data = self.yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # 提取收盘价作为波动率数据
            if 'Close' in data.columns:
                volatility_data = data['Close']
            elif 'close' in data.columns:
                volatility_data = data['close']
            else:
                raise ValueError("No 'Close' column found in Yahoo Finance data")
            
            return volatility_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch volatility index {index_id}: {e}")
            raise ValueError(f"Failed to fetch volatility data for {index_id}: {e}")
    
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
    
    def _standardize_format(self, data: pd.DataFrame, include_ohlcv: bool = False) -> pd.DataFrame:
        """
        标准化yfinance数据格式
        
        Args:
            data: yfinance返回的原始DataFrame
            include_ohlcv: 是否包含完整OHLCV数据
        
        Returns:
            标准化DataFrame with columns:
            - 默认: ['date', 'close', 'volume']
            - include_ohlcv=True: ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        # yfinance返回的列名可能是大写或小写
        standardized = pd.DataFrame()
        
        # 提取date（从index）
        standardized['date'] = data.index
        
        # 如果需要完整OHLCV数据
        if include_ohlcv:
            # 提取Open
            if 'Open' in data.columns:
                standardized['open'] = data['Open'].values
            elif 'open' in data.columns:
                standardized['open'] = data['open'].values
            else:
                standardized['open'] = np.nan
            
            # 提取High
            if 'High' in data.columns:
                standardized['high'] = data['High'].values
            elif 'high' in data.columns:
                standardized['high'] = data['high'].values
            else:
                standardized['high'] = np.nan
            
            # 提取Low
            if 'Low' in data.columns:
                standardized['low'] = data['Low'].values
            elif 'low' in data.columns:
                standardized['low'] = data['low'].values
            else:
                standardized['low'] = np.nan
        
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
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        数据质量验证报告
        
        Args:
            data: 待验证的数据DataFrame
            
        Returns:
            DataQualityReport: 数据质量报告
        """
        if data is None or data.empty:
            return DataQualityReport(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                outliers_detected=0,
                total_rows=0,
                missing_values=0
            )
        
        total_rows = len(data)
        
        # 计算缺失值
        missing_values = data.isnull().sum().sum()
        
        # 完整性评分 (基于缺失值比例)
        completeness_score = 1.0 - (missing_values / (total_rows * len(data.columns))) if total_rows > 0 else 0.0
        
        # 一致性评分 (检查数据类型一致性)
        consistency_score = self._calculate_consistency_score(data)
        
        # 准确性评分 (基于数据范围检查)
        accuracy_score = self._calculate_accuracy_score(data)
        
        # 异常值检测
        outliers_detected = self._detect_outliers(data)
        
        return DataQualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            outliers_detected=outliers_detected,
            total_rows=total_rows,
            missing_values=missing_values
        )
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算数据一致性评分"""
        if data.empty:
            return 0.0
            
        # 检查数值列的数据类型一致性
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return 0.5  # 如果没有数值列，给中等分数
            
        # 检查是否有混合类型的数据
        consistency_issues = 0
        for col in numeric_columns:
            if data[col].dtype == 'object':
                # 尝试转换为数值
                try:
                    pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    consistency_issues += 1
        
        # 计算一致性评分
        consistency_score = 1.0 - (consistency_issues / len(numeric_columns)) if len(numeric_columns) > 0 else 1.0
        return max(0.0, consistency_score)  # 确保不为负数
    
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """计算数据准确性评分"""
        if data.empty:
            return 0.0
            
        accuracy_score = 1.0
        
        # 检查价格数据是否合理
        if 'close' in data.columns:
            close_prices = data['close']
            # 检查是否有负价格
            negative_prices = (close_prices < 0).sum()
            if negative_prices > 0:
                accuracy_score -= 0.2 * (negative_prices / len(close_prices))
            
            # 检查是否有异常大的价格
            if len(close_prices) > 0:
                mean_price = close_prices.mean()
                if mean_price > 0:
                    # 检查超过均值10倍的价格
                    extreme_prices = (close_prices > mean_price * 10).sum()
                    if extreme_prices > 0:
                        accuracy_score -= 0.1 * (extreme_prices / len(close_prices))
        
        # 确保评分在0-1范围内
        return max(0.0, min(1.0, accuracy_score))
    
    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """检测异常值数量"""
        if data.empty:
            return 0
            
        outliers = 0
        
        # 对数值列进行异常值检测
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            series = data[col]
            # 使用IQR方法检测异常值
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outliers += col_outliers
            
        return outliers
    
    def get_dividends(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        获取股息数据（从yahoo.py整合）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame with columns: ['date', 'dividends']
        """
        # Convert datetime to string
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        try:
            if self.yf is None:
                raise RuntimeError("yfinance not available")
            
            ticker = self.yf.Ticker(symbol)
            dividends = ticker.dividends
            
            if dividends.empty:
                logger.warning(f"No dividends data for {symbol}")
                return pd.DataFrame(columns=['date', 'dividends'])
            
            # 转换为DataFrame
            df = dividends.to_frame(name='dividends')
            df['date'] = df.index
            df = df.reset_index(drop=True)
            
            # 筛选日期范围
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df[mask]
            
            return df[['date', 'dividends']]
            
        except Exception as e:
            logger.error(f"Failed to fetch dividends for {symbol}: {e}")
            return pd.DataFrame(columns=['date', 'dividends'])
    
    def get_splits(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        获取股票拆分数据（从yahoo.py整合）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame with columns: ['date', 'splits']
        """
        # Convert datetime to string
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        try:
            if self.yf is None:
                raise RuntimeError("yfinance not available")
            
            ticker = self.yf.Ticker(symbol)
            splits = ticker.splits
            
            if splits.empty:
                logger.info(f"No splits data for {symbol}")
                return pd.DataFrame(columns=['date', 'splits'])
            
            # 转换为DataFrame
            df = splits.to_frame(name='splits')
            df['date'] = df.index
            df = df.reset_index(drop=True)
            
            # 筛选日期范围
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df[mask]
            
            return df[['date', 'splits']]
            
        except Exception as e:
            logger.error(f"Failed to fetch splits for {symbol}: {e}")
            return pd.DataFrame(columns=['date', 'splits'])
    
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
