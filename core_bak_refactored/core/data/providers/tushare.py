"""
Tushare数据提供者 - A股/港股数据源
实现HistoricalDataProvider接口

职责：
- 通过Tushare Pro API获取A股和港股历史数据
- 支持指数和个股数据获取
- 数据标准化和质量验证
- 实现统一的HistoricalDataProvider接口

依赖：
pip install tushare
需要token: https://tushare.pro/register
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('DeepSeekQuant.TushareProvider')


class TushareDataProvider:
    """
    Tushare数据提供者（实现HistoricalDataProvider接口）
    
    功能：
    - A股指数/个股数据
    - 港股指数/个股数据（部分支持）
    - 自动token管理
    - 数据质量验证
    - 实现HistoricalDataProvider标准接口
    
    使用示例：
        provider = TushareDataProvider(token='your_token_here')
        data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-09-01')
        returns = provider.get_index_returns('000300.SH', '2015-06-01', '2015-09-01')
    """
    
    # 指数代码映射（Tushare代码）
    INDEX_MAPPING = {
        # A股指数
        '000300.SH': '000300.SH',      # 沪深300
        '000001.SH': '000001.SH',      # 上证指数
        '399001.SZ': '399001.SZ',      # 深证成指
        '000016.SH': '000016.SH',      # 上证50
        '000905.SH': '000905.SH',      # 中证500
        
        # 港股指数（部分）
        'HSI': 'HSI',                  # 恒生指数
        'HSCEI': 'HSCEI',              # 国企指数
    }
    
    def __init__(self, token: Optional[str] = None, fallback_to_mock: bool = True):
        """
        初始化Tushare数据提供者
        
        Args:
            token: Tushare Pro API token（可从环境变量TUSHARE_TOKEN读取）
            fallback_to_mock: 是否在失败时回退到Mock数据（默认True）
        """
        self.fallback = fallback_to_mock
        self.token = token
        self.available = False
        
        # 尝试初始化Tushare
        try:
            import tushare as ts
            import os
            
            # 优先使用传入token，否则从环境变量读取
            final_token = token or os.getenv('TUSHARE_TOKEN')
            
            if not final_token:
                logger.warning("Tushare token未配置，请设置TUSHARE_TOKEN环境变量或传入token参数")
                self.ts_pro = None
                return
            
            ts.set_token(final_token)
            self.ts_pro = ts.pro_api()
            self.available = True
            logger.info("Tushare API initialized successfully")
            
        except ImportError:
            logger.warning("tushare库未安装，请运行: pip install tushare")
            self.ts_pro = None
        except Exception as e:
            logger.warning(f"Tushare初始化失败: {e}")
            self.ts_pro = None
    
    def get_index_prices(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
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
        if not self.available or self.ts_pro is None:
            if self.fallback:
                return self._fallback_to_mock(index_id, start_date, end_date)
            else:
                raise RuntimeError("Tushare API不可用且fallback禁用")
        
        # 转换日期格式为Tushare要求的YYYYMMDD
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
            ts_code = self._map_index_to_tushare(index_id)
            logger.info(f"Fetching data for {index_id} (mapped to {ts_code}) from {start_date_str} to {end_date_str}")
            
            # 调用Tushare API
            # A股指数使用index_daily
            if ts_code.endswith('.SH') or ts_code.endswith('.SZ'):
                df = self.ts_pro.index_daily(
                    ts_code=ts_code,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            # 港股指数（部分支持）
            elif ts_code in ['HSI', 'HSCEI']:
                # 注意：Tushare对港股指数支持有限
                logger.warning(f"港股指数{ts_code}数据可能不完整，建议使用Wind")
                df = self.ts_pro.hk_index_daily(
                    ts_code=ts_code,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            else:
                raise ValueError(f"不支持的指数代码: {ts_code}")
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ts_code}")
            
            # 标准化格式
            standardized_data = self._standardize_format(df)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {index_id}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"Tushare failed for {index_id}: {e}")
            
            if self.fallback:
                logger.info(f"Falling back to Mock data for {index_id}")
                return self._fallback_to_mock(index_id, start_date, end_date)
            else:
                raise ValueError(f"Failed to fetch data for {index_id}: {e}")
    
    def get_index_returns(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.Series:
        """
        获取指数收益率序列（实现HistoricalDataProvider接口）
        
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
            symbol: 股票代码（如'600036.SH'招商银行）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
        """
        if not self.available or self.ts_pro is None:
            if self.fallback:
                return self._fallback_to_mock(symbol, start_date, end_date)
            else:
                raise RuntimeError("Tushare API不可用且fallback禁用")
        
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
            
            # A股使用daily
            if symbol.endswith('.SH') or symbol.endswith('.SZ'):
                df = self.ts_pro.daily(
                    ts_code=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            # 港股使用hk_daily
            elif symbol.endswith('.HK'):
                df = self.ts_pro.hk_daily(
                    ts_code=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            else:
                raise ValueError(f"不支持的股票代码: {symbol}")
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            standardized_data = self._standardize_format(df)
            
            logger.info(f"Successfully fetched {len(standardized_data)} rows for {symbol}")
            return standardized_data
            
        except Exception as e:
            logger.warning(f"Tushare failed for {symbol}: {e}")
            
            if self.fallback:
                return self._fallback_to_mock(symbol, start_date, end_date)
            else:
                raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def _map_index_to_tushare(self, index_id: str) -> str:
        """映射指数代码到Tushare格式"""
        return self.INDEX_MAPPING.get(index_id, index_id)
    
    def _standardize_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化Tushare数据格式
        
        Tushare列名：trade_date, close, vol
        标准格式：date, close, volume
        """
        standardized = pd.DataFrame()
        
        # 日期（Tushare格式：YYYYMMDD）
        if 'trade_date' in data.columns:
            standardized['date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        else:
            raise ValueError("No 'trade_date' column found in Tushare data")
        
        # 收盘价
        if 'close' in data.columns:
            standardized['close'] = data['close'].values
        else:
            raise ValueError("No 'close' column found in Tushare data")
        
        # 成交量（Tushare使用vol，单位：手）
        if 'vol' in data.columns:
            standardized['volume'] = data['vol'].values * 100  # 手转换为股
        else:
            logger.warning("No 'vol' column found, filling with NaN")
            standardized['volume'] = np.nan
        
        # 按日期排序
        standardized = standardized.sort_values('date').reset_index(drop=True)
        
        # 数据清洗
        original_len = len(standardized)
        standardized = standardized.dropna(subset=['close'])
        if len(standardized) < original_len:
            logger.warning(f"Removed {original_len - len(standardized)} rows with missing close prices")
        
        return standardized
    
    def _fallback_to_mock(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """回退到Mock数据"""
        from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
        
        mock_provider = MockHistoricalDataProvider()
        
        # 转换日期格式，支持 datetime 或 字符串
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        elif isinstance(start_date, str) and len(start_date) == 8:  # YYYYMMDD
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        elif isinstance(end_date, str) and len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        
        return mock_provider.get_index_prices(symbol, start_date, end_date)
    
    def test_connection(self) -> bool:
        """测试API连接"""
        if not self.available or self.ts_pro is None:
            logger.error("Tushare API未初始化")
            return False
        
        try:
            # 测试获取少量数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            df = self.ts_pro.index_daily(
                ts_code='000300.SH',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and len(df) > 0:
                logger.info(f"Connection test passed: fetched {len(df)} rows")
                return True
            else:
                logger.warning("Connection test failed: no data returned")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
