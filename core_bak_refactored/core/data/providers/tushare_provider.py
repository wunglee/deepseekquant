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

优势：
- A股数据质量高
- 港股数据较为完整
- 需要注册获取token

配置：
- 需要Token（从 https://tushare.pro/register 注册获取）
- 环境变量：TUSHARE_TOKEN
- 或通过credentials.yml配置
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Any, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import os

# 导入新的数据结构
from core_bak_refactored.core.data.providers.protocols import PriceData, OHLCVRecord
from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
from core_bak_refactored.core.risk.backtest_framework import HistoricalDataProvider

logger = logging.getLogger('DeepSeekQuant.TushareProvider')


@dataclass
class DataQualityReport:
    """数据质量报告"""
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    outliers_detected: int
    total_rows: int
    missing_values: int
    overall_score: float = field(init=False)
    
    def __post_init__(self):
        self.overall_score = (self.completeness_score + self.consistency_score + self.accuracy_score) / 3


class TushareDataProvider(BaseDataProvider, HistoricalDataProvider):
    """
    Tushare数据提供者（实现HistoricalDataProvider接口）
    
    功能：
    - A股指数/个股数据
    - 港股指数/个股数据（部分支持）
    - 自动token管理
    - 数据质量验证
    - 实现HistoricalDataProvider标准接口
    - 完全符合HistoricalDataProvider协议标准
    
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
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化Tushare数据提供者
        
        Args:
            token: Tushare Pro API token（可从环境变量TUSHARE_TOKEN读取）
        """
        # 优先级：传入参数 > 环境变量 > 配置文件 > None
        token = token or os.getenv('TUSHARE_TOKEN') or self._load_token_from_config()
        self.ts_pro = None
        
        # 尝试初始化Tushare
        try:
            import tushare as ts
            
            if token:
                ts.set_token(token)
                self.ts_pro = ts.pro_api()
                logger.info(f"Tushare API initialized successfully with token: {self.token[:8]}...")
            else:
                self.ts_pro = None
                logger.warning("Tushare token未配置，请设置TUSHARE_TOKEN环境变量或传入token参数")
            
        except ImportError:
            logger.error("tushare库未安装，请运行: pip install tushare")
            self.ts_pro = None
            # 不抛出异常，允许实例创建但标记为不可用
        except Exception as e:
            logger.error(f"Tushare初始化失败: {e}")
            self.ts_pro = None
            # 不抛出异常，允许实例创建但标记为不可用
    
    def initialize(self, token: str = None, **kwargs):
        """
        初始化客户端
        
        Args:
            token: Tushare Token
            **kwargs: 其他初始化参数
        """
        # 使用传入的token或已有token
        if token:
            try:
                import tushare as ts
                ts.set_token(token)
                self.ts_pro = ts.pro_api()
                logger.info(f"Tushare客户端初始化成功: {token[:8]}...")
            except Exception as e:
                logger.error(f"Tushare客户端初始化失败: {e}")
                self.ts_pro = None
        else:
            logger.warning("未提供Token，无法初始化Tushare客户端")

    def _load_token_from_config(self) -> Optional[str]:
        """
        从配置文件加载Token
        
        Returns:
            Token字符串，如果未找到则返回None
        """
        try:
            # 使用基类方法获取配置路径
            config_path = self._get_config_path('credentials.yml')
            
            # 检查配置文件是否存在
            if not config_path.exists():
                logger.debug("凭证配置文件不存在")
                return None
            
            # 读取配置文件
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                credentials_data = yaml.safe_load(f) or {}
            
            # 获取Tushare的Token
            tushare_creds = credentials_data.get('tushare', {})
            token = tushare_creds.get('token')
            
            if token:
                logger.debug("从配置文件加载Tushare Token成功")
                return token
            else:
                logger.debug("配置文件中未找到Tushare Token")
                return None
                
        except Exception as e:
            logger.warning(f"从配置文件加载Token失败: {e}")
            return None

    def get_test_symbol(self) -> str:
        """获取测试符号"""
        return '000300.SH'  # 沪深300指数
    
    def get_index_prices(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> PriceData:
        """
        获取指数价格数据（实现HistoricalDataProvider接口）
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            PriceData: 包含标准OHLCV数据的结构化对象
            
        Raises:
            ValueError: 数据不可用
        """
        if not self.ts_pro:
            raise RuntimeError("Tushare API不可用，请配置 TUSHARE_TOKEN")
        
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
            # 返回PriceData对象而不是原始DataFrame
            return PriceData.from_dataframe(standardized_data, index_id)
            
        except Exception as e:
            logger.error(f"Tushare failed for {index_id}: {e}")
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
            
        price_data = self.get_index_prices(index_id, start_date, end_date)
        prices = price_data.to_dataframe().set_index('date')
        returns = prices['close'].pct_change().dropna()
        return returns
    
    def get_stock_prices(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> PriceData:
        """
        获取个股价格数据（实现HistoricalDataProvider接口）
        
        Args:
            symbol: 股票代码（如'600036.SH'招商银行）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            PriceData: 包含标准OHLCV数据的结构化对象
        """
        if not self.ts_pro:
            raise RuntimeError("Tushare API不可用，请配置 TUSHARE_TOKEN")
        
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
            # 返回PriceData对象而不是原始DataFrame
            return PriceData.from_dataframe(standardized_data, symbol)
            
        except Exception as e:
            logger.error(f"Tushare failed for {symbol}: {e}")
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def _map_index_to_tushare(self, index_id: str) -> str:
        """映射指数代码到Tushare格式"""
        return self.INDEX_MAPPING.get(index_id, index_id)
    
    def _standardize_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化Tushare数据格式（实现HistoricalDataProvider协议标准）
        
        Tushare列名：trade_date, close, vol
        标准格式：date, open, high, low, close, volume
        
        Args:
            data: Tushare返回的原始DataFrame
        
        Returns:
            标准化DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            
        数据标准：
        - date: pd.Timestamp 类型，交易日期时间
        - open: float，开盘价
        - high: float，最高价
        - low: float，最低价
        - close: float，收盘价
        - volume: float，成交量
        """
        standardized = pd.DataFrame()
        
        # 日期（Tushare格式：YYYYMMDD）
        if 'trade_date' in data.columns:
            standardized['date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        else:
            raise ValueError("No 'trade_date' column found in Tushare data")
        
        # 开盘价（如果不存在，使用收盘价填充）
        if 'open' in data.columns:
            standardized['open'] = data['open'].values
        else:
            standardized['open'] = data['close'].values if 'close' in data.columns else np.nan
        
        # 最高价（如果不存在，使用收盘价填充）
        if 'high' in data.columns:
            standardized['high'] = data['high'].values
        else:
            standardized['high'] = data['close'].values if 'close' in data.columns else np.nan
        
        # 最低价（如果不存在，使用收盘价填充）
        if 'low' in data.columns:
            standardized['low'] = data['low'].values
        else:
            standardized['low'] = data['close'].values if 'close' in data.columns else np.nan
        
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
        
        # 确保所有数值列为float类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in standardized.columns:
                standardized[col] = standardized[col].astype(float)
        
        return standardized
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        数据质量验证报告（实现HistoricalDataProvider接口）
        
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
    
    def test_connection(self) -> bool:
        """测试API连接"""
        if not self.ts_pro:
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