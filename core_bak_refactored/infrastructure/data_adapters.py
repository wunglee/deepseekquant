"""
数据源适配器 - 基础设施层
从 core_bak/data_fetcher.py 拆分
职责: 提供通用的数据源适配器（Tushare、AKShare等）
"""

import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.DataAdapters')


class DataSourceAdapter:
    """数据源适配器基类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        raise NotImplementedError
    
    def fetch_fund_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取基金数据"""
        raise NotImplementedError


class TushareAdapter(DataSourceAdapter):
    """Tushare数据源适配器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.token = config.get('tushare_token')
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从Tushare获取股票数据"""
        try:
            import tushare as ts
            ts.set_token(self.token)
            pro = ts.pro_api()
            
            df = pro.daily(
                ts_code=symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            return df
        except Exception as e:
            logger.error(f"Tushare获取数据失败: {e}")
            return pd.DataFrame()


class AKShareAdapter(DataSourceAdapter):
    """AKShare数据源适配器"""
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从AKShare获取股票数据"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            return df
        except Exception as e:
            logger.error(f"AKShare获取数据失败: {e}")
            return pd.DataFrame()


class YFinanceAdapter(DataSourceAdapter):
    """YFinance数据源适配器"""
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从YFinance获取股票数据"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            return df
        except Exception as e:
            logger.error(f"YFinance获取数据失败: {e}")
            return pd.DataFrame()
