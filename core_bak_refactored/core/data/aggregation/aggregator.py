"""
数据聚合器

职责：
1. 聚合多个时间粒度的数据
2. 计算聚合统计指标
3. 重采样和降采样
4. 滚动窗口聚合
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataAggregator:
    """数据聚合器，支持多种聚合操作。"""
    
    def __init__(self):
        """初始化数据聚合器。"""
        pass
    
    def aggregate_ohlcv(
        self,
        data: List[Dict],
        target_interval: str
    ) -> List[Dict]:
        """
        将OHLCV数据聚合到更大的时间间隔。
        
        Args:
            data: 原始OHLCV数据
            target_interval: 目标时间间隔
        
        Returns:
            聚合后的数据
        """
        if not data:
            return []
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        if 'timestamp' not in df.columns:
            logger.error("数据缺少timestamp字段")
            return []
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 映射时间间隔到pandas频率
        freq = self._map_interval_to_freq(target_interval)
        
        # 聚合规则
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # 重采样
        resampled = df.resample(freq).agg(agg_rules)
        
        # 删除空行
        resampled.dropna(subset=['close'], inplace=True)
        
        # 转换回字典列表
        result = []
        for timestamp, row in resampled.iterrows():
            item = {
                'timestamp': timestamp.to_pydatetime(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            }
            
            # 保留symbol
            if 'symbol' in df.columns:
                item['symbol'] = df['symbol'].iloc[0]
            
            result.append(item)
        
        logger.info(f"聚合完成：{len(data)} -> {len(result)} 条记录")
        return result
    
    def _map_interval_to_freq(self, interval: str) -> str:
        """
        映射时间间隔到pandas频率字符串。
        
        Args:
            interval: 时间间隔
        
        Returns:
            pandas频率字符串
        """
        mapping = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '1d': '1D',
            '1wk': '1W',
            '1mo': '1M'
        }
        
        return mapping.get(interval, '1D')
    
    def calculate_rolling_metrics(
        self,
        data: List[Dict],
        window: int,
        metrics: List[str]
    ) -> List[Dict]:
        """
        计算滚动窗口指标。
        
        Args:
            data: 原始数据
            window: 窗口大小
            metrics: 指标列表（mean/std/max/min等）
        
        Returns:
            包含滚动指标的数据
        """
        if not data or window <= 0:
            return []
        
        df = pd.DataFrame(data)
        
        if 'timestamp' not in df.columns or 'close' not in df.columns:
            logger.error("数据缺少必要字段")
            return []
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 计算滚动指标
        for metric in metrics:
            if metric == 'mean':
                df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            elif metric == 'std':
                df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            elif metric == 'max':
                df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
            elif metric == 'min':
                df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            elif metric == 'sum':
                df[f'rolling_sum_{window}'] = df['close'].rolling(window=window).sum()
        
        # 转换回字典列表
        df.reset_index(inplace=True)
        return df.to_dict('records')
    
    def aggregate_by_symbol(
        self,
        data: List[Dict],
        aggregation_func: str = 'mean'
    ) -> Dict[str, float]:
        """
        按股票代码聚合。
        
        Args:
            data: 数据列表
            aggregation_func: 聚合函数
        
        Returns:
            股票代码到聚合值的映射
        """
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        if 'symbol' not in df.columns or 'close' not in df.columns:
            logger.error("数据缺少必要字段")
            return {}
        
        # 按symbol分组聚合
        if aggregation_func == 'mean':
            result = df.groupby('symbol')['close'].mean()
        elif aggregation_func == 'sum':
            result = df.groupby('symbol')['close'].sum()
        elif aggregation_func == 'max':
            result = df.groupby('symbol')['close'].max()
        elif aggregation_func == 'min':
            result = df.groupby('symbol')['close'].min()
        elif aggregation_func == 'count':
            result = df.groupby('symbol')['close'].count()
        else:
            logger.warning(f"未知的聚合函数: {aggregation_func}")
            return {}
        
        return result.to_dict()
    
    def calculate_period_statistics(
        self,
        data: List[Dict]
    ) -> Dict[str, Any]:
        """
        计算周期统计指标。
        
        Args:
            data: 数据列表
        
        Returns:
            统计指标字典
        """
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        if 'close' not in df.columns:
            return {}
        
        prices = df['close'].values
        
        statistics = {
            'count': len(prices),
            'mean': float(np.mean(prices)),
            'median': float(np.median(prices)),
            'std': float(np.std(prices)),
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'range': float(np.max(prices) - np.min(prices)),
            'percentile_25': float(np.percentile(prices, 25)),
            'percentile_75': float(np.percentile(prices, 75))
        }
        
        # 计算回报率
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            statistics['return_mean'] = float(np.mean(returns))
            statistics['return_std'] = float(np.std(returns))
            statistics['return_min'] = float(np.min(returns))
            statistics['return_max'] = float(np.max(returns))
        
        return statistics
    
    def downsample(
        self,
        data: List[Dict],
        sample_rate: int
    ) -> List[Dict]:
        """
        降采样数据。
        
        Args:
            data: 原始数据
            sample_rate: 采样率（每N条取1条）
        
        Returns:
            降采样后的数据
        """
        if not data or sample_rate <= 0:
            return []
        
        return data[::sample_rate]
    
    def group_by_time_period(
        self,
        data: List[Dict],
        period: str
    ) -> Dict[str, List[Dict]]:
        """
        按时间周期分组。
        
        Args:
            data: 数据列表
            period: 周期（day/week/month/year）
        
        Returns:
            周期到数据的映射
        """
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        if 'timestamp' not in df.columns:
            logger.error("数据缺少timestamp字段")
            return {}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 根据周期分组
        if period == 'day':
            df['group_key'] = df['timestamp'].dt.date
        elif period == 'week':
            df['group_key'] = df['timestamp'].dt.to_period('W')
        elif period == 'month':
            df['group_key'] = df['timestamp'].dt.to_period('M')
        elif period == 'year':
            df['group_key'] = df['timestamp'].dt.year
        else:
            logger.warning(f"未知的周期类型: {period}")
            return {}
        
        # 分组
        grouped = {}
        for key, group_df in df.groupby('group_key'):
            group_df = group_df.drop('group_key', axis=1)
            grouped[str(key)] = group_df.to_dict('records')
        
        return grouped
    
    def calculate_vwap(
        self,
        data: List[Dict]
    ) -> float:
        """
        计算成交量加权平均价（VWAP）。
        
        Args:
            data: OHLCV数据
        
        Returns:
            VWAP值
        """
        if not data:
            return 0.0
        
        df = pd.DataFrame(data)
        
        if 'close' not in df.columns or 'volume' not in df.columns:
            logger.error("数据缺少close或volume字段")
            return 0.0
        
        # VWAP = sum(price * volume) / sum(volume)
        total_value = (df['close'] * df['volume']).sum()
        total_volume = df['volume'].sum()
        
        if total_volume == 0:
            return 0.0
        
        return float(total_value / total_volume)
    
    def merge_data_sources(
        self,
        data_list: List[List[Dict]],
        merge_strategy: str = 'concat'
    ) -> List[Dict]:
        """
        合并多个数据源。
        
        Args:
            data_list: 数据源列表
            merge_strategy: 合并策略（concat/union/intersect）
        
        Returns:
            合并后的数据
        """
        if not data_list:
            return []
        
        if merge_strategy == 'concat':
            # 简单连接
            result = []
            for data in data_list:
                result.extend(data)
            return result
        
        elif merge_strategy == 'union':
            # 去重合并
            all_data = []
            for data in data_list:
                all_data.extend(data)
            
            df = pd.DataFrame(all_data)
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                df.drop_duplicates(subset=['timestamp', 'symbol'], inplace=True)
            
            return df.to_dict('records')
        
        else:
            logger.warning(f"未知的合并策略: {merge_strategy}")
            return []
