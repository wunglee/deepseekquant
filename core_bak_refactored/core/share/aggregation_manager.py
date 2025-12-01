"""
聚合管理器（共享模块）

职责：提供标准化的数据聚合管理接口
用途：统一管理各种数据聚合操作
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger('DeepSeekQuant.Core.Share.AggregationManager')


class AggregationManager:
    """
    聚合管理器
    
    职责：提供标准化的数据聚合管理接口
    """
    
    def __init__(self):
        self._aggregators = {}
    
    def register_aggregator(self, name: str, aggregator_func):
        """
        注册聚合器
        
        Args:
            name: 聚合器名称
            aggregator_func: 聚合函数
        """
        self._aggregators[name] = aggregator_func
        logger.debug(f"注册聚合器: {name}")
    
    def aggregate_ohlcv(
        self,
        data: List[Dict],
        target_interval: str
    ) -> List[Dict]:
        """
        聚合OHLCV数据到更大的时间间隔
        
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
        映射时间间隔到pandas频率字符串
        
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
        计算滚动窗口指标
        
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
    
    def aggregate_by_field(
        self,
        data: List[Dict],
        group_by_field: str,
        aggregation_func: str = 'mean'
    ) -> Dict[str, float]:
        """
        按字段聚合
        
        Args:
            data: 数据列表
            group_by_field: 分组字段
            aggregation_func: 聚合函数
            
        Returns:
            字段值到聚合值的映射
        """
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        if group_by_field not in df.columns or 'close' not in df.columns:
            logger.error("数据缺少必要字段")
            return {}
        
        # 按字段分组聚合
        if aggregation_func == 'mean':
            result = df.groupby(group_by_field)['close'].mean()
        elif aggregation_func == 'sum':
            result = df.groupby(group_by_field)['close'].sum()
        elif aggregation_func == 'max':
            result = df.groupby(group_by_field)['close'].max()
        elif aggregation_func == 'min':
            result = df.groupby(group_by_field)['close'].min()
        elif aggregation_func == 'count':
            result = df.groupby(group_by_field)['close'].count()
        else:
            logger.warning(f"未知的聚合函数: {aggregation_func}")
            return {}
        
        return result.to_dict()
    
    def calculate_statistics(
        self,
        data: List[Dict],
        value_field: str = 'close'
    ) -> Dict[str, Any]:
        """
        计算统计指标
        
        Args:
            data: 数据列表
            value_field: 数值字段名
            
        Returns:
            统计指标字典
        """
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        if value_field not in df.columns:
            return {}
        
        values = df[value_field].values
        
        statistics = {
            'count': len(values),
            'mean': float(pd.Series(values).mean()),
            'median': float(pd.Series(values).median()),
            'std': float(pd.Series(values).std()),
            'min': float(pd.Series(values).min()),
            'max': float(pd.Series(values).max()),
            'range': float(pd.Series(values).max() - pd.Series(values).min())
        }
        
        # 计算回报率
        if len(values) > 1:
            returns = pd.Series(values).pct_change().dropna()
            statistics['return_mean'] = float(returns.mean())
            statistics['return_std'] = float(returns.std())
            statistics['return_min'] = float(returns.min())
            statistics['return_max'] = float(returns.max())
        
        return statistics
    
    def downsample(
        self,
        data: List[Dict],
        sample_rate: int
    ) -> List[Dict]:
        """
        降采样数据
        
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
        按时间周期分组
        
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