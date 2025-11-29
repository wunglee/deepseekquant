"""
数据转换器

职责：
1. 将不同数据源的格式统一为标准格式
2. 处理时区转换和时间戳标准化
3. 数据清洗和规范化
4. 支持多种数据类型转换
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pytz
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataTransformer:
    """数据转换器，统一不同数据源的格式。"""
    
    def __init__(self, target_timezone: str = 'UTC'):
        """
        初始化数据转换器。
        
        Args:
            target_timezone: 目标时区
        """
        self.target_tz = pytz.timezone(target_timezone)
    
    def transform_to_standard_format(
        self,
        data: List[Dict],
        source_type: str
    ) -> List[Dict]:
        """
        将数据转换为标准格式。
        
        Args:
            data: 原始数据列表
            source_type: 数据源类型
        
        Returns:
            标准格式数据列表
        """
        if not data:
            return []
        
        transformer_map = {
            'yahoo': self._transform_yahoo_format,
            'alphavantage': self._transform_alphavantage_format,
            'polygon': self._transform_polygon_format,
            'iex_cloud': self._transform_iex_format,
            'finnhub': self._transform_finnhub_format,
            'twelve_data': self._transform_twelve_data_format
        }
        
        transformer = transformer_map.get(source_type)
        if not transformer:
            logger.warning(f"未知的数据源类型: {source_type}")
            return data
        
        try:
            return [transformer(item) for item in data]
        except Exception as e:
            logger.error(f"数据转换失败 ({source_type}): {e}")
            return []
    
    def _transform_yahoo_format(self, item: Dict) -> Dict:
        """转换Yahoo Finance格式。"""
        return {
            'symbol': item.get('symbol'),
            'timestamp': self._normalize_timestamp(item.get('timestamp')),
            'open': float(item.get('open', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'close': float(item.get('close', 0)),
            'volume': int(item.get('volume', 0)),
            'adjusted_close': float(item.get('adjusted_close', item.get('close', 0))),
            'data_source': 'yahoo',
            'quality_score': self._calculate_quality_score(item)
        }
    
    def _transform_alphavantage_format(self, item: Dict) -> Dict:
        """转换Alpha Vantage格式。"""
        return {
            'symbol': item.get('symbol'),
            'timestamp': self._normalize_timestamp(item.get('timestamp')),
            'open': float(item.get('open', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'close': float(item.get('close', 0)),
            'volume': int(item.get('volume', 0)),
            'adjusted_close': float(item.get('adjusted_close', item.get('close', 0))),
            'data_source': 'alphavantage',
            'quality_score': self._calculate_quality_score(item)
        }
    
    def _transform_polygon_format(self, item: Dict) -> Dict:
        """转换Polygon.io格式。"""
        return {
            'symbol': item.get('symbol'),
            'timestamp': self._normalize_timestamp(item.get('timestamp')),
            'open': float(item.get('open', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'close': float(item.get('close', 0)),
            'volume': int(item.get('volume', 0)),
            'vwap': float(item.get('vwap', 0)),
            'transactions': int(item.get('transactions', 0)),
            'data_source': 'polygon',
            'quality_score': self._calculate_quality_score(item)
        }
    
    def _transform_iex_format(self, item: Dict) -> Dict:
        """转换IEX Cloud格式。"""
        return {
            'symbol': item.get('symbol'),
            'timestamp': self._normalize_timestamp(item.get('timestamp')),
            'open': float(item.get('open', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'close': float(item.get('close', 0)),
            'volume': int(item.get('volume', 0)),
            'change': float(item.get('change', 0)),
            'change_percent': float(item.get('change_percent', 0)),
            'data_source': 'iex_cloud',
            'quality_score': self._calculate_quality_score(item)
        }
    
    def _transform_finnhub_format(self, item: Dict) -> Dict:
        """转换Finnhub格式。"""
        return {
            'symbol': item.get('symbol'),
            'timestamp': self._normalize_timestamp(item.get('timestamp')),
            'open': float(item.get('open', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'close': float(item.get('close', 0)),
            'volume': int(item.get('volume', 0)),
            'data_source': 'finnhub',
            'quality_score': self._calculate_quality_score(item)
        }
    
    def _transform_twelve_data_format(self, item: Dict) -> Dict:
        """转换Twelve Data格式。"""
        return {
            'symbol': item.get('symbol'),
            'timestamp': self._normalize_timestamp(item.get('timestamp')),
            'open': float(item.get('open', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'close': float(item.get('close', 0)),
            'volume': int(item.get('volume', 0)),
            'data_source': 'twelve_data',
            'quality_score': self._calculate_quality_score(item)
        }
    
    def _normalize_timestamp(self, timestamp: Any) -> datetime:
        """
        标准化时间戳。
        
        Args:
            timestamp: 时间戳（可能是datetime、字符串或Unix时间戳）
        
        Returns:
            标准化的datetime对象
        """
        if isinstance(timestamp, datetime):
            # 确保有时区信息
            if timestamp.tzinfo is None:
                timestamp = self.target_tz.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(self.target_tz)
            return timestamp
        
        elif isinstance(timestamp, (int, float)):
            # Unix时间戳
            return datetime.fromtimestamp(timestamp, tz=self.target_tz)
        
        elif isinstance(timestamp, str):
            # 字符串格式
            try:
                dt = pd.to_datetime(timestamp)
                if dt.tzinfo is None:
                    dt = self.target_tz.localize(dt)
                return dt.to_pydatetime()
            except Exception as e:
                logger.error(f"解析时间戳失败: {timestamp}, {e}")
                return datetime.now(tz=self.target_tz)
        
        else:
            logger.warning(f"未知的时间戳类型: {type(timestamp)}")
            return datetime.now(tz=self.target_tz)
    
    def _calculate_quality_score(self, item: Dict) -> float:
        """
        计算数据质量评分。
        
        Args:
            item: 数据项
        
        Returns:
            质量评分（0-1）
        """
        score = 1.0
        
        # 检查必要字段
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            if field not in item or item[field] is None:
                score -= 0.1
        
        # 检查OHLC逻辑
        try:
            open_price = float(item.get('open', 0))
            high = float(item.get('high', 0))
            low = float(item.get('low', 0))
            close = float(item.get('close', 0))
            
            # High应该是最高价
            if high < max(open_price, close):
                score -= 0.1
            
            # Low应该是最低价
            if low > min(open_price, close):
                score -= 0.1
            
            # 价格不应为负数
            if any(p < 0 for p in [open_price, high, low, close]):
                score -= 0.2
                
        except (ValueError, TypeError):
            score -= 0.2
        
        return max(0.0, score)
    
    def convert_to_dataframe(
        self,
        data: List[Dict],
        index_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        将数据转换为DataFrame。
        
        Args:
            data: 数据列表
            index_column: 索引列名
        
        Returns:
            DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        if index_column in df.columns:
            df.set_index(index_column, inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def merge_multiple_sources(
        self,
        data_dict: Dict[str, List[Dict]],
        strategy: str = 'prefer_quality'
    ) -> List[Dict]:
        """
        合并多个数据源的数据。
        
        Args:
            data_dict: 数据源名称到数据的映射
            strategy: 合并策略（prefer_quality/average/latest等）
        
        Returns:
            合并后的数据列表
        """
        if not data_dict:
            return []
        
        # 转换所有数据为标准格式
        standardized_data = {}
        for source, data in data_dict.items():
            standardized = self.transform_to_standard_format(data, source)
            if standardized:
                standardized_data[source] = standardized
        
        if not standardized_data:
            return []
        
        # 根据策略合并
        if strategy == 'prefer_quality':
            return self._merge_prefer_quality(standardized_data)
        elif strategy == 'average':
            return self._merge_average(standardized_data)
        elif strategy == 'latest':
            return self._merge_latest(standardized_data)
        else:
            logger.warning(f"未知的合并策略: {strategy}")
            return list(standardized_data.values())[0]
    
    def _merge_prefer_quality(
        self,
        data_dict: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """优先选择高质量数据源。"""
        # 按质量评分排序数据源
        quality_scores = {}
        for source, data in data_dict.items():
            avg_quality = sum(item.get('quality_score', 0) for item in data) / len(data)
            quality_scores[source] = avg_quality
        
        # 选择质量最高的数据源
        best_source = max(quality_scores, key=quality_scores.get)
        return data_dict[best_source]
    
    def _merge_average(
        self,
        data_dict: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """平均多个数据源的值。"""
        # 简化实现：返回第一个数据源
        # 完整实现需要按时间戳对齐并平均
        return list(data_dict.values())[0]
    
    def _merge_latest(
        self,
        data_dict: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """选择最新的数据。"""
        all_data = []
        for data in data_dict.values():
            all_data.extend(data)
        
        # 按时间戳排序
        all_data.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return all_data
