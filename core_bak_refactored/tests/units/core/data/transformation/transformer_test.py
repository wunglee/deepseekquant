import pytest
from unittest.mock import Mock
from datetime import datetime
import pytz
import pandas as pd
from core_bak_refactored.core.data.transformation.transformer import DataTransformer


class TestDataTransformer:
    """测试数据转换器。"""

    def test_init(self):
        """测试初始化。"""
        transformer = DataTransformer('US/Eastern')
        assert transformer.target_tz == pytz.timezone('US/Eastern')

    def test_transform_yahoo_format(self):
        """测试Yahoo格式转换。"""
        transformer = DataTransformer()
        
        data = [{
            'symbol': 'AAPL',
            'timestamp': datetime(2024, 1, 1),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000,
            'adjusted_close': 151.0
        }]
        
        result = transformer.transform_to_standard_format(data, 'yahoo')
        
        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['open'] == 150.0
        assert result[0]['data_source'] == 'yahoo'
        assert 'quality_score' in result[0]

    def test_transform_polygon_format(self):
        """测试Polygon格式转换。"""
        transformer = DataTransformer()
        
        data = [{
            'symbol': 'GOOGL',
            'timestamp': datetime(2024, 1, 1),
            'open': 140.0,
            'high': 142.0,
            'low': 139.0,
            'close': 141.0,
            'volume': 500000,
            'vwap': 140.5,
            'transactions': 1000
        }]
        
        result = transformer.transform_to_standard_format(data, 'polygon')
        
        assert len(result) == 1
        assert result[0]['symbol'] == 'GOOGL'
        assert result[0]['vwap'] == 140.5
        assert result[0]['data_source'] == 'polygon'

    def test_normalize_timestamp_datetime(self):
        """测试datetime时间戳标准化。"""
        transformer = DataTransformer('UTC')
        
        # 无时区信息
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = transformer._normalize_timestamp(dt)
        assert result.tzinfo is not None
        
        # 有时区信息
        dt_aware = pytz.timezone('US/Eastern').localize(datetime(2024, 1, 1))
        result = transformer._normalize_timestamp(dt_aware)
        assert result.tzinfo is not None

    def test_normalize_timestamp_unix(self):
        """测试Unix时间戳标准化。"""
        transformer = DataTransformer('UTC')
        
        unix_ts = 1704096000  # 2024-01-01 00:00:00 UTC
        result = transformer._normalize_timestamp(unix_ts)
        
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_normalize_timestamp_string(self):
        """测试字符串时间戳标准化。"""
        transformer = DataTransformer('UTC')
        
        ts_str = '2024-01-01 12:00:00'
        result = transformer._normalize_timestamp(ts_str)
        
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_calculate_quality_score_perfect(self):
        """测试完美数据的质量评分。"""
        transformer = DataTransformer()
        
        item = {
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }
        
        score = transformer._calculate_quality_score(item)
        assert score == 1.0

    def test_calculate_quality_score_missing_fields(self):
        """测试缺少字段的质量评分。"""
        transformer = DataTransformer()
        
        item = {
            'open': 150.0,
            'high': 152.0,
            'low': 149.0
            # 缺少 close 和 volume
        }
        
        score = transformer._calculate_quality_score(item)
        assert score < 1.0

    def test_calculate_quality_score_invalid_ohlc(self):
        """测试无效OHLC的质量评分。"""
        transformer = DataTransformer()
        
        # high应该大于等于open和close
        item = {
            'open': 150.0,
            'high': 148.0,  # 无效：high低于open
            'low': 147.0,
            'close': 149.0,
            'volume': 1000000
        }
        
        score = transformer._calculate_quality_score(item)
        assert score < 1.0

    def test_convert_to_dataframe(self):
        """测试转换为DataFrame。"""
        transformer = DataTransformer()
        
        data = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1),
                'close': 150.0
            },
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 2),
                'close': 151.0
            }
        ]
        
        df = transformer.convert_to_dataframe(data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.index.name == 'timestamp'

    def test_convert_to_dataframe_empty(self):
        """测试空数据转DataFrame。"""
        transformer = DataTransformer()
        
        df = transformer.convert_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_merge_prefer_quality(self):
        """测试按质量偏好合并。"""
        transformer = DataTransformer()
        
        data_dict = {
            'yahoo': [{
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1),
                'open': 150.0,
                'high': 152.0,
                'low': 149.0,
                'close': 151.0,
                'volume': 1000000
            }],
            'polygon': [{
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1),
                'open': 150.5,
                'high': 152.5,
                'low': 149.5,
                'close': 151.5,
                'volume': 1100000,
                'vwap': 151.0,
                'transactions': 1000
            }]
        }
        
        result = transformer.merge_multiple_sources(data_dict, 'prefer_quality')
        
        assert len(result) > 0
        assert result[0]['symbol'] == 'AAPL'

    def test_merge_latest(self):
        """测试按最新时间合并。"""
        transformer = DataTransformer()
        
        data_dict = {
            'source1': [{
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1),
                'close': 150.0
            }],
            'source2': [{
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 2),
                'close': 151.0
            }]
        }
        
        result = transformer.merge_multiple_sources(data_dict, 'latest')
        
        assert len(result) == 2
        # 应该按时间倒序排列
        assert result[0]['timestamp'] >= result[1]['timestamp']

    def test_transform_unknown_source(self):
        """测试未知数据源类型。"""
        transformer = DataTransformer()
        
        data = [{'symbol': 'AAPL'}]
        result = transformer.transform_to_standard_format(data, 'unknown_source')
        
        # 应该返回原始数据
        assert result == data
