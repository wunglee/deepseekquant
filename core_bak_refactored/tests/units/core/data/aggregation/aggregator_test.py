import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from core_bak_refactored.core.data.aggregation.aggregator import DataAggregator


class TestDataAggregator:
    """测试数据聚合器。"""

    def test_init(self):
        """测试初始化。"""
        aggregator = DataAggregator()
        assert aggregator is not None

    def test_aggregate_ohlcv(self):
        """测试OHLCV聚合。"""
        aggregator = DataAggregator()
        
        # 创建分钟级数据
        data = []
        base_time = datetime(2024, 1, 1, 9, 30)
        for i in range(60):
            data.append({
                'symbol': 'AAPL',
                'timestamp': base_time + timedelta(minutes=i),
                'open': 150.0 + i * 0.1,
                'high': 151.0 + i * 0.1,
                'low': 149.0 + i * 0.1,
                'close': 150.5 + i * 0.1,
                'volume': 1000
            })
        
        # 聚合到小时级
        result = aggregator.aggregate_ohlcv(data, '1h')
        
        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['volume'] == 60000  # 60 * 1000

    def test_aggregate_ohlcv_empty(self):
        """测试空数据聚合。"""
        aggregator = DataAggregator()
        result = aggregator.aggregate_ohlcv([], '1h')
        assert result == []

    def test_map_interval_to_freq(self):
        """测试间隔映射。"""
        aggregator = DataAggregator()
        
        assert aggregator._map_interval_to_freq('1m') == '1T'
        assert aggregator._map_interval_to_freq('1h') == '1H'
        assert aggregator._map_interval_to_freq('1d') == '1D'
        assert aggregator._map_interval_to_freq('1wk') == '1W'
        assert aggregator._map_interval_to_freq('unknown') == '1D'

    def test_calculate_rolling_metrics(self):
        """测试滚动指标计算。"""
        aggregator = DataAggregator()
        
        data = []
        base_time = datetime(2024, 1, 1)
        for i in range(20):
            data.append({
                'timestamp': base_time + timedelta(days=i),
                'close': 150.0 + i
            })
        
        result = aggregator.calculate_rolling_metrics(
            data, window=5, metrics=['mean', 'std']
        )
        
        assert len(result) == 20
        assert 'rolling_mean_5' in result[-1]
        assert 'rolling_std_5' in result[-1]

    def test_calculate_rolling_metrics_empty(self):
        """测试空数据滚动计算。"""
        aggregator = DataAggregator()
        result = aggregator.calculate_rolling_metrics([], 5, ['mean'])
        assert result == []

    def test_aggregate_by_symbol(self):
        """测试按股票聚合。"""
        aggregator = DataAggregator()
        
        data = [
            {'symbol': 'AAPL', 'close': 150.0},
            {'symbol': 'AAPL', 'close': 151.0},
            {'symbol': 'GOOGL', 'close': 140.0},
            {'symbol': 'GOOGL', 'close': 142.0}
        ]
        
        result = aggregator.aggregate_by_symbol(data, 'mean')
        
        assert len(result) == 2
        assert result['AAPL'] == 150.5
        assert result['GOOGL'] == 141.0

    def test_aggregate_by_symbol_sum(self):
        """测试按股票求和聚合。"""
        aggregator = DataAggregator()
        
        data = [
            {'symbol': 'AAPL', 'close': 100.0},
            {'symbol': 'AAPL', 'close': 200.0}
        ]
        
        result = aggregator.aggregate_by_symbol(data, 'sum')
        
        assert result['AAPL'] == 300.0

    def test_calculate_period_statistics(self):
        """测试周期统计计算。"""
        aggregator = DataAggregator()
        
        data = [{'close': i * 10.0} for i in range(1, 11)]
        
        stats = aggregator.calculate_period_statistics(data)
        
        assert stats['count'] == 10
        assert stats['mean'] == 55.0
        assert stats['min'] == 10.0
        assert stats['max'] == 100.0
        assert 'std' in stats
        assert 'return_mean' in stats

    def test_calculate_period_statistics_empty(self):
        """测试空数据统计。"""
        aggregator = DataAggregator()
        stats = aggregator.calculate_period_statistics([])
        assert stats == {}

    def test_downsample(self):
        """测试降采样。"""
        aggregator = DataAggregator()
        
        data = [{'index': i} for i in range(100)]
        
        result = aggregator.downsample(data, sample_rate=10)
        
        assert len(result) == 10
        assert result[0]['index'] == 0
        assert result[1]['index'] == 10

    def test_downsample_empty(self):
        """测试空数据降采样。"""
        aggregator = DataAggregator()
        result = aggregator.downsample([], 5)
        assert result == []

    def test_group_by_time_period_day(self):
        """测试按天分组。"""
        aggregator = DataAggregator()
        
        data = []
        base_time = datetime(2024, 1, 1, 10, 0)
        for i in range(48):
            data.append({
                'timestamp': base_time + timedelta(hours=i),
                'close': 150.0
            })
        
        result = aggregator.group_by_time_period(data, 'day')
        
        assert len(result) == 2  # 2天

    def test_group_by_time_period_month(self):
        """测试按月分组。"""
        aggregator = DataAggregator()
        
        data = []
        base_time = datetime(2024, 1, 1)
        for i in range(60):
            data.append({
                'timestamp': base_time + timedelta(days=i),
                'close': 150.0
            })
        
        result = aggregator.group_by_time_period(data, 'month')
        
        assert len(result) >= 2  # 至少2个月

    def test_calculate_vwap(self):
        """测试VWAP计算。"""
        aggregator = DataAggregator()
        
        data = [
            {'close': 100.0, 'volume': 1000},
            {'close': 110.0, 'volume': 2000},
            {'close': 120.0, 'volume': 1000}
        ]
        
        vwap = aggregator.calculate_vwap(data)
        
        # VWAP = (100*1000 + 110*2000 + 120*1000) / (1000+2000+1000)
        expected = (100*1000 + 110*2000 + 120*1000) / 4000
        assert abs(vwap - expected) < 0.01

    def test_calculate_vwap_empty(self):
        """测试空数据VWAP。"""
        aggregator = DataAggregator()
        vwap = aggregator.calculate_vwap([])
        assert vwap == 0.0

    def test_merge_data_sources_concat(self):
        """测试连接合并。"""
        aggregator = DataAggregator()
        
        data1 = [{'symbol': 'AAPL', 'close': 150.0}]
        data2 = [{'symbol': 'GOOGL', 'close': 140.0}]
        
        result = aggregator.merge_data_sources([data1, data2], 'concat')
        
        assert len(result) == 2

    def test_merge_data_sources_union(self):
        """测试去重合并。"""
        aggregator = DataAggregator()
        
        data1 = [
            {'symbol': 'AAPL', 'timestamp': datetime(2024, 1, 1), 'close': 150.0}
        ]
        data2 = [
            {'symbol': 'AAPL', 'timestamp': datetime(2024, 1, 1), 'close': 150.0},
            {'symbol': 'GOOGL', 'timestamp': datetime(2024, 1, 1), 'close': 140.0}
        ]
        
        result = aggregator.merge_data_sources([data1, data2], 'union')
        
        # 应该去重
        assert len(result) == 2

    def test_merge_data_sources_empty(self):
        """测试空数据合并。"""
        aggregator = DataAggregator()
        result = aggregator.merge_data_sources([], 'concat')
        assert result == []
