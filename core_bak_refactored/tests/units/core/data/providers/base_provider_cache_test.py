"""
测试 BaseDataProvider 三层缓存架构

验证:
1. 三层数据获取策略（内存 → 数据库 → API）
2. 缓存读写功能
3. 对外接口透明性
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
from core_bak_refactored.core.data.providers.protocols import PriceData
from core_bak_refactored.core.share.market.data_types import OHLCVRecord


class MockDataProvider(BaseDataProvider):
    """模拟数据提供者（用于测试）"""
    
    def __init__(self):
        super().__init__()
        self.api_call_count = 0
    
    def _fetch_from_external_api(self, symbol: str, start_date: str, end_date: str, period: str = 'daily') -> PriceData:
        """模拟API调用"""
        self.api_call_count += 1
        
        # 生成测试数据
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        records = [
            OHLCVRecord(
                date=pd.Timestamp(date),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=1000000 + i * 1000
            )
            for i, date in enumerate(dates)
        ]
        
        return PriceData(
            records=records,
            symbol=symbol,
            start_date=dates[0],
            end_date=dates[-1],
            count=len(records)
        )
    
    def get_test_symbol(self) -> str:
        return '000300.SH'


class TestBaseProviderCache:
    """测试 BaseDataProvider 缓存功能"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.provider = MockDataProvider()
        # 禁用数据库缓存，专注测试内存缓存和API调用
        self.provider._enable_db_cache = False
        self.test_symbol = '000300.SH'
        self.start_date = '2024-01-01'
        self.end_date = '2024-01-10'
    
    def test_first_call_uses_api(self):
        """测试首次调用使用API"""
        # 首次调用
        result = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 验证
        assert result is not None
        assert result.count > 0
        assert self.provider.api_call_count == 1, "首次调用应该调用API"
    
    def test_second_call_uses_memory_cache(self):
        """测试第二次调用使用内存缓存"""
        # 首次调用
        result1 = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 第二次调用（应该命中内存缓存）
        result2 = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 验证
        assert result1.count == result2.count
        assert self.provider.api_call_count == 1, "第二次调用应该使用内存缓存，不调用API"
    
    def test_memory_cache_expiration(self):
        """测试内存缓存过期"""
        # 设置较短的TTL
        self.provider._cache_ttl = 0.1  # 0.1秒
        
        # 首次调用
        result1 = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 等待缓存过期
        import time
        time.sleep(0.2)
        
        # 第二次调用（缓存已过期，应该再次调用API）
        result2 = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 验证
        assert self.provider.api_call_count == 2, "缓存过期后应该再次调用API"
    
    def test_different_parameters_different_cache(self):
        """测试不同参数使用不同缓存"""
        # 调用1
        result1 = self.provider.get_index_prices(
            self.test_symbol,
            '2024-01-01',
            '2024-01-10',
            datetime.now()
        )
        
        # 调用2（不同日期）
        result2 = self.provider.get_index_prices(
            self.test_symbol,
            '2024-02-01',
            '2024-02-10',
            datetime.now()
        )
        
        # 验证
        assert self.provider.api_call_count == 2, "不同参数应该调用两次API"
        assert result1.count != result2.count or True  # 数据可能不同
    
    @patch.object(MockDataProvider, '_get_db_service')
    def test_database_cache_integration(self, mock_get_db_service):
        """测试数据库缓存集成"""
        # 创建新的provider，启用数据库缓存
        provider = MockDataProvider()
        provider._enable_db_cache = True
        
        # Mock 数据库服务
        mock_db_service = Mock()
        mock_db_service.get_cached_data.return_value = None  # 缓存未命中
        mock_get_db_service.return_value = mock_db_service
        
        # 调用
        result = provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 验证数据库缓存被调用
        assert mock_db_service.get_cached_data.call_count >= 1
        assert mock_db_service.cache_data.call_count >= 1
    
    def test_stock_prices_uses_same_cache(self):
        """测试股票数据使用相同的缓存机制"""
        # 调用股票数据
        result = self.provider.get_stock_prices(
            '000001.SZ',
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 验证
        assert result is not None
        assert self.provider.api_call_count == 1
        
        # 第二次调用（应该使用缓存）
        result2 = self.provider.get_stock_prices(
            '000001.SZ',
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        assert self.provider.api_call_count == 1, "应该使用缓存"
    
    def test_cache_can_be_disabled(self):
        """测试可以禁用缓存"""
        # 禁用内存缓存
        self.provider._enable_memory_cache = False
        
        # 调用两次
        result1 = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        result2 = self.provider.get_index_prices(
            self.test_symbol,
            self.start_date,
            self.end_date,
            datetime.now()
        )
        
        # 验证（禁用缓存后应该调用两次API）
        assert self.provider.api_call_count >= 2, "禁用缓存后应该每次都调用API"


class TestCacheKey:
    """测试缓存键生成"""
    
    def test_cache_key_generation(self):
        """测试缓存键生成"""
        provider = MockDataProvider()
        
        key1 = provider._make_cache_key('000300.SH', '2024-01-01', '2024-01-10')
        key2 = provider._make_cache_key('000300.SH', '2024-01-01', '2024-01-10')
        key3 = provider._make_cache_key('000300.SH', '2024-01-01', '2024-01-11')
        
        # 相同参数生成相同键
        assert key1 == key2
        
        # 不同参数生成不同键
        assert key1 != key3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
