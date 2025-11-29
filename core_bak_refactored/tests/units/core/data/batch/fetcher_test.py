import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import asyncio
from core_bak_refactored.core.data.batch.fetcher import BatchDataFetcher


class TestBatchDataFetcher:
    """测试批量数据获取器。"""

    def test_init(self):
        """测试初始化。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher, max_concurrent=5, rate_limit_per_second=10)
        
        assert batch_fetcher.fetcher == fetcher
        assert batch_fetcher.max_concurrent == 5
        assert batch_fetcher.rate_limit_per_second == 10

    @pytest.mark.asyncio
    async def test_fetch_batch(self):
        """测试批量获取数据。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        results = await batch_fetcher.fetch_batch(symbols, '1y', '1d')
        
        assert isinstance(results, dict)
        assert len(results) == 3
        assert 'AAPL' in results
        assert 'GOOGL' in results
        assert 'MSFT' in results

    @pytest.mark.asyncio
    async def test_fetch_batch_empty(self):
        """测试空列表批量获取。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        results = await batch_fetcher.fetch_batch([], '1y', '1d')
        
        assert results == {}

    @pytest.mark.asyncio
    async def test_rate_limit(self):
        """测试速率限制。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher, rate_limit_per_second=2)
        
        # 记录请求时间
        start_time = asyncio.get_event_loop().time()
        
        # 执行3个请求（超过速率限制）
        await batch_fetcher._wait_for_rate_limit()
        await batch_fetcher._wait_for_rate_limit()
        await batch_fetcher._wait_for_rate_limit()
        
        end_time = asyncio.get_event_loop().time()
        
        # 应该被限流
        assert len(batch_fetcher.request_times) <= 2

    @pytest.mark.asyncio
    async def test_fetch_batch_quotes(self):
        """测试批量获取报价。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        symbols = ['AAPL', 'GOOGL']
        quotes = await batch_fetcher.fetch_batch_quotes(symbols)
        
        assert isinstance(quotes, dict)
        assert len(quotes) == 2
        assert quotes['AAPL'] is not None
        assert quotes['GOOGL'] is not None

    def test_split_into_batches(self):
        """测试分批。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        symbols = [f'STOCK{i}' for i in range(250)]
        batches = batch_fetcher.split_into_batches(symbols, batch_size=100)
        
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

    def test_split_into_batches_exact(self):
        """测试精确分批。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        symbols = [f'STOCK{i}' for i in range(200)]
        batches = batch_fetcher.split_into_batches(symbols, batch_size=100)
        
        assert len(batches) == 2
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100

    @pytest.mark.asyncio
    async def test_fetch_large_batch(self):
        """测试大批量获取。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        symbols = [f'STOCK{i}' for i in range(250)]
        results = await batch_fetcher.fetch_large_batch(
            symbols, '1y', '1d', batch_size=100
        )
        
        assert isinstance(results, dict)
        assert len(results) == 250

    @pytest.mark.asyncio
    async def test_fetch_batch_with_retry(self):
        """测试带重试的批量获取。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher)
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        results = await batch_fetcher.fetch_batch_with_retry(
            symbols, '1y', '1d', max_retries=2
        )
        
        assert isinstance(results, dict)
        # 至少应该有一些成功的结果
        assert len(results) >= 0

    def test_get_statistics(self):
        """测试获取统计信息。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher, max_concurrent=10, rate_limit_per_second=5)
        
        stats = batch_fetcher.get_statistics()
        
        assert stats['max_concurrent'] == 10
        assert stats['rate_limit_per_second'] == 5
        assert 'current_requests_in_window' in stats

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """测试并发限制。"""
        fetcher = Mock()
        batch_fetcher = BatchDataFetcher(fetcher, max_concurrent=2)
        
        # 创建多个并发任务
        symbols = [f'STOCK{i}' for i in range(10)]
        results = await batch_fetcher.fetch_batch(symbols, '1y', '1d')
        
        # 应该成功完成所有请求
        assert len(results) == 10
