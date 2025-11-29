import pytest
from unittest.mock import Mock, AsyncMock
import asyncio
from core_bak_refactored.core.data.fallback.orchestrator import try_fallback_sources


class TestTryFallbackSources:
    """测试备用数据源编排功能（扩展版）。"""

    @pytest.mark.asyncio
    async def test_try_fallback_sources_all_success_first_source(self):
        """测试第一个备用源即全部成功的情况。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1', 'source2']
        
        # 模拟数据源函数（全部成功）
        async def mock_source1(symbol, *args):
            return [{'symbol': symbol, 'data': 'value'}]
        
        mock_fetcher.data_sources = {'source1': mock_source1}
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True
        )
        
        # 验证两个符号都成功获取
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert result['AAPL'][0]['symbol'] == 'AAPL'
        assert result['MSFT'][0]['symbol'] == 'MSFT'

    @pytest.mark.asyncio
    async def test_try_fallback_sources_partial_success(self):
        """测试部分成功，需要多个备用源的情况。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1', 'source2']
        
        # 第一个源只成功一个符号
        async def mock_source1(symbol, *args):
            if symbol == 'AAPL':
                return [{'symbol': symbol, 'source': 'source1'}]
            raise Exception("Data not available")
        
        # 第二个源成功剩余符号
        async def mock_source2(symbol, *args):
            if symbol == 'MSFT':
                return [{'symbol': symbol, 'source': 'source2'}]
            return None
        
        mock_fetcher.data_sources = {
            'source1': mock_source1,
            'source2': mock_source2
        }
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True
        )
        
        # 验证两个符号都获取成功，但来自不同源
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert result['AAPL'][0]['source'] == 'source1'
        assert result['MSFT'][0]['source'] == 'source2'

    @pytest.mark.asyncio
    async def test_try_fallback_sources_all_fail(self):
        """测试所有备用源都失败的情况。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1', 'source2']
        
        # 所有源都失败
        async def mock_failing_source(symbol, *args):
            raise Exception(f"Failed to fetch {symbol}")
        
        mock_fetcher.data_sources = {
            'source1': mock_failing_source,
            'source2': mock_failing_source
        }
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True
        )
        
        # 应该返回空字典
        assert result == {}

    @pytest.mark.asyncio
    async def test_try_fallback_sources_empty_symbols(self):
        """测试空符号列表的情况。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1']
        mock_fetcher.data_sources = {}
        
        result = await try_fallback_sources(
            mock_fetcher, [], '1y', '1d', 'ohlcv', True
        )
        
        # 应该立即返回空字典
        assert result == {}

    @pytest.mark.asyncio
    async def test_try_fallback_sources_unregistered_source(self):
        """测试未注册的数据源（跳过）。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['unregistered_source', 'source2']
        
        async def mock_source2(symbol, *args):
            return [{'symbol': symbol, 'data': 'value'}]
        
        mock_fetcher.data_sources = {'source2': mock_source2}
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL'], '1y', '1d', 'ohlcv', True
        )
        
        # 应该跳过未注册源，使用source2成功获取
        assert 'AAPL' in result

    @pytest.mark.asyncio
    async def test_try_fallback_sources_concurrent_execution(self):
        """测试并发执行多个符号。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1']
        
        execution_times = []
        
        async def mock_source1(symbol, *args):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # 模拟耗时操作
            execution_times.append((symbol, asyncio.get_event_loop().time() - start))
            return [{'symbol': symbol}]
        
        mock_fetcher.data_sources = {'source1': mock_source1}
        
        start_time = asyncio.get_event_loop().time()
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL', 'MSFT', 'GOOGL'], '1y', '1d', 'ohlcv', True
        )
        total_time = asyncio.get_event_loop().time() - start_time
        
        # 验证并发执行（总时间应接近单个任务时间，而非累加）
        assert total_time < 0.2  # 3个任务并发，应在0.2秒内完成
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_try_fallback_sources_early_exit(self):
        """测试所有符号成功后提前退出。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1', 'source2']
        
        source2_called = False
        
        async def mock_source1(symbol, *args):
            return [{'symbol': symbol, 'source': 'source1'}]
        
        async def mock_source2(symbol, *args):
            nonlocal source2_called
            source2_called = True
            return [{'symbol': symbol, 'source': 'source2'}]
        
        mock_fetcher.data_sources = {
            'source1': mock_source1,
            'source2': mock_source2
        }
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True
        )
        
        # 验证source1成功后不会调用source2
        assert not source2_called
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_try_fallback_sources_source_exception_handling(self):
        """测试数据源整体异常的处理。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1', 'source2']
        
        # source1整体抛出异常（非单个符号失败）
        def mock_source1_error(*args):
            raise RuntimeError("Source1 is down")
        
        async def mock_source2(symbol, *args):
            return [{'symbol': symbol, 'source': 'source2'}]
        
        mock_fetcher.data_sources = {
            'source1': mock_source1_error,
            'source2': mock_source2
        }
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL'], '1y', '1d', 'ohlcv', True
        )
        
        # 应该继续尝试source2并成功
        assert 'AAPL' in result
        assert result['AAPL'][0]['source'] == 'source2'

    @pytest.mark.asyncio
    async def test_try_fallback_sources_none_result(self):
        """测试数据源返回None的情况。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1']
        
        async def mock_source1(symbol, *args):
            return None  # 数据源返回None
        
        mock_fetcher.data_sources = {'source1': mock_source1}
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL'], '1y', '1d', 'ohlcv', True
        )
        
        # 应该返回空字典
        assert result == {}

    @pytest.mark.asyncio
    async def test_try_fallback_sources_progressive_reduction(self):
        """测试失败列表的逐步减少。"""
        mock_fetcher = Mock()
        mock_fetcher.fallback_sources = ['source1', 'source2', 'source3']
        
        # source1成功1个
        async def mock_source1(symbol, *args):
            if symbol == 'AAPL':
                return [{'symbol': symbol}]
            raise Exception("Failed")
        
        # source2成功1个
        async def mock_source2(symbol, *args):
            if symbol == 'MSFT':
                return [{'symbol': symbol}]
            raise Exception("Failed")
        
        # source3成功剩余的
        async def mock_source3(symbol, *args):
            if symbol == 'GOOGL':
                return [{'symbol': symbol}]
            raise Exception("Failed")
        
        mock_fetcher.data_sources = {
            'source1': mock_source1,
            'source2': mock_source2,
            'source3': mock_source3
        }
        
        result = await try_fallback_sources(
            mock_fetcher, ['AAPL', 'MSFT', 'GOOGL'], '1y', '1d', 'ohlcv', True
        )
        
        # 验证所有符号都成功获取
        assert len(result) == 3
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert 'GOOGL' in result
