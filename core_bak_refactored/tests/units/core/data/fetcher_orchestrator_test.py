"""
测试DataFetcherOrchestrator - 验证与DataFetcher完全等效

关键等效性验证：
1. 缓存统计：cache_hits/cache_misses 同时更新 cache_stats 和 performance_metrics
2. 备用数据源日志：info/warning/debug 完整记录
3. _fetch_symbol_data 异常处理：外层try-except + 内层fallback
4. _try_fallback_sources 日志跟踪：每步骤详细记录
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from core_bak_refactored.core.data.fetcher_orchestrator import DataFetcherOrchestrator
from core_bak_refactored.core.share import MarketData
from core_bak_refactored.core.share.market.market_enums import MarketCode


@pytest.mark.asyncio
async def test_fetcher_orchestrator_delegates_with_custom_source():
    """测试基本委派功能"""
    async def mock_fetch(symbol, period, interval, data_type, adjustments):
        return [MarketData(symbol=symbol, timestamp=datetime(2024,1,1), open=1, high=2, low=0.5, close=1.5, volume=10,
                           metadata={'market_type': MarketCode.US.value})]

    config = {'primary': 'yahoo', 'fallback_sources': [], 'cache_enabled': False}
    orch = DataFetcherOrchestrator(config, custom_sources={'yahoo': mock_fetch})
    res = await orch.get_historical_data(['AAPL'], '1y', '1d', 'ohlcv', True)
    assert 'AAPL' in res and len(res['AAPL']) == 1


@pytest.mark.asyncio
async def test_cache_statistics_equivalence():
    """
    等效性验证：缓存统计必须同时更新 cache_hits 和 cache_misses
    对比DataFetcher的实现（L329-335）
    """
    async def mock_fetch(symbol, period, interval, data_type, adjustments):
        return [MarketData(symbol=symbol, timestamp=datetime(2024,1,1), open=150.0, high=151.0,
                           low=149.0, close=150.5, volume=2000000)]
    
    config = {
        'primary': 'custom',
        'cache_enabled': True,
        'cache_ttl': 300
    }
    orchestrator = DataFetcherOrchestrator(config, custom_sources={'custom': mock_fetch})
    
    # 第一次调用：缓存未命中
    result1 = await orchestrator.get_historical_data(['AAPL'], '1mo', '1d', 'ohlcv', True)
    assert orchestrator.performance_metrics.get('cache_hits', 0) == 0
    assert orchestrator.performance_metrics.get('cache_misses', 0) == 1  # 必须统计cache_misses
    
    # 第二次调用：缓存命中
    result2 = await orchestrator.get_historical_data(['AAPL'], '1mo', '1d', 'ohlcv', True)
    assert orchestrator.performance_metrics.get('cache_hits', 0) == 1  # 必须统计cache_hits
    assert orchestrator.performance_metrics.get('cache_misses', 0) == 1
    
    # 验证get_data_quality_metrics也使用performance_metrics
    metrics = orchestrator.get_data_quality_metrics()
    assert metrics['cache_hits'] == 1
    assert metrics['cache_misses'] == 1


@pytest.mark.asyncio
async def test_fallback_source_logging():
    """
    等效性验证：备用数据源必须记录详细日志
    对比DataFetcher._fetch_symbol_data内部fallback（L398-409）
    """
    async def primary_fail(symbol, period, interval, data_type, adjustments):
        raise Exception("Primary failed")
    
    async def fallback_success(symbol, period, interval, data_type, adjustments):
        return [MarketData(symbol=symbol, timestamp=datetime(2024,1,1), open=300.0, high=301.0,
                           low=299.0, close=300.5, volume=3000000)]
    
    config = {
        'primary': 'primary_source',
        'fallback_sources': ['fallback1'],
        'cache_enabled': False
    }
    
    orchestrator = DataFetcherOrchestrator(
        config,
        custom_sources={
            'primary_source': primary_fail,
            'fallback1': fallback_success
        }
    )
    
    with patch.object(orchestrator.logger, 'info') as mock_info, \
         patch.object(orchestrator.logger, 'warning') as mock_warning:
        result = await orchestrator.get_historical_data(['MSFT'], '1mo', '1d', 'ohlcv', True)
        
        # 验证：
        # 1. _fetch_symbol_data内部的fallback会记录 logger.info("备用数据源 {fallback_source} 成功获取 {symbol} 数据")
        # 2. 如果主数据源失败，fallback成功，不会进入_try_fallback_sources
        assert any('成功获取' in str(call) for call in mock_info.call_args_list)
        assert 'MSFT' in result  # 必须成功获取数据


@pytest.mark.asyncio
async def test_fetch_symbol_data_exception_handling():
    """
    等效性验证：_fetch_symbol_data 必须有多层try-except
    对比DataFetcher._fetch_symbol_data（L387-415）
    """
    async def primary_error(symbol, period, interval, data_type, adjustments):
        raise RuntimeError("主数据源异常")
    
    async def fallback_error(symbol, period, interval, data_type, adjustments):
        raise ValueError("备用数据源异常")
    
    config = {
        'primary': 'primary',
        'fallback_sources': ['fallback'],
        'cache_enabled': False
    }
    
    orchestrator = DataFetcherOrchestrator(
        config,
        custom_sources={
            'primary': primary_error,
            'fallback': fallback_error
        }
    )
    
    with patch.object(orchestrator.logger, 'debug') as mock_debug, \
         patch.object(orchestrator.logger, 'warning') as mock_warning:
        
        result = await orchestrator._fetch_symbol_data('ERROR', '1mo', '1d', 'ohlcv', True)
        
        # 验证：
        # 1. 主数据源异常被内层try-except捕获，记录debug日志
        # 2. 备用数据源异常被捕获，记录warning日志
        # 3. 最终返回None，不抛出异常
        assert result is None  # 必须返回None，不能抛出异常
        assert any('主数据源' in str(call) and '失败' in str(call) for call in mock_debug.call_args_list)
        assert any('备用数据源' in str(call) and '失败' in str(call) for call in mock_warning.call_args_list)


@pytest.mark.asyncio
async def test_performance_metrics_update():
    """
    等效性验证：性能指标必须包括 requests_total, requests_failed, avg_response_time
    对比DataFetcher.get_historical_data（L371-377）
    """
    async def success_fetch(symbol, period, interval, data_type, adjustments):
        return [MarketData(symbol=symbol, timestamp=datetime(2024,1,1), open=2800.0, high=2801.0,
                           low=2799.0, close=2800.5, volume=1500000)]
    
    config = {
        'primary': 'source',
        'cache_enabled': False
    }
    
    orchestrator = DataFetcherOrchestrator(config, custom_sources={'source': success_fetch})
    
    # 调用前的性能指标
    initial_total = orchestrator.performance_metrics['requests_total']
    
    # 执行请求
    result = await orchestrator.get_historical_data(['GOOGL'], '1mo', '1d', 'ohlcv', True)
    
    # 验证指标更新
    assert orchestrator.performance_metrics['requests_total'] == initial_total + 1  # 总请求数
    assert orchestrator.performance_metrics['avg_response_time'] > 0  # 平均响应时间
    assert 'last_update' in orchestrator.performance_metrics  # 最后更新时间
    assert orchestrator.performance_metrics['data_points_processed'] >= 0  # 处理的数据点
