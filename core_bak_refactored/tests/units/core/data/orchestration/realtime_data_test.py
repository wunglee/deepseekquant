from unittest.mock import Mock

import pytest

from core_bak_refactored.core.data.orchestration.realtime_data import (
    RealtimeDataOrchestrator
)


class TestRealtimeDataOrchestrator:
    """测试实时数据编排器。"""

    def test_init(self):
        """测试初始化。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        assert orchestrator.fetcher == fetcher
        assert orchestrator.subscriptions == {}
        assert orchestrator.active_connections == {}
        assert orchestrator.streaming is False

    @pytest.mark.asyncio
    async def test_subscribe(self):
        """测试订阅数据流。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        callback = Mock()
        result = await orchestrator.subscribe(['AAPL', 'GOOGL'], 'quote', callback)
        
        assert result is True
        assert len(orchestrator.subscriptions) == 1
        
        # 检查订阅键
        subscription_key = 'quote:AAPL,GOOGL'
        assert subscription_key in orchestrator.subscriptions
        assert callback in orchestrator.subscriptions[subscription_key]

    @pytest.mark.asyncio
    async def test_subscribe_multiple_callbacks(self):
        """测试多个回调订阅同一数据流。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        callback1 = Mock()
        callback2 = Mock()
        
        await orchestrator.subscribe(['AAPL'], 'quote', callback1)
        await orchestrator.subscribe(['AAPL'], 'quote', callback2)
        
        subscription_key = 'quote:AAPL'
        assert len(orchestrator.subscriptions[subscription_key]) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """测试取消订阅。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        callback = Mock()
        await orchestrator.subscribe(['AAPL'], 'quote', callback)
        
        result = await orchestrator.unsubscribe(['AAPL'], 'quote', callback)
        
        assert result is True
        assert len(orchestrator.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_partial(self):
        """测试部分取消订阅（还有其他回调）。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        callback1 = Mock()
        callback2 = Mock()
        
        await orchestrator.subscribe(['AAPL'], 'quote', callback1)
        await orchestrator.subscribe(['AAPL'], 'quote', callback2)
        
        await orchestrator.unsubscribe(['AAPL'], 'quote', callback1)
        
        subscription_key = 'quote:AAPL'
        assert subscription_key in orchestrator.subscriptions
        assert callback2 in orchestrator.subscriptions[subscription_key]
        assert callback1 not in orchestrator.subscriptions[subscription_key]

    @pytest.mark.asyncio
    async def test_notify_subscribers_sync(self):
        """测试通知同步回调。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        callback = Mock()
        await orchestrator.subscribe(['AAPL'], 'quote', callback)
        
        data = {'symbol': 'AAPL', 'price': 150.0}
        subscription_key = 'quote:AAPL'
        
        await orchestrator._notify_subscribers(subscription_key, data)
        
        callback.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_notify_subscribers_async(self):
        """测试通知异步回调。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        async def async_callback(data):
            pass
        
        callback = Mock(wraps=async_callback)
        await orchestrator.subscribe(['AAPL'], 'quote', callback)
        
        data = {'symbol': 'AAPL', 'price': 150.0}
        subscription_key = 'quote:AAPL'
        
        await orchestrator._notify_subscribers(subscription_key, data)

    @pytest.mark.asyncio
    async def test_fetch_latest_data(self):
        """测试获取最新数据。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        result = await orchestrator._fetch_latest_data(['AAPL', 'GOOGL'], 'quote')
        
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止。"""
        fetcher = Mock()
        orchestrator = RealtimeDataOrchestrator(fetcher)
        
        await orchestrator.start()
        assert orchestrator.streaming is True
        
        await orchestrator.stop()
        assert orchestrator.streaming is False
        assert len(orchestrator.subscriptions) == 0
        assert len(orchestrator.active_connections) == 0
