import pytest
from datetime import datetime
from typing import List

from core_bak_refactored.app.data.data_service import DataService
from core_bak_refactored.core.data.data_fetcher import MarketData
from core_bak_refactored.core.share.market_enums import MarketCode


@pytest.mark.asyncio
async def test_data_service_historical_with_custom_source():
    """应用层门面通过自定义数据源获取历史数据（委派到领域层）"""

    async def mock_fetch(symbol: str, period: str, interval: str, data_type: str, adjustments: bool) -> List[MarketData]:
        # 返回包含枚举型市场类型的 MarketData
        return [
            MarketData(
                symbol=symbol,
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1_000_000,
                metadata={
                    'market_type': MarketCode.CN,  # 测试中直接使用枚举
                    'data_source': 'mock'
                }
            )
        ]

    custom_sources = {'test_mock': mock_fetch}
    config = {
        'cache_enabled': False,
        'primary': 'test_mock'
    }

    svc = DataService(config=config, custom_sources=custom_sources)
    result = await svc.get_historical_data(symbols=['AAPL'], period='1mo', interval='1d', data_type='ohlcv')

    assert 'AAPL' in result
    assert isinstance(result['AAPL'], list)
    assert result['AAPL'][0].symbol == 'AAPL'
    assert result['AAPL'][0].metadata.get('market_type') == MarketCode.CN  # 断言枚举

    await svc.cleanup()


@pytest.mark.asyncio
async def test_data_service_fundamental_via_stub(monkeypatch):
    """应用层门面通过stub获取基本面数据，验证委派与返回结构"""

    svc = DataService(config={'cache_enabled': False})

    async def stub_fundamental(symbol: str):
        return {
            'company_name': 'Apple Inc.',
            'market': str(MarketCode.US),  # 字段值规范化为字符串
            'last_updated': datetime(2024, 1, 1).isoformat()
        }

    # 替换领域层方法为stub
    monkeypatch.setattr(svc._fetcher, 'get_fundamental_data', stub_fundamental)

    fundamentals = await svc.get_fundamental_data('AAPL')
    assert fundamentals['company_name'] == 'Apple Inc.'
    assert fundamentals['market'] == 'US'

    await svc.cleanup()
