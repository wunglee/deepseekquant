from core_bak_refactored.core.share import MarketData
from core_bak_refactored.core.data import DataSourceType, DataFrequency
from datetime import datetime


def test_models_exports():
    md = MarketData(symbol='AAPL', timestamp=datetime(2024,1,1), open=1, high=2, low=0.5, close=1.5, volume=100)
    assert md.symbol == 'AAPL'
    assert DataSourceType.YAHOO_FINANCE.value == 'yahoo'
    assert DataFrequency.DAILY.value == 'daily'
