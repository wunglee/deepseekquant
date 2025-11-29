from core_bak_refactored.core.data.analytics.volatility import calculate_daily_volatility
from datetime import datetime
from unittest.mock import Mock


def test_calculate_daily_volatility_returns_zero_for_insufficient_data():
    data = [Mock(close=100)]
    result = calculate_daily_volatility(data)
    assert result == 0.0


def test_calculate_daily_volatility_calculates_annualized():
    d1 = Mock(close=100)
    d2 = Mock(close=101)
    d3 = Mock(close=102)
    result = calculate_daily_volatility([d1, d2, d3])
    assert isinstance(result, float)
    assert result > 0
