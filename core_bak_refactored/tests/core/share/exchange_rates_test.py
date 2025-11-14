import os
import sys
import pytest

# 允许导入重构后的共享业务模块
sys.path.insert(0, os.path.abspath('.'))

from core_bak_refactored.core.share.exchange_rates import CurrencyConverter, MockExchangeRateAdapter


def test_convert_portfolio_currency_to_usd():
    adapter = MockExchangeRateAdapter()
    rates = adapter.get_rates('US')
    converter = CurrencyConverter()
    portfolio = {
        'allocations': {
            'AAPL': {'currency': 'USD', 'value': 1000.0},
            '0700.HK': {'currency': 'HKD', 'value': 7800.0},  # ≈ 1000 USD
        }
    }
    result = converter.convert_portfolio_currency(portfolio, target_currency='USD', rates=rates)
    assert result['target_currency'] == 'USD'
    assert abs(result['total_converted_value'] - 2000.0) < 1e-3


def test_calculate_currency_exposure():
    converter = CurrencyConverter()
    portfolio = {
        'allocations': {
            'AAPL': {'currency': 'USD', 'value': 1000.0},
            'BABA': {'currency': 'USD', 'value': 500.0},
            '0700.HK': {'currency': 'HKD', 'value': 1000.0},
        }
    }
    exposure = converter.calculate_currency_exposure(portfolio)
    assert exposure['USD'] == 1500.0
    assert exposure['HKD'] == 1000.0


def test_rate_lookup_nested_and_flat():
    converter = CurrencyConverter()
    rates = {
        'USD': {'CNY': 7.1},
        'USD->EUR': 0.92,
    }
    assert abs(converter._get_rate('USD', 'CNY', rates) - 7.1) < 1e-9
    assert abs(converter._get_rate('USD', 'EUR', rates) - 0.92) < 1e-9
    assert converter._get_rate('JPY', 'USD', rates) == 1.0
