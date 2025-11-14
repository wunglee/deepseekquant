import os
import sys
import pytest

# 允许导入重构后的基础设施模块
sys.path.insert(0, os.path.abspath('.'))

from core_bak_refactored.infrastructure.currency_converter import CurrencyConverter


def make_converter() -> CurrencyConverter:
    return CurrencyConverter({
        'exchange_rate_sources': {
            'USD': {'CNY': 7.1, 'HKD': 7.8},
            'HKD': {'USD': 0.128205},
        },
        'fallback_exchange_rates': {
            'CNY': {'USD': 0.140845},  # 约等于1/7.1
            'USD->EUR': 0.92,
        }
    })


def test_convert_portfolio_currency_to_usd():
    converter = make_converter()
    portfolio = {
        'allocations': {
            'AAPL': {'currency': 'USD', 'value': 1000.0},
            '0700.HK': {'currency': 'HKD', 'value': 7800.0},  # 约等于1000 USD
        }
    }
    result = converter.convert_portfolio_currency(portfolio, target_currency='USD')
    assert result['target_currency'] == 'USD'
    # 1000 USD + 7800 HKD * 0.128205 ≈ 1000 + 1000 = 2000 USD
    assert abs(result['total_converted_value'] - 2000.0) < 1e-3
    assert 'AAPL' in result['details'] and '0700.HK' in result['details']


def test_calculate_currency_exposure():
    converter = make_converter()
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


def test_rate_fallback_and_identity():
    converter = make_converter()
    # 未声明的汇率，返回1.0（MVP策略）
    assert converter._get_rate('JPY', 'USD') == 1.0
    # 同币种转换为1.0
    assert converter._get_rate('USD', 'USD') == 1.0


def test_flat_key_fallback():
    converter = make_converter()
    # 使用扁平键回退汇率
    rate = converter._get_rate('USD', 'EUR')
    assert abs(rate - 0.92) < 1e-9
