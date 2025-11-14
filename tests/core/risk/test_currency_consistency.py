import os
import sys
import pytest

# 确保可以导入重构后的核心模块
sys.path.insert(0, os.path.abspath('.'))

from core_bak_refactored.core.risk.risk_calculator import RiskCalculator


def make_rc(strict: bool = False, market_type: str = 'US', base_currency: str = 'USD') -> RiskCalculator:
    config = {
        'market_type': market_type,
        'market_configs': {
            market_type: {
                'base_currency': base_currency,
            }
        },
        'strict_currency_check': strict,
    }
    return RiskCalculator(config)


def test_currency_check_no_warnings():
    rc = make_rc(strict=False, market_type='US', base_currency='USD')
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': 'USD'},
                'MSFT': {'currency': 'USD'},
            }
        },
        'portfolio': {
            'base_currency': 'USD'
        }
    }
    warnings = rc._runtime_currency_check(data)
    assert warnings == []


def test_currency_check_missing_field_warning():
    rc = make_rc(strict=False)
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': 'USD'},
                'BABA': {},  # 缺少currency
            }
        }
    }
    warnings = rc._runtime_currency_check(data)
    assert any('缺少货币信息' in w for w in warnings)


def test_currency_check_multi_currency_warning():
    rc = make_rc(strict=False)
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': 'USD'},
                '0700.HK': {'currency': 'HKD'},
            }
        }
    }
    warnings = rc._runtime_currency_check(data)
    assert any('多币种检测' in w for w in warnings)


def test_portfolio_currency_mismatch_warning():
    rc = make_rc(strict=False, base_currency='USD')
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': 'USD'},
            }
        },
        'portfolio': {
            'base_currency': 'HKD'
        }
    }
    warnings = rc._runtime_currency_check(data)
    assert any('组合货币' in w and '基准货币' in w for w in warnings)


def test_strict_mode_raises_on_critical():
    rc = make_rc(strict=True)
    data = {
        'market_data': {
            'prices': {
                'BABA': {},  # 缺少currency，触发critical
            }
        }
    }
    warnings = rc._runtime_currency_check(data)
    with pytest.raises(ValueError):
        rc._handle_currency_warnings(warnings)
