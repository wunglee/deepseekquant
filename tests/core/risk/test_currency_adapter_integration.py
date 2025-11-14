import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('.'))

from core_bak_refactored.core.risk.risk_calculator import RiskCalculator
from core_bak_refactored.core.share.exchange_rates import MockExchangeRateAdapter


def make_rc(market_type='US', base_currency='USD') -> RiskCalculator:
    config = {
        'market_type': market_type,
        'market_configs': {
            market_type: {
                'base_currency': base_currency,
            }
        },
    }
    return RiskCalculator(config)


def test_unify_currency_summary_without_adapter():
    rc = make_rc()
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': 'USD', 'close': [100.0]},
                '0700.HK': {'currency': 'HKD', 'close': [780.0]},
            }
        }
    }
    # 无适配器时不执行统一转换，返回None
    assert rc._unify_currency_for_portfolio(data) is None


def test_unify_currency_summary_with_adapter():
    rc = make_rc()
    rc.attach_exchange_rate_adapter(MockExchangeRateAdapter())
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': 'USD', 'close': [1000.0]},
                '0700.HK': {'currency': 'HKD', 'close': [7800.0]},  # ≈ 1000 USD
            }
        },
        'portfolio': {
            'base_currency': 'USD'
        }
    }
    summary = rc._unify_currency_for_portfolio(data)
    assert summary is not None
    assert summary['target_currency'] == 'USD'
    assert abs(summary['total_converted_value'] - 2000.0) < 1e-3
