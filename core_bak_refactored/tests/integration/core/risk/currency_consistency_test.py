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


def test_strict_mode_not_raise_on_missing_currency():
    rc = make_rc(strict=True)
    data = {
        'market_data': {
            'prices': {
                'BABA': {},  # 缺少currency，严格模式下不抛错（按新分级策略）
            }
        }
    }
    warnings = rc._runtime_currency_check(data)
    # 不应抛异常
    rc._handle_currency_warnings(warnings)


def test_default_strict_mode_by_market():
    # US默认严格
    rc_us = RiskCalculator({
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}}
    })
    assert rc_us.strict_currency_check is True
    # CN默认宽松
    rc_cn = RiskCalculator({
        'market_type': 'CN',
        'market_configs': {'CN': {'base_currency': 'CNY'}}
    })
    assert rc_cn.strict_currency_check is False


def test_risk_parameter_currency_check():
    rc = make_rc(strict=False, market_type='US', base_currency='USD')
    data = {
        'market_data': {
            'risk_free_rate_info': {'currency': 'HKD'},
            'market_returns_info': {'currency': 'USD'},
        }
    }
    warnings = rc._check_risk_parameters_currency(data)
    assert any('无风险利率货币' in w for w in warnings)


def test_classification_error_in_strict_mode():
    rc = make_rc(strict=True, market_type='US', base_currency='USD')
    # 人工构造组合不一致错误
    warnings = ["组合货币HKD≠基准货币USD"]
    with pytest.raises(ValueError):
        rc._handle_currency_warnings(warnings)


def test_assess_data_source_quality():
    rc = make_rc(strict=False)
    prices = {
        'AAPL': {'currency': 'USD'},
        'MSFT': {'currency': 'USD'},
        '0700.HK': {},
    }
    quality = rc._assess_data_source_quality(prices)
    assert 'quality_rating' in quality


# ========== P1.5 边界条件测试补充 ==========

def test_empty_market_data():
    """边界测试：空市场数据"""
    rc = make_rc(strict=False)
    data = {'market_data': {'prices': {}}}
    warnings = rc._runtime_currency_check(data)
    assert warnings == []


def test_none_currency_field():
    """边界测试：currency字段为None"""
    rc = make_rc(strict=False)
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'currency': None},
            }
        }
    }
    warnings = rc._runtime_currency_check(data)
    assert any('缺少货币信息' in w for w in warnings)


def test_extremely_low_quality_data():
    """边界测试：极低质量数据（90%缺失货币）"""
    rc = make_rc(strict=False)
    prices = {f'STOCK_{i}': {} for i in range(90)}
    prices.update({f'GOOD_{i}': {'currency': 'USD'} for i in range(10)})
    quality = rc._assess_data_source_quality(prices)
    assert quality['quality_rating'] == 'D'
    assert quality['currency_coverage'] < 0.15


def test_strict_mode_with_high_quality_data():
    """边界测试：严格模式+高质量数据不应抛错"""
    rc = make_rc(strict=True, market_type='US', base_currency='USD')
    data = {
        'market_data': {
            'prices': {f'STOCK_{i}': {'currency': 'USD'} for i in range(100)}
        },
        'portfolio': {'base_currency': 'USD'}
    }
    warnings = rc._runtime_currency_check(data)
    rc._handle_currency_warnings(warnings)  # 不应抛错


def test_extreme_currency_diversity():
    """边界测试：极端货币多样性（10种不同货币）"""
    rc = make_rc(strict=False)
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'HKD', 'SGD', 'AUD', 'CAD', 'CHF']
    data = {
        'market_data': {
            'prices': {f'STOCK_{i}': {'currency': cur} for i, cur in enumerate(currencies)}
        }
    }
    warnings = rc._runtime_currency_check(data)
    # 检测到多币种即可，不强制要求具体数量
    assert any('多币种检测' in w for w in warnings)


def test_risk_parameters_all_missing_currency():
    """边界测试：所有风险参数缺失货币信息"""
    rc = make_rc(strict=False)
    data = {
        'market_data': {
            'risk_free_rate_info': {},
            'market_returns_info': {},
        }
    }
    warnings = rc._check_risk_parameters_currency(data)
    # 缺失时不应报错（兼容历史数据）
    assert warnings == []


def test_zero_coverage_data_quality():
    """边界测试：0%货币覆盖率"""
    rc = make_rc(strict=False)
    prices = {f'STOCK_{i}': {} for i in range(100)}
    quality = rc._assess_data_source_quality(prices)
    assert quality['quality_rating'] == 'D'
    assert quality['currency_coverage'] == 0.0


def test_perfect_coverage_data_quality():
    """边界测试：100%货币覆盖率"""
    rc = make_rc(strict=False)
    prices = {f'STOCK_{i}': {'currency': 'USD'} for i in range(100)}
    quality = rc._assess_data_source_quality(prices)
    assert quality['quality_rating'] == 'A'
    assert quality['currency_coverage'] == 1.0


def test_large_scale_portfolio():
    """边界测试：大规模组合（1000个标的）性能"""
    import time
    rc = make_rc(strict=False)
    data = {
        'market_data': {
            'prices': {f'STOCK_{i}': {'currency': 'USD' if i % 10 != 0 else 'HKD'} for i in range(1000)}
        }
    }
    start = time.time()
    warnings = rc._runtime_currency_check(data)
    elapsed = time.time() - start
    # 应该在100ms内完成
    assert elapsed < 0.1
    assert any('多币种检测' in w for w in warnings)
