import os
import sys


# 允许导入重构后的共享业务模块
sys.path.insert(0, os.path.abspath('.'))

from core_bak_refactored.core.share.exchange_rates import CurrencyConverter, MockExchangeRateAdapter
from core_bak_refactored.core.share.market.market_config import MarketConfig


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


def test_market_thresholds_config_values():
    mcm = MarketConfig()
    cfg_cn = mcm.generate_config_template('CN')
    cn = cfg_cn['market_configs']['CN']
    assert cn['volatility_spike_threshold'] == 0.05
    assert cn['limit_hit_ratio_threshold'] == 0.25

    cfg_us = mcm.generate_config_template('US')
    us = cfg_us['market_configs']['US']
    assert us['volatility_spike_threshold'] == 0.03
    assert us['limit_hit_ratio_threshold'] == 0.20

    cfg_eu = mcm.generate_config_template('EU')
    eu = cfg_eu['market_configs']['EU']
    assert eu['volatility_spike_threshold'] == 0.035
    assert eu['limit_hit_ratio_threshold'] == 0.20


def test_event_weights_and_sensitivity_values():
    mcm = MarketConfig()
    cfg_us = mcm.generate_config_template('US')
    us = cfg_us['market_configs']['US']
    assert us['major_event_sensitivity'] == 'HIGH'
    weights_us = us.get('event_weights', {})
    assert weights_us.get('circuit_breaker') == 0.4
    assert weights_us.get('extreme_correlation') == 0.3
    assert weights_us.get('limit_hits') == 0.2
    assert weights_us.get('major_event') == 0.5

    cfg_cn = mcm.generate_config_template('CN')
    cn = cfg_cn['market_configs']['CN']
    assert cn['major_event_sensitivity'] == 'MEDIUM'
    weights_cn = cn.get('event_weights', {})
    assert weights_cn.get('circuit_breaker') == 0.6
    assert weights_cn.get('extreme_correlation') == 0.2
    assert weights_cn.get('limit_hits') == 0.4
    assert weights_cn.get('major_event') == 0.3


def test_volatility_tiers_cache_ttl_values():
    mcm = MarketConfig()
    cfg_us = mcm.generate_config_template('US')
    tiers = cfg_us['market_configs']['US']['volatility_tiers']
    assert tiers['NORMAL']['cache_ttl'] == 3600
    assert tiers['MEDIUM']['cache_ttl'] == 1800
    assert tiers['HIGH']['cache_ttl'] == 600
    assert tiers['EXTREME']['cache_ttl'] == 60


def test_trading_days_per_year_by_market():
    mcm = MarketConfig()
    expected = {'CN': 245, 'US': 252, 'HK': 247, 'JP': 245, 'EU': 255, 'SG': 250}
    for mt, days in expected.items():
        cfg = mcm.generate_config_template(mt)
        assert cfg['trading_days_per_year'] == days


def test_validate_market_config_errors():
    mcm = MarketConfig()
    errors = mcm.validate_market_config({'market_type': 'ZZ', 'market_configs': {}})
    assert any('不支持的市场类型' in e for e in errors)
    errors2 = mcm.validate_market_config({'market_type': 'US', 'market_configs': {}})
    assert any('缺少US市场的具体配置' in e for e in errors2)


def test_cn_limit_thresholds_and_flags():
    mcm = MarketConfig()
    cn = mcm.generate_config_template('CN')['market_configs']['CN']
    assert cn['has_limit_up_down'] is True
    lt = cn['limit_thresholds']
    assert lt['main_board'] == 0.10
    assert lt['gem'] == 0.20
    assert lt['st'] == 0.05
    assert lt['kcb'] == 0.20


def test_us_luld_and_circuit_breaker_levels():
    mcm = MarketConfig()
    us = mcm.generate_config_template('US')['market_configs']['US']
    assert us['has_limit_up_down'] is False
    assert us['luld_threshold'] == 0.05
    assert us['circuit_breaker_levels'] == [0.07, 0.13, 0.20]


def test_major_event_sensitivity_per_market():
    mcm = MarketConfig()
    assert mcm.generate_config_template('US')['market_configs']['US']['major_event_sensitivity'] == 'HIGH'
    assert mcm.generate_config_template('CN')['market_configs']['CN']['major_event_sensitivity'] == 'MEDIUM'
    assert mcm.generate_config_template('EU')['market_configs']['EU']['major_event_sensitivity'] == 'HIGH'


def test_volatility_tiers_structure_cn():
    mcm = MarketConfig()
    tiers = mcm.generate_config_template('CN')['market_configs']['CN']['volatility_tiers']
    for k in ['NORMAL', 'MEDIUM', 'HIGH', 'EXTREME']:
        assert 'max' in tiers[k] and 'cache_ttl' in tiers[k]


def test_default_trading_hours_map():
    mcm = MarketConfig()
    for mt in ['CN', 'US', 'HK', 'JP', 'EU', 'SG']:
        cfg = mcm.generate_config_template(mt)
        hours = cfg['market_configs'][mt]['trading_hours']
        assert isinstance(hours, dict)
        assert 'regular' in hours


def test_brexit_risk_weight_lower_bound():
    mcm = MarketConfig()
    weight_today = mcm._get_brexit_risk_weight()
    assert weight_today >= 1.0


def test_var_method_priority_by_market():
    mcm = MarketConfig()
    cn = mcm.generate_config_template('CN')['market_configs']['CN']
    us = mcm.generate_config_template('US')['market_configs']['US']
    hk = mcm.generate_config_template('HK')['market_configs']['HK']
    eu = mcm.generate_config_template('EU')['market_configs']['EU']
    jp = mcm.generate_config_template('JP')['market_configs']['JP']
    sg = mcm.generate_config_template('SG')['market_configs']['SG']
    assert cn['var_method_priority'] == 'historical_simulation'
    assert us['var_method_priority'] == 't_distribution'
    assert hk['var_method_priority'] == 'evt'
    assert eu['var_method_priority'] == 'historical_simulation'
    assert jp['var_method_priority'] == 't_distribution'
    assert sg['var_method_priority'] == 'evt'


def test_covariance_lookback_values():
    mcm = MarketConfig()
    assert mcm.generate_config_template('CN')['market_configs']['CN']['covariance_lookback'] == 126
    assert mcm.generate_config_template('US')['market_configs']['US']['covariance_lookback'] == 756
    assert mcm.generate_config_template('HK')['market_configs']['HK']['covariance_lookback'] == 378
    assert mcm.generate_config_template('JP')['market_configs']['JP']['covariance_lookback'] == 504
    assert mcm.generate_config_template('EU')['market_configs']['EU']['covariance_lookback'] == 252
    assert mcm.generate_config_template('SG')['market_configs']['SG']['covariance_lookback'] == 189


def test_volatility_persistence_values():
    mcm = MarketConfig()
    assert mcm.generate_config_template('CN')['market_configs']['CN']['volatility_persistence'] == 0.94
    assert mcm.generate_config_template('US')['market_configs']['US']['volatility_persistence'] == 0.97
    assert mcm.generate_config_template('HK')['market_configs']['HK']['volatility_persistence'] == 0.92
    assert mcm.generate_config_template('JP')['market_configs']['JP']['volatility_persistence'] == 0.95
    assert mcm.generate_config_template('EU')['market_configs']['EU']['volatility_persistence'] == 0.90
    assert mcm.generate_config_template('SG')['market_configs']['SG']['volatility_persistence'] == 0.88


def test_liquidity_risk_weight_values():
    mcm = MarketConfig()
    assert mcm.generate_config_template('CN')['market_configs']['CN']['liquidity_risk_weight'] == 1.2
    assert mcm.generate_config_template('US')['market_configs']['US']['liquidity_risk_weight'] == 0.85
    assert mcm.generate_config_template('HK')['market_configs']['HK']['liquidity_risk_weight'] == 1.1
    assert mcm.generate_config_template('JP')['market_configs']['JP']['liquidity_risk_weight'] == 0.9
    assert mcm.generate_config_template('EU')['market_configs']['EU']['liquidity_risk_weight'] == 1.0
    assert mcm.generate_config_template('SG')['market_configs']['SG']['liquidity_risk_weight'] == 1.25


def test_limit_adjustment_and_min_required_returns_flags():
    mcm = MarketConfig()
    cn = mcm.generate_config_template('CN')['market_configs']['CN']
    hk = mcm.generate_config_template('HK')['market_configs']['HK']
    us = mcm.generate_config_template('US')['market_configs']['US']
    assert cn['limit_adjustment_enabled'] is True
    assert hk['limit_adjustment_enabled'] is True
    assert us['limit_adjustment_enabled'] is False
    assert cn['min_required_returns'] == 30
    assert hk['min_required_returns'] == 50
    assert us['min_required_returns'] == 50


def test_risk_premium_base_values():
    mcm = MarketConfig()
    expected = {'CN': 0.015, 'US': 0.010, 'HK': 0.020, 'JP': 0.008, 'EU': 0.012, 'SG': 0.014}
    for mt, val in expected.items():
        cfg = mcm.generate_config_template(mt)
        assert cfg['market_configs'][mt]['risk_premium_base'] == val


def test_us_trading_hours_keys():
    mcm = MarketConfig()
    us_hours = mcm.generate_config_template('US')['market_configs']['US']['trading_hours']
    assert 'pre_market' in us_hours and 'after_hours' in us_hours
    assert us_hours['after_hours'] == '16:00-20:00'


def test_performance_monitoring_defaults():
    mcm = MarketConfig()
    cfg = mcm.generate_config_template('US')
    pm = cfg['performance_monitoring']
    assert pm['enable_calculation_timing'] is True
    assert pm['enable_memory_monitoring'] is False
    assert pm['sample_size_warning_threshold'] == 50


def test_log_level_default_info():
    mcm = MarketConfig()
    cfg = mcm.generate_config_template('CN')
    assert cfg['log_level'] == 'INFO'


def test_dynamic_risk_free_rate_is_none():
    mcm = MarketConfig()
    cfg = mcm.generate_config_template('JP')
    assert cfg['dynamic_risk_free_rate'] is None


def test_market_registry_contains_all():
    mcm = MarketConfig()
    for mt in ['CN', 'US', 'HK', 'JP', 'EU', 'SG']:
        info = mcm.get_market_info(mt)
        assert 'currency' in info and 'default_trading_days' in info


def test_template_has_single_market_configs_key():
    mcm = MarketConfig()
    cfg = mcm.generate_config_template('HK')
    mc = cfg['market_configs']
    # 模板仅包含当前市场配置键
    assert list(mc.keys()) == ['HK']





def test_volatility_spike_thresholds_more_markets():
    mcm = MarketConfig()
    hk = mcm.generate_config_template('HK')['market_configs']['HK']
    jp = mcm.generate_config_template('JP')['market_configs']['JP']
    sg = mcm.generate_config_template('SG')['market_configs']['SG']
    assert hk['volatility_spike_threshold'] == 0.04
    assert jp['volatility_spike_threshold'] == 0.04
    assert sg['volatility_spike_threshold'] == 0.05


def test_limit_hit_ratio_thresholds_more_markets():
    mcm = MarketConfig()
    hk = mcm.generate_config_template('HK')['market_configs']['HK']
    jp = mcm.generate_config_template('JP')['market_configs']['JP']
    sg = mcm.generate_config_template('SG')['market_configs']['SG']
    assert hk['limit_hit_ratio_threshold'] == 0.25
    assert jp['limit_hit_ratio_threshold'] == 0.25
    assert sg['limit_hit_ratio_threshold'] == 0.30


def test_risk_free_rate_defaults():
    mcm = MarketConfig()
    expected = {'CN': 0.03, 'US': 0.045, 'HK': 0.035, 'JP': 0.005, 'EU': 0.025, 'SG': 0.030}
    for mt, rate in expected.items():
        cfg = mcm.generate_config_template(mt)
        assert cfg['market_configs'][mt]['risk_free_rate'] == rate


def test_unknown_market_fallback_to_cn():
    mcm = MarketConfig()
    cfg = mcm.generate_config_template('ZZ')
    assert cfg['market_type'] == 'CN'
    cn = cfg['market_configs']['CN']
    assert cn['risk_free_rate'] == 0.03
    assert 'volatility_spike_threshold' in cn





