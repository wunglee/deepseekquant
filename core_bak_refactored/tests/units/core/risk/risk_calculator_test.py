import pytest

from core_bak_refactored.core.risk.risk_calculator import RiskCalculator


def test_dynamic_strict_enabled_triggers():
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'US': 0.25},
            'component_weights': {'multi_currency': 0.5, 'cross_border': 0.5},
            'comprehensive_trigger_score': 0.65
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {
            'prices': {
                'AAPL': {'close': [100.0], 'currency': 'USD'},
                'TSM': {'close': [100.0], 'currency': 'TWD'},
                'BABA': {'close': [100.0], 'currency': 'HKD'},
            }
        },
        'portfolio': {
            'allocations': {
                'AAPL': {'weight': 0.2},
                'TSM': {'weight': 0.4},
                'BABA': {'weight': 0.4},
            },
            'cross_border_exposure': 0.60,
        }
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # multi_currency = 0.8, cross_border = 0.6, comp = 0.8*0.5 + 0.6*0.5 = 0.7 >= 0.65
    assert decision is True


def test_data_quality_multi_enabled():
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'data_quality_assessment': {
            'enabled': True,
            'base_weights': {'completeness': 1.0},
            'grade_thresholds': {'A': 90, 'B': 75, 'C': 60, 'D': 0},
            'usage_scenarios': {'reporting_only': True, 'affect_calculation': False}
        }
    }
    rc = RiskCalculator(config)
    prices = {
        'AAPL': {'close': [100.0], 'currency': 'USD'},
        'MSFT': {'close': [200.0], 'currency': 'USD'},
    }
    dq_cfg = config['data_quality_assessment']
    result = rc._assess_data_quality_multi({'prices': prices}, dq_cfg)
    assert result is not None
    assert result['quality_grade'] == 'A'
    assert result['dimension_scores']['completeness'] == 100.0


def test_dynamic_strict_disabled_returns_none():
    """配置未启用时，动态严格模式不覆盖静态模式"""
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'dynamic_currency_strict_mode': {'enabled': False}
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {'prices': {'AAPL': {'close': [100.0], 'currency': 'USD'}}},
        'portfolio': {'allocations': {'AAPL': {'weight': 1.0}}, 'cross_border_exposure': 0.5}
    }
    decision = rc._determine_dynamic_strict_mode(data)
    assert decision is None


def test_dynamic_strict_missing_config_returns_none():
    """缺少必要配置项时返回None"""
    config = {
        'market_type': 'CN',
        'market_configs': {'CN': {'base_currency': 'CNY'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            # 缺少 component_weights 和 comprehensive_trigger_score
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {'prices': {'000001.SZ': {'close': [10.0], 'currency': 'CNY'}}},
        'portfolio': {'allocations': {'000001.SZ': {'weight': 1.0}}}
    }
    decision = rc._determine_dynamic_strict_mode(data)
    assert decision is None


def test_data_quality_multi_disabled_returns_none():
    """配置未启用时，多维数据质量评估返回None"""
    config = {
        'market_type': 'HK',
        'market_configs': {'HK': {'base_currency': 'HKD'}},
        'data_quality_assessment': {'enabled': False}
    }
    rc = RiskCalculator(config)
    prices = {'0700.HK': {'close': [300.0], 'currency': 'HKD'}}
    dq_cfg = config['data_quality_assessment']
    result = rc._assess_data_quality_multi({'prices': prices}, dq_cfg)
    assert result is None


def test_dynamic_strict_cross_market_thresholds():
    """跨市场阈值差异化：HK市场跨境敞口阈值40%"""
    config = {
        'market_type': 'HK',
        'market_configs': {'HK': {'base_currency': 'HKD'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'HK': 0.40},
            'component_weights': {'multi_currency': 0.6, 'cross_border': 0.4},
            'comprehensive_trigger_score': 0.60
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {
            'prices': {
                '0700.HK': {'close': [300.0], 'currency': 'HKD'},
                'BABA': {'close': [100.0], 'currency': 'USD'},
            }
        },
        'portfolio': {
            'allocations': {
                '0700.HK': {'weight': 0.3},
                'BABA': {'weight': 0.7},
            },
            'cross_border_exposure': 0.50,
        }
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # multi_currency = 0.7, cross_border = 0.5, comp = 0.7*0.6 + 0.5*0.4 = 0.62 >= 0.60
    assert decision is True


def test_dynamic_strict_jp_market():
    """JP市场跨境敞口阈值20%"""
    config = {
        'market_type': 'JP',
        'market_configs': {'JP': {'base_currency': 'JPY'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'JP': 0.20},
            'component_weights': {'multi_currency': 0.5, 'cross_border': 0.5},
            'comprehensive_trigger_score': 0.60
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {
            'prices': {
                'SONY': {'close': [100.0], 'currency': 'JPY'},
                'AAPL': {'close': [100.0], 'currency': 'USD'},
                'BABA': {'close': [100.0], 'currency': 'HKD'},
            }
        },
        'portfolio': {
            'allocations': {
                'SONY': {'weight': 0.4},
                'AAPL': {'weight': 0.3},
                'BABA': {'weight': 0.3},
            },
            'cross_border_exposure': 0.60,
        }
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # comp = 0.6*0.5 + 0.6*0.5 = 0.60 >= 0.60
    assert decision is True


def test_dynamic_strict_sg_market():
    """SG市场跨境敞口阈值35%"""
    config = {
        'market_type': 'SG',
        'market_configs': {'SG': {'base_currency': 'SGD'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'SG': 0.35},
            'component_weights': {'multi_currency': 0.5, 'cross_border': 0.5},
            'comprehensive_trigger_score': 0.60
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {
            'prices': {
                'SINGTEL': {'close': [100.0], 'currency': 'SGD'},
                'AAPL': {'close': [100.0], 'currency': 'USD'},
                'BABA': {'close': [100.0], 'currency': 'HKD'},
            }
        },
        'portfolio': {
            'allocations': {
                'SINGTEL': {'weight': 0.3},
                'AAPL': {'weight': 0.4},
                'BABA': {'weight': 0.3},
            },
            'cross_border_exposure': 0.50,
        }
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # comp = 0.7*0.5 + 0.5*0.5 = 0.60 >= 0.60
    assert decision is True


def test_dynamic_strict_cn_market():
    """CN市场跨境敞口阈值50%"""
    config = {
        'market_type': 'CN',
        'market_configs': {'CN': {'base_currency': 'CNY'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'CN': 0.50},
            'component_weights': {'multi_currency': 0.5, 'cross_border': 0.5},
            'comprehensive_trigger_score': 0.65
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {
            'prices': {
                '000001.SZ': {'close': [10.0], 'currency': 'CNY'},
                'AAPL': {'close': [100.0], 'currency': 'USD'},
                'BABA': {'close': [100.0], 'currency': 'HKD'},
            }
        },
        'portfolio': {
            'allocations': {
                '000001.SZ': {'weight': 0.28},
                'AAPL': {'weight': 0.42},
                'BABA': {'weight': 0.30},
            },
            'cross_border_exposure': 0.60,
        }
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # comp = 0.72*0.5 + 0.6*0.5 = 0.66 >= 0.65
    assert decision is True


def test_dynamic_strict_eu_market():
    """EU市场跨境敞口阈值30%"""
    config = {
        'market_type': 'EU',
        'market_configs': {'EU': {'base_currency': 'EUR'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'EU': 0.30},
            'component_weights': {'multi_currency': 0.6, 'cross_border': 0.4},
            'comprehensive_trigger_score': 0.65
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {
            'prices': {
                'SAP': {'close': [100.0], 'currency': 'EUR'},
                'AAPL': {'close': [100.0], 'currency': 'USD'},
                'VOD': {'close': [100.0], 'currency': 'GBP'},
            }
        },
        'portfolio': {
            'allocations': {
                'SAP': {'weight': 0.2},
                'AAPL': {'weight': 0.4},
                'VOD': {'weight': 0.4},
            },
            'cross_border_exposure': 0.60,
        }
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # comp = 0.8*0.6 + 0.6*0.4 = 0.72 >= 0.65
    assert decision is True


def test_dynamic_strict_boundary_64():
    """综合评分=0.64（低于阈值）"""
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'US': 0.25},
            'component_weights': {'multi_currency': 0.5, 'cross_border': 0.5},
            'comprehensive_trigger_score': 0.65
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {'prices': {'AAPL': {'currency': 'USD'}, 'BABA': {'currency': 'HKD'}}},
        'portfolio': {'allocations': {'AAPL': {'weight': 0.32}, 'BABA': {'weight': 0.68}}, 'cross_border_exposure': 0.60}
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # comp = 0.68*0.5 + 0.6*0.5 = 0.64 < 0.65
    assert decision is False


def test_dynamic_strict_boundary_65():
    """综合评分=0.65（等于阈值）"""
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'dynamic_currency_strict_mode': {
            'enabled': True,
            'multi_currency_ratio_threshold': 0.30,
            'cross_border_exposure_threshold': {'US': 0.25},
            'component_weights': {'multi_currency': 0.5, 'cross_border': 0.5},
            'comprehensive_trigger_score': 0.65
        }
    }
    rc = RiskCalculator(config)
    data = {
        'market_data': {'prices': {'AAPL': {'currency': 'USD'}, 'BABA': {'currency': 'HKD'}}},
        'portfolio': {'allocations': {'AAPL': {'weight': 0.28}, 'BABA': {'weight': 0.72}}, 'cross_border_exposure': 0.60}
    }
    decision = rc._determine_dynamic_strict_mode(data)
    # comp = 0.72*0.5 + 0.6*0.5 = 0.66 >= 0.65
    assert decision is True


def test_data_quality_grade_b():
    """B级数据质量（75≤score<90）"""
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'data_quality_assessment': {
            'enabled': True,
            'base_weights': {'completeness': 1.0},
            'grade_thresholds': {'A': 90, 'B': 75, 'C': 60, 'D': 0}
        }
    }
    rc = RiskCalculator(config)
    prices = {
        'S1': {'currency': 'USD'}, 'S2': {'currency': 'USD'}, 'S3': {'currency': 'USD'}, 'S4': {'currency': 'USD'}, 'S5': {}
    }
    result = rc._assess_data_quality_multi({'prices': prices}, config['data_quality_assessment'])
    assert result is not None
    assert result['quality_grade'] == 'B'


def test_data_quality_grade_c():
    """C级数据质量（60≤score<75）"""
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'data_quality_assessment': {
            'enabled': True,
            'base_weights': {'completeness': 1.0},
            'grade_thresholds': {'A': 90, 'B': 75, 'C': 60, 'D': 0}
        }
    }
    rc = RiskCalculator(config)
    prices = {
        'S1': {'currency': 'USD'}, 'S2': {'currency': 'USD'}, 'S3': {'currency': 'USD'}, 'S4': {}, 'S5': {}
    }
    result = rc._assess_data_quality_multi({'prices': prices}, config['data_quality_assessment'])
    assert result is not None
    assert result['quality_grade'] == 'C'


def test_data_quality_grade_d():
    """D级数据质量（score<60）"""
    config = {
        'market_type': 'US',
        'market_configs': {'US': {'base_currency': 'USD'}},
        'data_quality_assessment': {
            'enabled': True,
            'base_weights': {'completeness': 1.0},
            'grade_thresholds': {'A': 90, 'B': 75, 'C': 60, 'D': 0}
        }
    }
    rc = RiskCalculator(config)
    prices = {
        'S1': {'currency': 'USD'}, 'S2': {}, 'S3': {}, 'S4': {}, 'S5': {}
    }
    result = rc._assess_data_quality_multi({'prices': prices}, config['data_quality_assessment'])
    assert result is not None
    assert result['quality_grade'] == 'D'

