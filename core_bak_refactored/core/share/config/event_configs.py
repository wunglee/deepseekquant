"""
事件窗口配置

职责：
- 定义不同事件类型的窗口参数
- 提供历史重大事件参数
- 支持事件驱动分析和回测

说明：
- EVENT_WINDOW_CONFIGS: 事件类型对应的窗口配置
- HISTORICAL_EVENT_PARAMS: 历史重大事件的实际参数（用于Mock数据生成和测试）
"""

# 事件窗口配置（用于事件驱动分析）
EVENT_WINDOW_CONFIGS = {
    'market_crash': {
        'window_days': 30,
        'baseline_days': 252,
        'description': '市场崩盘：快速暴跌事件'
    },
    'liquidity_crisis': {
        'window_days': 45,
        'baseline_days': 252,
        'description': '流动性危机：中期持续事件'
    },
    'currency_crisis': {
        'window_days': 60,
        'baseline_days': 252,
        'description': '货币危机：货币大幅贬值事件'
    },
    'sovereign_debt_crisis': {
        'window_days': 90,
        'baseline_days': 504,
        'description': '主权债务危机：结构性长期事件'
    },
    'geopolitical_risk': {
        'window_days': 30,
        'baseline_days': 252,
        'description': '地缘政治风险：突发冲击事件'
    }
}

# 历史重大事件参数（用于Mock数据生成和测试验证）
HISTORICAL_EVENT_PARAMS = {
    '2015_china_market_crash': {
        'period': ('2015-06-15', '2015-08-26'),
        'expected_decline': -0.43,
        'volatility_multiplier': 2.5,
        'description': '2015年中国股市崩盘'
    },
    'covid_19_pandemic': {
        'period': ('2020-02-20', '2020-03-23'),
        'expected_decline': -0.20,
        'volatility_multiplier': 3.0,
        'description': 'COVID-19疫情全球市场暴跌'
    },
    '2008_financial_crisis': {
        'period': ('2008-09-15', '2008-11-20'),
        'expected_decline': -0.40,
        'volatility_multiplier': 3.5,
        'description': '2008年金融危机'
    },
    '2011_eurozone_debt_crisis': {
        'period': ('2011-09-01', '2011-11-30'),
        'expected_decline': -0.25,
        'volatility_multiplier': 2.5,
        'description': '2011年欧债危机'
    },
    '2011_us_debt_ceiling_crisis': {
        'period': ('2011-07-22', '2011-08-10'),
        'expected_decline': -0.12,
        'volatility_multiplier': 2.0,
        'description': '2011年美国债务上限危机'
    },
    '2016_china_circuit_breaker': {
        'period': ('2016-01-04', '2016-01-08'),
        'expected_decline': -0.15,
        'volatility_multiplier': 2.0,
        'description': '2016年中国熔断机制事件'
    },
    '2022_russia_ukraine_conflict': {
        'period': ('2022-02-24', '2022-03-15'),
        'expected_decline': -0.12,
        'volatility_multiplier': 1.8,
        'description': '2022年俄乌冲突'
    },
    '1997_asian_financial_crisis': {
        'period': ('1997-07-02', '1998-08-28'),
        'expected_decline': -0.35,
        'volatility_multiplier': 2.8,
        'description': '1997年亚洲金融危机'
    }
}
