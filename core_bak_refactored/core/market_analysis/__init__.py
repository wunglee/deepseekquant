"""
市场分析模块

职责：
- 市场情绪评估
- 流动性分析
- 波动率状态判定
- 市场行为分析

设计原则：
- 独立于数据获取（依赖注入数据源）
- 业务规则清晰
- 可配置化
"""

from core_bak_refactored.core.market_analysis.sentiment_analyzer import (
    assess_market_sentiment,
    assess_liquidity_conditions,
    determine_volatility_regime
)

__all__ = [
    'assess_market_sentiment',
    'assess_liquidity_conditions',
    'determine_volatility_regime'
]
