"""
风险管理系统 - 枚举定义
拆分自: core_bak/risk_manager.py (line 39-92)
职责: 定义所有风险相关的枚举类型
"""

from enum import Enum


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = "very_low"  # 极低风险
    LOW = "low"  # 低风险
    MODERATE = "moderate"  # 中等风险
    HIGH = "high"  # 高风险
    VERY_HIGH = "very_high"  # 极高风险
    EXTREME = "extreme"  # 极端风险
    BLACK_SWAN = "black_swan"  # 黑天鹅风险


class RiskType(Enum):
    """风险类型枚举"""
    MARKET_RISK = "market_risk"  # 市场风险
    CREDIT_RISK = "credit_risk"  # 信用风险
    LIQUIDITY_RISK = "liquidity_risk"  # 流动性风险
    OPERATIONAL_RISK = "operational_risk"  # 操作风险
    SYSTEMIC_RISK = "systemic_risk"  # 系统性风险
    CONCENTRATION_RISK = "concentration_risk"  # 集中度风险
    LEVERAGE_RISK = "leverage_risk"  # 杠杆风险
    COUNTERPARTY_RISK = "counterparty_risk"  # 对手方风险
    REGULATORY_RISK = "regulatory_risk"  # 监管风险
    MODEL_RISK = "model_risk"  # 模型风险


class RiskMetric(Enum):
    """风险指标枚举"""
    VOLATILITY = "volatility"  # 波动率
    VALUE_AT_RISK = "value_at_risk"  # 在险价值
    EXPECTED_SHORTFALL = "expected_shortfall"  # 预期短缺
    BETA = "beta"  # Beta系数
    CORRELATION = "correlation"  # 相关性
    DRAWDOWN = "drawdown"  # 回撤
    STRESS_TEST = "stress_test"  # 压力测试
    SCENARIO_ANALYSIS = "scenario_analysis"  # 情景分析
    LIQUIDITY_GAP = "liquidity_gap"  # 流动性缺口
    LEVERAGE_RATIO = "leverage_ratio"  # 杠杆比率
    RISK_CONTRIBUTION = "risk_contribution"  # 风险贡献度
    MARGINAL_RISK = "marginal_risk"  # 边际风险
    TAIL_RISK = "tail_risk"  # 尾部风险
    MAX_POSITION_SIZE = "max_position_size"  # 最大头寸规模


class RiskControlAction(Enum):
    """风险控制动作枚举"""
    ALLOW = "allow"  # 允许交易
    WARN = "warn"  # 警告但允许
    REDUCE = "reduce"  # 减少头寸
    REJECT = "reject"  # 拒绝交易
    HEDGE = "hedge"  # 对冲风险
    LIQUIDATE = "liquidate"  # 平仓
    SUSPEND = "suspend"  # 暂停交易
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断机制
