"""
风险数据模型
从 core_bak/risk_manager.py 拆分
职责: 定义风险管理相关的枚举和数据结构
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any

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


@dataclass
class RiskLimit:
    """风险限额配置"""
    risk_type: RiskType
    metric: RiskMetric
    threshold: float
    time_horizon: str = "1d"  # 时间范围: 1d, 1w, 1m, 1y
    confidence_level: float = 0.95  # 置信水平
    calculation_method: str = "historical"  # 计算方法: historical, parametric, monte_carlo
    action: RiskControlAction = RiskControlAction.WARN
    grace_period: int = 0  # 宽限期（分钟）
    escalation_level: int = 1  # 升级级别
    is_hard_limit: bool = False  # 是否为硬性限额
    notification_channels: List[str] = field(default_factory=lambda: ["email", "dashboard"])
    review_required: bool = False  # 是否需要人工审核

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskLimit':
        """
        从字典创建 RiskLimit，支持枚举字符串/字典容错解析
        
        Args:
            data: 配置字典，支持以下格式：
                - 枚举对象：RiskType.MARKET_RISK
                - 字符串：'market_risk'
                - 字典：{'value': 'market_risk'}
        """
        # 复制数据避免修改原始字典
        parsed_data = data.copy()
        
        # 解析 risk_type
        if 'risk_type' in parsed_data:
            rt = parsed_data['risk_type']
            if isinstance(rt, dict) and 'value' in rt:
                parsed_data['risk_type'] = RiskType(rt['value'])
            elif isinstance(rt, str):
                parsed_data['risk_type'] = RiskType(rt)
            elif not isinstance(rt, RiskType):
                parsed_data['risk_type'] = RiskType.MARKET_RISK  # 默认值
        
        # 解析 metric
        if 'metric' in parsed_data:
            m = parsed_data['metric']
            if isinstance(m, dict) and 'value' in m:
                parsed_data['metric'] = RiskMetric(m['value'])
            elif isinstance(m, str):
                parsed_data['metric'] = RiskMetric(m)
            elif not isinstance(m, RiskMetric):
                parsed_data['metric'] = RiskMetric.VALUE_AT_RISK  # 默认值
        
        # 解析 action
        if 'action' in parsed_data:
            a = parsed_data['action']
            if isinstance(a, dict) and 'value' in a:
                parsed_data['action'] = RiskControlAction(a['value'])
            elif isinstance(a, str):
                parsed_data['action'] = RiskControlAction(a)
            elif not isinstance(a, RiskControlAction):
                parsed_data['action'] = RiskControlAction.WARN  # 默认值
        
        return cls(**parsed_data)


@dataclass
class PositionLimit:
    """头寸限额配置"""
    symbol: str
    max_notional: float  # 最大名义价值
    max_quantity: float  # 最大数量
    max_weight: float  # 最大权重
    min_liquidity_ratio: float = 0.1  # 最小流动性比率
    max_leverage: float = 1.0  # 最大杠杆
    concentration_limit: float = 0.2  # 集中度限制
    sector_limit: float = 0.3  # 行业限制
    region_limit: float = 0.4  # 地区限制
    var_limit: float = -0.05  # VaR限制
    stress_test_limit: float = -0.15  # 压力测试限制

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionLimit':
        return cls(**data)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    timestamp: str
    portfolio_id: str
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100风险评分
    value_at_risk: float  # 在险价值
    expected_shortfall: float  # 预期短缺
    max_drawdown: float  # 最大回撤
    liquidity_risk: float  # 流动性风险
    concentration_risk: float  # 集中度风险
    leverage_risk: float  # 杠杆风险
    stress_test_results: Dict[str, float]  # 压力测试结果
    scenario_analysis: Dict[str, float]  # 情景分析结果
    risk_contributions: Dict[str, float]  # 风险贡献度
    limit_breaches: List[Dict[str, Any]]  # 限额违反情况
    recommendations: List[Dict[str, Any]]  # 风险建议
    confidence_level: float = 0.95  # 评估置信度

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """从字典创建 RiskAssessment，支持枚举容错解析"""
        parsed_data = data.copy()
        
        # 解析 overall_risk_level
        if 'overall_risk_level' in parsed_data:
            orl = parsed_data['overall_risk_level']
            if isinstance(orl, dict) and 'value' in orl:
                parsed_data['overall_risk_level'] = RiskLevel(orl['value'])
            elif isinstance(orl, str):
                parsed_data['overall_risk_level'] = RiskLevel(orl)
        
        return cls(**parsed_data)


@dataclass
class RiskEvent:
    """风险事件记录"""
    event_id: str
    event_type: RiskType
    severity: RiskLevel
    timestamp: str
    description: str
    triggered_by: str  # 触发因素
    impact_assessment: Dict[str, Any]  # 影响评估
    action_taken: RiskControlAction  # 采取的措施
    resolved: bool = False  # 是否已解决
    resolution_time: Optional[str] = None  # 解决时间
    root_cause: Optional[str] = None  # 根本原因
    prevention_measures: List[str] = field(default_factory=list)  # 预防措施

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskEvent':
        """从字典创建 RiskEvent，支持枚举容错解析"""
        parsed_data = data.copy()
        
        # 解析 event_type
        if 'event_type' in parsed_data:
            et = parsed_data['event_type']
            if isinstance(et, dict) and 'value' in et:
                parsed_data['event_type'] = RiskType(et['value'])
            elif isinstance(et, str):
                parsed_data['event_type'] = RiskType(et)
        
        # 解析 severity
        if 'severity' in parsed_data:
            s = parsed_data['severity']
            if isinstance(s, dict) and 'value' in s:
                parsed_data['severity'] = RiskLevel(s['value'])
            elif isinstance(s, str):
                parsed_data['severity'] = RiskLevel(s)
        
        # 解析 action_taken
        if 'action_taken' in parsed_data:
            at = parsed_data['action_taken']
            if isinstance(at, dict) and 'value' in at:
                parsed_data['action_taken'] = RiskControlAction(at['value'])
            elif isinstance(at, str):
                parsed_data['action_taken'] = RiskControlAction(at)
        
        return cls(**parsed_data)


@dataclass
class StressTestScenario:
    """压力测试场景"""
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]  # 场景参数
    probability: float  # 发生概率
    impact_level: RiskLevel  # 影响程度
    duration: str  # 持续时间
    triggers: List[str]  # 触发条件
    mitigation_strategies: List[str]  # 缓解策略
    historical_precedent: Optional[str] = None  # 历史先例
    recovery_time: Optional[str] = None  # 恢复时间

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StressTestScenario':
        """从字典创建 StressTestScenario，支持枚举容错解析"""
        parsed_data = data.copy()
        
        # 解析 impact_level
        if 'impact_level' in parsed_data:
            il = parsed_data['impact_level']
            if isinstance(il, dict) and 'value' in il:
                parsed_data['impact_level'] = RiskLevel(il['value'])
            elif isinstance(il, str):
                parsed_data['impact_level'] = RiskLevel(il)
        
        return cls(**parsed_data)

