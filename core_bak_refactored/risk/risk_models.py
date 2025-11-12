"""
风险管理系统 - 数据模型
拆分自: core_bak/risk_manager.py (line 95-215)
职责: 定义风险相关的数据类和模型
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from .risk_enums import RiskType, RiskMetric, RiskControlAction, RiskLevel


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
        """转换为字典"""
        result = asdict(self)
        # 处理枚举类型
        result['risk_type'] = self.risk_type.value
        result['metric'] = self.metric.value
        result['action'] = self.action.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskLimit':
        """从字典创建"""
        # 处理枚举类型
        if 'risk_type' in data and isinstance(data['risk_type'], str):
            data['risk_type'] = RiskType(data['risk_type'])
        if 'metric' in data and isinstance(data['metric'], str):
            data['metric'] = RiskMetric(data['metric'])
        if 'action' in data and isinstance(data['action'], str):
            data['action'] = RiskControlAction(data['action'])
        return cls(**data)


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
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionLimit':
        """从字典创建"""
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
        """转换为字典"""
        result = asdict(self)
        result['overall_risk_level'] = self.overall_risk_level.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """从字典创建"""
        if 'overall_risk_level' in data and isinstance(data['overall_risk_level'], str):
            data['overall_risk_level'] = RiskLevel(data['overall_risk_level'])
        return cls(**data)


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
        """转换为字典"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        result['action_taken'] = self.action_taken.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskEvent':
        """从字典创建"""
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = RiskType(data['event_type'])
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = RiskLevel(data['severity'])
        if 'action_taken' in data and isinstance(data['action_taken'], str):
            data['action_taken'] = RiskControlAction(data['action_taken'])
        return cls(**data)


@dataclass
class StressTestScenario:
    """压力测试场景"""
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]  # 场景参数
    probability: float  # 发生概率
    severity: RiskLevel = RiskLevel.HIGH
    expected_impact: Optional[float] = None  # 预期影响
    historical_occurrences: List[str] = field(default_factory=list)  # 历史发生日期

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['severity'] = self.severity.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StressTestScenario':
        """从字典创建"""
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = RiskLevel(data['severity'])
        return cls(**data)
