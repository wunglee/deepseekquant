"""
风险数据模型
从 core_bak/risk_manager.py 拆分
职责: 定义风险管理相关的枚举和数据结构

修订历史：
- 2024-11-12: 基于专家第3轮咨询修正（阶段1-数据模型层评审第1轮）
  * P0: RiskLevel简化为6级，移除BLACK_SWAN
  * P0: timestamp改为datetime类型
  * P1: 补充RiskType（5个新类型）和RiskMetric（核心指标）
  * P1: 新增TimeHorizon和CalculationMethod枚举
  * P1: 为嵌套结构定义专门dataclass
  * P1: 补充关键字段和评估维度
- 2024-11-12: 基于专家第4轮咨询修正（阶段1-数据模型层评审第2轮）
  * P0: 添加RiskLevel.from_legacy_value()兼容BLACK_SWAN
  * P0: 添加RiskType.BLACK_SWAN_EVENT事件类型
  * P0: RiskAssessment/RiskEvent添加__post_init__支持timestamp字符串转换
  * P1: 新增RecommendationType枚举
  * P1: LimitBreach补充breach_duration字段
  * P1: Recommendation补充created_at/status字段，type改为枚举
  * P1: RiskAssessment添加详细字段语义文档
  * P1: RiskLimit添加使用示例文档
  * P2: TimeHorizon添加display_name和timedelta属性
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger('DeepSeekQuant.RiskModels')

class RiskLevel(Enum):
    """风险等级枚举（专家修正：6级分类）
    
    等级定义与数值映射：
    - VERY_LOW: 0-20分，极低风险，无需特别关注
    - LOW: 20-40分，低风险，常规监控
    - MODERATE: 40-60分，中等风险，需要关注
    - HIGH: 60-80分，高风险，需要采取措施
    - VERY_HIGH: 80-95分，极高风险，紧急处理
    - EXTREME: 95-100分，极端风险，立即行动
    
    注：BLACK_SWAN作为事件类型已移至RiskType.BLACK_SWAN_EVENT
    """
    VERY_LOW = "very_low"  # 0-20分
    LOW = "low"  # 20-40分
    MODERATE = "moderate"  # 40-60分
    HIGH = "high"  # 60-80分
    VERY_HIGH = "very_high"  # 80-95分
    EXTREME = "extreme"  # 95-100分
    
    @classmethod
    def from_score(cls, score: float) -> 'RiskLevel':
        """从风险评分转换为风险等级"""
        if score < 20: return cls.VERY_LOW
        elif score < 40: return cls.LOW
        elif score < 60: return cls.MODERATE
        elif score < 80: return cls.HIGH
        elif score < 95: return cls.VERY_HIGH
        else: return cls.EXTREME
    
    @classmethod
    def from_legacy_value(cls, legacy_value: str) -> 'RiskLevel':
        """兼容旧BLACK_SWAN值（P0增强：向后兼容）
        
        Args:
            legacy_value: 旧的风险等级值，可能包含black_swan
            
        Returns:
            对应的RiskLevel，black_swan映射到EXTREME
        """
        if legacy_value == "black_swan":
            logger.warning("Legacy BLACK_SWAN risk level detected, mapping to EXTREME")
            return cls.EXTREME
        return cls(legacy_value)


class RiskType(Enum):
    """风险类型枚举（专家补充：新增5个关键类型）"""
    # 原有风险类型
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
    
    # 专家补充：通用金融风险
    CURRENCY_RISK = "currency_risk"  # 汇率风险（跨境投资必需）
    INTEREST_RATE_RISK = "interest_rate_risk"  # 利率风险（债券投资核心）
    INFLATION_RISK = "inflation_risk"  # 通胀风险（长期投资）
    POLITICAL_RISK = "political_risk"  # 政治风险（新兴市场）
    
    # 专家补充：量化交易特有风险
    ALGORITHMIC_RISK = "algorithmic_risk"  # 算法风险（策略逻辑错误）
    DATA_QUALITY_RISK = "data_quality_risk"  # 数据质量风险（延迟/错误）
    EXECUTION_RISK = "execution_risk"  # 执行风险（交易执行失败）
    TECHNOLOGY_RISK = "technology_risk"  # 技术风险（系统故障）
    
    # P0补充：黑天鹅事件类型（从RiskLevel迁移）
    BLACK_SWAN_EVENT = "black_swan_event"  # 黑天鹅事件（极端罕见的高影响事件）


class RiskMetric(Enum):
    """风险指标枚举（专家补充：核心指标完整化）"""
    # 原有指标
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
    
    # 专家补充：风险调整收益指标
    SHARPE_RATIO = "sharpe_ratio"  # 夏普比率
    SORTINO_RATIO = "sortino_ratio"  # 索提诺比率
    INFORMATION_RATIO = "information_ratio"  # 信息比率
    CALMAR_RATIO = "calmar_ratio"  # 卡玛比率
    
    # 专家补充：回撤指标细化
    MAX_DRAWDOWN = "max_drawdown"  # 最大回撤
    DRAWDOWN_DURATION = "drawdown_duration"  # 回撤持续时间
    
    # 专家补充：流动性指标
    BID_ASK_SPREAD = "bid_ask_spread"  # 买卖价差
    MARKET_IMPACT = "market_impact"  # 市场冲击
    LIQUIDATION_TIME = "liquidation_time"  # 清算时间
    VOLUME_RATIO = "volume_ratio"  # 成交量比率
    
    # 专家补充：跟踪误差
    TRACKING_ERROR = "tracking_error"  # 跟踪误差
    
    # 注：MAX_POSITION_SIZE已移除，应在RiskLimit配置中定义


class RiskControlAction(Enum):
    """风险控制动作枚举（系统自动执行）"""
    ALLOW = "allow"  # 允许交易
    WARN = "warn"  # 警告但允许
    REDUCE = "reduce"  # 减少头寸
    REJECT = "reject"  # 拒绝交易
    HEDGE = "hedge"  # 对冲风险
    LIQUIDATE = "liquidate"  # 平仓
    SUSPEND = "suspend"  # 暂停交易
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断机制


class RecommendationType(Enum):
    """风险建议类型枚举（P1新增：人工决策建议）
    
    与RiskControlAction的区别：
    - RiskControlAction: 系统自动执行的强制控制动作
    - RecommendationType: 给人工决策者的建议类型，更宏观和策略性
    """
    REDUCE = "reduce"  # 减少头寸
    HEDGE = "hedge"  # 对冲风险
    MONITOR = "monitor"  # 加强监控
    LIQUIDATE = "liquidate"  # 平仓
    DIVERSIFY = "diversify"  # 分散投资
    REBALANCE = "rebalance"  # 再平衡


class TimeHorizon(Enum):
    """时间范围枚举（专家建议：字符串改为枚举+添加显示名称）
    
    使用"数字+单位"格式符合金融行业标准，便于解析和计算。
    """
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"
    YEARLY = "1y"
    
    @property
    def display_name(self) -> str:
        """P2增强：显示名称（中文）"""
        names = {
            "1d": "每日",
            "1w": "每周",
            "1m": "每月",
            "1y": "每年"
        }
        return names.get(self.value, self.value)
    
    @property
    def timedelta(self) -> timedelta:
        """P2增强：转换为时间增量，便于计算"""
        return {
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1m": timedelta(days=30),  # 近似
            "1y": timedelta(days=365)
        }[self.value]


class CalculationMethod(Enum):
    """风险计算方法枚举（专家建议：字符串改为枚举）"""
    HISTORICAL = "historical"  # 历史模拟法
    PARAMETRIC = "parametric"  # 参数法
    MONTE_CARLO = "monte_carlo"  # 蒙特卡洛模拟


class ImpactLevel(Enum):
    """场景影响程度枚举（专家建议：独立于RiskLevel）"""
    NEGLIGIBLE = "negligible"  # 可忽略
    MINOR = "minor"  # 轻微
    MODERATE = "moderate"  # 中等
    SEVERE = "severe"  # 严重
    CATASTROPHIC = "catastrophic"  # 灾难性


@dataclass
class LimitBreach:
    """限额违反详情(专家建议：结构化替代Dict)
    
    记录风险限额被违反的核心信息，用于风险监控和预警。
    
    字段说明：
    - breach_amount: 超出阈值的绝对值，正数表示超出量
    - severity: 违规严重程度，基于超出比例评估
    - breach_duration_seconds: P1新增，违规持续时间(秒)，用于级联判断(P1命名优化：添加单位后缀)
    """
    limit_id: str
    risk_type: RiskType
    metric: RiskMetric
    current_value: float
    threshold: float
    breach_amount: float
    timestamp: datetime
    severity: RiskLevel = RiskLevel.MODERATE
    
    # P1补充+命名优化：违规持续时间(秒)
    breach_duration_seconds: int = 0  # 违规持续时间(秒，P1命名优化)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LimitBreach':
        """从字典创建LimitBreach对象(P0新增：序列化对称性)
        
        Args:
            data: 包含LimitBreach字段的字典
            
        Returns:
            LimitBreach实例
        """
        parsed_data = data.copy()
        
        # 解析timestamp
        if 'timestamp' in parsed_data and isinstance(parsed_data['timestamp'], str):
            try:
                parsed_data['timestamp'] = datetime.fromisoformat(parsed_data['timestamp'])
            except ValueError:
                logger.warning(f"Invalid timestamp: {parsed_data['timestamp']}, using now()")
                parsed_data['timestamp'] = datetime.now()
        
        # 解析risk_type
        if 'risk_type' in parsed_data:
            rt = parsed_data['risk_type']
            try:
                if isinstance(rt, dict) and 'value' in rt:
                    parsed_data['risk_type'] = RiskType(rt['value'])
                elif isinstance(rt, str):
                    parsed_data['risk_type'] = RiskType(rt)
                elif not isinstance(rt, RiskType):
                    raise TypeError(f"Unsupported risk_type type: {type(rt)}")
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Invalid risk_type: {rt}, error: {e}, using MARKET_RISK")
                parsed_data['risk_type'] = RiskType.MARKET_RISK
        
        # 解析metric
        if 'metric' in parsed_data:
            m = parsed_data['metric']
            try:
                if isinstance(m, dict) and 'value' in m:
                    parsed_data['metric'] = RiskMetric(m['value'])
                elif isinstance(m, str):
                    parsed_data['metric'] = RiskMetric(m)
                elif not isinstance(m, RiskMetric):
                    raise TypeError(f"Unsupported metric type: {type(m)}")
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Invalid metric: {m}, error: {e}, using VALUE_AT_RISK")
                parsed_data['metric'] = RiskMetric.VALUE_AT_RISK
        
        # 解析severity
        if 'severity' in parsed_data:
            s = parsed_data['severity']
            try:
                if isinstance(s, dict) and 'value' in s:
                    parsed_data['severity'] = RiskLevel(s['value'])
                elif isinstance(s, str):
                    parsed_data['severity'] = RiskLevel(s)
                elif not isinstance(s, RiskLevel):
                    raise TypeError(f"Unsupported severity type: {type(s)}")
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Invalid severity: {s}, error: {e}, using MODERATE")
                parsed_data['severity'] = RiskLevel.MODERATE
        
        # 兼容旧字段名(breach_duration -> breach_duration_seconds)
        if 'breach_duration' in parsed_data and 'breach_duration_seconds' not in parsed_data:
            parsed_data['breach_duration_seconds'] = parsed_data.pop('breach_duration')
        
        return cls(**parsed_data)


@dataclass
class Recommendation:
    """风险建议(专家建议：结构化替代Dict)
    
    给人工决策者的风险处置建议，区别于系统自动控制动作。
    
    使用示例：
    >>> rec = Recommendation(
    >>>     type=RecommendationType.REDUCE,
    >>>     priority=1,  # 1-10，数值越小优先级越高(最高=1, 最低=10)
    >>>     description="市场风险过高，建议减少权益仓位",
    >>>     action_items=["sell AAPL 100 shares", "reduce leverage"]
    >>> )
    
    字段说明：
    - type: P1修正，改为枚举类型，增强类型安全
    - priority: 1-10的整数，1最高，10最低(最高优先级=1, 最低优先级=10)
    - estimated_impact: 预期影响，正数表示风险降低，负数表示收益损失
    - created_at: P1新增，建议创建时间
    - status: P1新增，建议状态跟踪
    """
    type: RecommendationType  # P1修正：改为枚举类型
    priority: int  # 1-10，数值越小优先级越高(最高=1, 最低=10)
    description: str
    action_items: List[str]
    estimated_impact: float = 0.0  # 预期影响(正数表示风险降低)
    
    # P1补充：建议管理字段
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending/approved/rejected/completed
    
    def __post_init__(self):
        """P0新增：字段验证"""
        # 验证priority范围(1-10)
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be in [1,10], got {self.priority}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化。
        
        Returns:
            包含所有字段的字典，枚举值转换为字符串。
        """
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recommendation':
        """从字典创建Recommendation对象(P0新增：序列化对称性)
        
        Args:
            data: 包含Recommendation字段的字典
            
        Returns:
            Recommendation实例
        """
        parsed_data = data.copy()
        
        # 解析created_at
        if 'created_at' in parsed_data and isinstance(parsed_data['created_at'], str):
            try:
                parsed_data['created_at'] = datetime.fromisoformat(parsed_data['created_at'])
            except ValueError:
                logger.warning(f"Invalid created_at: {parsed_data['created_at']}, using now()")
                parsed_data['created_at'] = datetime.now()
        
        # 解析type
        if 'type' in parsed_data:
            t = parsed_data['type']
            try:
                if isinstance(t, dict) and 'value' in t:
                    parsed_data['type'] = RecommendationType(t['value'])
                elif isinstance(t, str):
                    parsed_data['type'] = RecommendationType(t)
                elif not isinstance(t, RecommendationType):
                    raise TypeError(f"Unsupported type: {type(t)}")
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Invalid recommendation type: {t}, error: {e}, using MONITOR")
                parsed_data['type'] = RecommendationType.MONITOR
        
        return cls(**parsed_data)


@dataclass
class RiskLimit:
    """风险限额配置（专家修正：枚举化+补充字段+使用示例）
    
    使用示例（P1文档增强）：
    >>> # 创建市场风险VaR限额
    >>> limit = RiskLimit(
    >>>     risk_type=RiskType.MARKET_RISK,
    >>>     metric=RiskMetric.VALUE_AT_RISK,
    >>>     threshold=0.05,  # 5% VaR限制
    >>>     time_horizon=TimeHorizon.DAILY,
    >>>     confidence_level=0.95,
    >>>     calculation_method=CalculationMethod.HISTORICAL,
    >>>     action=RiskControlAction.REJECT,
    >>>     scope="portfolio",
    >>>     priority=1  # 最高优先级
    >>> )
    >>>
    >>> # 序列化存储
    >>> data = limit.to_dict()
    >>>
    >>> # 反序列化恢复
    >>> restored = RiskLimit.from_dict(data)
    
    字段说明：
    - grace_period: 宽限期（秒），专家建议统一使用秒为单位
    - scope: 适用范围 (portfolio/strategy/asset_class/individual)
    - priority: 优先级，数值越小优先级越高
    - valid_from/valid_to: P1新增，限额有效期控制
    """
    risk_type: RiskType
    metric: RiskMetric
    threshold: float
    time_horizon: TimeHorizon = TimeHorizon.DAILY  # 专家建议：枚举化
    confidence_level: float = 0.95  # 置信水平
    calculation_method: CalculationMethod = CalculationMethod.HISTORICAL  # 专家建议：枚举化
    action: RiskControlAction = RiskControlAction.WARN
    grace_period: int = 0  # 宽限期（秒）- 专家建议：统一单位
    escalation_level: int = 1  # 升级级别
    is_hard_limit: bool = False  # 是否为硬性限额
    notification_channels: List[str] = field(default_factory=lambda: ["email", "dashboard"])
    review_required: bool = False  # 是否需要人工审核
    
    # 专家补充：有效期控制
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    is_active: bool = True
    
    # 专家补充：适用范围
    scope: str = "portfolio"  # "portfolio", "strategy", "asset_class", "individual"
    
    # 专家补充：优先级机制
    priority: int = 1  # 数值越小优先级越高
    
    # 专家补充：监管标记
    regulatory_required: bool = False

    def __post_init__(self):
        """P0新增：字段验证逻辑
        
        验证关键字段的取值范围，防止无效数据。
        """
        # 验证confidence_level必须在0-1范围
        if not 0 <= self.confidence_level <= 1:
            raise ValueError(f"confidence_level must be in [0,1], got {self.confidence_level}")
        
        # 验证priority必须为正数
        if self.priority <= 0:
            raise ValueError(f"priority must be positive, got {self.priority}")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.valid_from:
            result['valid_from'] = self.valid_from.isoformat()
        if self.valid_to:
            result['valid_to'] = self.valid_to.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskLimit':
        """从字典创建 RiskLimit，支持枚举字符串/字典容错解析（专家建议：增强异常处理）"""
        parsed_data = data.copy()
        
        # 解析 risk_type（专家建议：增加异常处理）
        if 'risk_type' in parsed_data:
            rt = parsed_data['risk_type']
            try:
                if isinstance(rt, dict) and 'value' in rt:
                    parsed_data['risk_type'] = RiskType(rt['value'])
                elif isinstance(rt, str):
                    parsed_data['risk_type'] = RiskType(rt)
                elif isinstance(rt, RiskType):
                    parsed_data['risk_type'] = rt
                else:
                    raise ValueError(f"Unsupported risk_type format: {rt}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid risk_type value: {rt}, using default MARKET_RISK")
                parsed_data['risk_type'] = RiskType.MARKET_RISK
        
        # 解析 metric
        if 'metric' in parsed_data:
            m = parsed_data['metric']
            try:
                if isinstance(m, dict) and 'value' in m:
                    parsed_data['metric'] = RiskMetric(m['value'])
                elif isinstance(m, str):
                    parsed_data['metric'] = RiskMetric(m)
                elif isinstance(m, RiskMetric):
                    parsed_data['metric'] = m
                else:
                    raise ValueError(f"Unsupported metric format: {m}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid metric value: {m}, using default VALUE_AT_RISK")
                parsed_data['metric'] = RiskMetric.VALUE_AT_RISK
        
        # 解析 action
        if 'action' in parsed_data:
            a = parsed_data['action']
            try:
                if isinstance(a, dict) and 'value' in a:
                    parsed_data['action'] = RiskControlAction(a['value'])
                elif isinstance(a, str):
                    parsed_data['action'] = RiskControlAction(a)
                elif isinstance(a, RiskControlAction):
                    parsed_data['action'] = a
                else:
                    raise ValueError(f"Unsupported action format: {a}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid action value: {a}, using default WARN")
                parsed_data['action'] = RiskControlAction.WARN
        
        # 解析 time_horizon（新增）
        if 'time_horizon' in parsed_data and isinstance(parsed_data['time_horizon'], str):
            try:
                parsed_data['time_horizon'] = TimeHorizon(parsed_data['time_horizon'])
            except ValueError:
                logger.warning(f"Invalid time_horizon: {parsed_data['time_horizon']}, using DAILY")
                parsed_data['time_horizon'] = TimeHorizon.DAILY
        
        # 解析 calculation_method（新增）
        if 'calculation_method' in parsed_data and isinstance(parsed_data['calculation_method'], str):
            try:
                parsed_data['calculation_method'] = CalculationMethod(parsed_data['calculation_method'])
            except ValueError:
                logger.warning(f"Invalid calculation_method: {parsed_data['calculation_method']}, using HISTORICAL")
                parsed_data['calculation_method'] = CalculationMethod.HISTORICAL
        
        # 解析 datetime 字段
        for field_name in ['valid_from', 'valid_to']:
            if field_name in parsed_data and isinstance(parsed_data[field_name], str):
                try:
                    parsed_data[field_name] = datetime.fromisoformat(parsed_data[field_name])
                except ValueError:
                    logger.warning(f"Invalid {field_name}: {parsed_data[field_name]}, setting to None")
                    parsed_data[field_name] = None
        
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
    """风险评估结果（专家修正：类型优化+补充维度+语义文档）
    
    字段语义约定（P1文档增强）：
    - value_at_risk: 正数表示潜在损失金额（0.05表示5%损失）
    - expected_shortfall: 正数表示极端损失期望值
    - max_drawdown: 负数表示跌幅（-0.15表示15%回撤）
    - beta: 系统风险暴露，>1波动大于市场
    - alpha: 超额收益能力，正数表示跑赢基准
    - sharpe_ratio: 风险调整收益，>1良好，>2优秀
    - sortino_ratio: 下行风险调整收益，仅考虑负面波动
    - volatility: 年化波动率，通常0-1为合理范围
    
    数值范围说明：
    - 风险评分: 0-100分，分数越高风险越大
    - 比率指标: 无上限，但通常0-3为合理范围
    - 百分比值: 0-1表示比例，>1表示倍数
    """
    timestamp: datetime  # 专家建议：改为datetime对象
    portfolio_id: str
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100风险评分
    
    # VaR/CVaR（注：正数表示损失金额，负数表示收益）
    value_at_risk: float
    expected_shortfall: float
    max_drawdown: float
    
    # 分项风险
    liquidity_risk: float
    concentration_risk: float
    leverage_risk: float
    
    # 测试结果
    stress_test_results: Dict[str, float]
    scenario_analysis: Dict[str, float]
    risk_contributions: Dict[str, float]
    
    # 专家补充：系统性风险指标（带默认值字段必须在后面）
    beta: float = 0.0  # 系统风险暴露
    alpha: float = 0.0  # 超额收益能力
    tracking_error: float = 0.0  # 跟踪误差
    
    # 专家补充：收益风险指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    volatility: float = 0.0  # 年化波动率
    
    # 专家建议：结构化违规和建议
    limit_breaches: List[LimitBreach] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    
    confidence_level: float = 0.95  # 评估置信度
    
    def __post_init__(self):
        """P0增强：timestamp初始化时支持字符串转换"""
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp)
            except ValueError:
                logger.warning(f"Invalid timestamp format: {self.timestamp}, using now()")
                self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['limit_breaches'] = [b.to_dict() if isinstance(b, LimitBreach) else b for b in self.limit_breaches]
        result['recommendations'] = [r.to_dict() if isinstance(r, Recommendation) else r for r in self.recommendations]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """从字典创建 RiskAssessment，支持枚举容错解析"""
        parsed_data = data.copy()
        
        # 解析 timestamp（专家建议）
        if 'timestamp' in parsed_data and isinstance(parsed_data['timestamp'], str):
            try:
                parsed_data['timestamp'] = datetime.fromisoformat(parsed_data['timestamp'])
            except ValueError:
                logger.warning(f"Invalid timestamp: {parsed_data['timestamp']}, using now()")
                parsed_data['timestamp'] = datetime.now()
        elif 'timestamp' not in parsed_data:
            parsed_data['timestamp'] = datetime.now()
        
        # 解析 overall_risk_level
        if 'overall_risk_level' in parsed_data:
            orl = parsed_data['overall_risk_level']
            try:
                if isinstance(orl, dict) and 'value' in orl:
                    parsed_data['overall_risk_level'] = RiskLevel(orl['value'])
                elif isinstance(orl, str):
                    parsed_data['overall_risk_level'] = RiskLevel(orl)
                elif isinstance(orl, RiskLevel):
                    parsed_data['overall_risk_level'] = orl
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid risk_level: {orl}, using MODERATE")
                parsed_data['overall_risk_level'] = RiskLevel.MODERATE
        
        # 解析 limit_breaches（专家建议：支持结构化）
        if 'limit_breaches' in parsed_data:
            breaches = []
            for b in parsed_data['limit_breaches']:
                if isinstance(b, dict):
                    # 尝试转换为LimitBreach，失败则保持dict
                    breaches.append(b)
                else:
                    breaches.append(b)
            parsed_data['limit_breaches'] = breaches
        
        # 解析 recommendations（专家建议：支持结构化）
        if 'recommendations' in parsed_data:
            recs = []
            for r in parsed_data['recommendations']:
                if isinstance(r, dict):
                    recs.append(r)
                else:
                    recs.append(r)
            parsed_data['recommendations'] = recs
        
        return cls(**parsed_data)


@dataclass
class RiskEvent:
    """风险事件记录"""
    event_id: str
    event_type: RiskType
    severity: RiskLevel
    timestamp: datetime  # 专家建议：改为datetime
    description: str
    triggered_by: str  # 触发因素
    impact_assessment: Dict[str, Any]  # 影响评估
    action_taken: RiskControlAction  # 采取的措施
    resolved: bool = False  # 是否已解决
    resolution_time: Optional[datetime] = None  # 专家建议：改为datetime
    root_cause: Optional[str] = None  # 根本原因
    prevention_measures: List[str] = field(default_factory=list)  # 预防措施
    
    def __post_init__(self):
        """P0增强：timestamp和resolution_time初始化时支持字符串转换"""
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp)
            except ValueError:
                logger.warning(f"Invalid timestamp format: {self.timestamp}, using now()")
                self.timestamp = datetime.now()
        
        if isinstance(self.resolution_time, str):
            try:
                self.resolution_time = datetime.fromisoformat(self.resolution_time)
            except ValueError:
                logger.warning(f"Invalid resolution_time format: {self.resolution_time}, setting to None")
                self.resolution_time = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolution_time:
            result['resolution_time'] = self.resolution_time.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskEvent':
        """从字典创建 RiskEvent，支持枚举容错解析"""
        parsed_data = data.copy()
        
        # 解析 timestamp
        if 'timestamp' in parsed_data and isinstance(parsed_data['timestamp'], str):
            try:
                parsed_data['timestamp'] = datetime.fromisoformat(parsed_data['timestamp'])
            except ValueError:
                parsed_data['timestamp'] = datetime.now()
        
        # 解析 resolution_time
        if 'resolution_time' in parsed_data and isinstance(parsed_data['resolution_time'], str):
            try:
                parsed_data['resolution_time'] = datetime.fromisoformat(parsed_data['resolution_time'])
            except ValueError:
                parsed_data['resolution_time'] = None
        
        # 解析 event_type
        if 'event_type' in parsed_data:
            et = parsed_data['event_type']
            try:
                if isinstance(et, dict) and 'value' in et:
                    parsed_data['event_type'] = RiskType(et['value'])
                elif isinstance(et, str):
                    parsed_data['event_type'] = RiskType(et)
            except (ValueError, KeyError):
                logger.warning(f"Invalid event_type: {et}, using MARKET_RISK")
                parsed_data['event_type'] = RiskType.MARKET_RISK
        
        # 解析 severity
        if 'severity' in parsed_data:
            s = parsed_data['severity']
            try:
                if isinstance(s, dict) and 'value' in s:
                    parsed_data['severity'] = RiskLevel(s['value'])
                elif isinstance(s, str):
                    parsed_data['severity'] = RiskLevel(s)
            except (ValueError, KeyError):
                logger.warning(f"Invalid severity: {s}, using MODERATE")
                parsed_data['severity'] = RiskLevel.MODERATE
        
        # 解析 action_taken
        if 'action_taken' in parsed_data:
            at = parsed_data['action_taken']
            try:
                if isinstance(at, dict) and 'value' in at:
                    parsed_data['action_taken'] = RiskControlAction(at['value'])
                elif isinstance(at, str):
                    parsed_data['action_taken'] = RiskControlAction(at)
            except (ValueError, KeyError):
                logger.warning(f"Invalid action_taken: {at}, using WARN")
                parsed_data['action_taken'] = RiskControlAction.WARN
        
        return cls(**parsed_data)


@dataclass
class StressTestScenario:
    """压力测试场景（专家修正：独立ImpactLevel+时间标准化）"""
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]  # 场景参数（专家建议：可结构化为ScenarioParameters）
    probability: float  # 发生概率（0-1）
    impact_level: ImpactLevel  # 专家建议：使用独立的ImpactLevel枚举
    duration_days: int  # 专家建议：持续时间（天数）
    triggers: List[str]  # 触发条件
    mitigation_strategies: List[str]  # 缓解策略
    historical_precedent: Optional[str] = None  # 历史先例
    recovery_days: Optional[int] = None  # 专家建议：恢复时间（天数）
    
    def __post_init__(self):
        """专家建议：添加验证逻辑"""
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")
        if self.probability < 0.0001:
            logger.warning(f"Scenario {self.scenario_id} has extremely low probability: {self.probability}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StressTestScenario':
        """从字典创建 StressTestScenario，支持枚举容错解析"""
        parsed_data = data.copy()
        
        # 解析 impact_level（专家建议：使用ImpactLevel）
        if 'impact_level' in parsed_data:
            il = parsed_data['impact_level']
            try:
                if isinstance(il, dict) and 'value' in il:
                    parsed_data['impact_level'] = ImpactLevel(il['value'])
                elif isinstance(il, str):
                    # 兼容旧数据：如果是RiskLevel的值，映射到ImpactLevel
                    if il in ['extreme', 'very_high']:
                        parsed_data['impact_level'] = ImpactLevel.CATASTROPHIC
                    elif il in ['high']:
                        parsed_data['impact_level'] = ImpactLevel.SEVERE
                    elif il in ['moderate']:
                        parsed_data['impact_level'] = ImpactLevel.MODERATE
                    elif il in ['low']:
                        parsed_data['impact_level'] = ImpactLevel.MINOR
                    else:
                        parsed_data['impact_level'] = ImpactLevel(il)
            except (ValueError, KeyError):
                logger.warning(f"Invalid impact_level: {il}, using MODERATE")
                parsed_data['impact_level'] = ImpactLevel.MODERATE
        
        # 兼容旧字段名（专家建议：duration字符串转为duration_days整数）
        if 'duration' in parsed_data and 'duration_days' not in parsed_data:
            duration_str = parsed_data.pop('duration')
            # 简单解析：提取数字（"18个月" -> 18*30, "6个月" -> 6*30, "1天" -> 1）
            import re
            match = re.search(r'(\d+)', duration_str)
            if match:
                num = int(match.group(1))
                if '月' in duration_str or 'month' in duration_str.lower():
                    parsed_data['duration_days'] = num * 30
                elif '年' in duration_str or 'year' in duration_str.lower():
                    parsed_data['duration_days'] = num * 365
                else:  # 天
                    parsed_data['duration_days'] = num
            else:
                parsed_data['duration_days'] = 1
        
        # 兼容旧字段名（recovery_time -> recovery_days）
        if 'recovery_time' in parsed_data and 'recovery_days' not in parsed_data:
            recovery_str = parsed_data.pop('recovery_time')
            if recovery_str:
                import re
                match = re.search(r'(\d+)', recovery_str)
                if match:
                    num = int(match.group(1))
                    if '月' in recovery_str or 'month' in recovery_str.lower():
                        parsed_data['recovery_days'] = num * 30
                    elif '年' in recovery_str or 'year' in recovery_str.lower():
                        parsed_data['recovery_days'] = num * 365
                    else:
                        parsed_data['recovery_days'] = num
        
        return cls(**parsed_data)


# =============================================================================
# 第5轮新增：压力测试结果标准化数据类（为5D集成准备）
# 基于专家answer.md 2.4节接口标准
# =============================================================================

@dataclass
class StressTestResult:
    """
    压力测试结果标准化输出（第5轮新增）
    
    用于5D风险计算协调器集成，提供统一的压力测试结果接口。
    
    基于专家answer.md 2.4节设计。
    
    Attributes:
        scenario_id: 场景ID
        scenario_name: 场景名称
        loss_amount: 损失金额（绝对值）
        loss_percentage: 损失百分比（相对值）
        confidence_level: 置信度水平，默认0.99（99%）
        risk_decomposition: 风险分解（市场/流动性/信用风险占比）
        recovery_period_months: 恢复期估计（月）
        triggered_actions: 触发的控制动作列表
        metadata: 其他元数据
    
    Example:
        >>> result = StressTestResult(
        ...     scenario_id='2008_financial_crisis',
        ...     scenario_name='2008金融危机',
        ...     loss_amount=1000000.0,
        ...     loss_percentage=-0.25,
        ...     confidence_level=0.99,
        ...     risk_decomposition={'market_risk': 0.7, 'liquidity_risk': 0.2, 'credit_risk': 0.1},
        ...     recovery_period_months=18,
        ...     triggered_actions=['WARN', 'REDUCE'],
        ...     metadata={'volatility_spike': 3.5, 'correlation_break': 0.8}
        ... )
    """
    scenario_id: str
    scenario_name: str
    loss_amount: float  # 损失金额
    loss_percentage: float  # 损失百分比
    confidence_level: float = 0.99  # 映射置信度
    risk_decomposition: Dict[str, float] = field(default_factory=dict)  # 风险分解
    recovery_period_months: int = 0  # 恢复期估计
    triggered_actions: List[str] = field(default_factory=list)  # 触发动作
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（供5D集成使用）"""
        return asdict(self)
    
    @classmethod
    def from_legacy_result(cls, scenario_id: str, scenario_name: str, 
                          loss_value: float, portfolio_value: float = 1000000.0,
                          **kwargs) -> 'StressTestResult':
        """
        从旧版本结果转换（向后兼容）
        
        Args:
            scenario_id: 场景ID
            scenario_name: 场景名称
            loss_value: 损失值（负数表示损失）
            portfolio_value: 组合总价值，默认100万
            **kwargs: 其他可选参数
        
        Returns:
            标准化的StressTestResult对象
        """
        loss_amount = abs(loss_value) if loss_value < 0 else loss_value
        loss_percentage = loss_value / portfolio_value if portfolio_value != 0 else loss_value
        
        return cls(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            **kwargs
        )


@dataclass
class CombinedStressTestResult:
    """
    组合场景测试结果（第5轮新增）
    
    用于顺序冲击/并发冲击/反馈循环测试结果输出。
    
    基于专家answer.md 2.4节设计。
    
    Attributes:
        test_type: 测试类型（sequential/concurrent/feedback）
        combined_loss: 组合损失（总计）
        individual_results: 各场景结果列表
        transmission_factors: 传导因子字典
        analysis: 分析结果
    
    Example:
        >>> result = CombinedStressTestResult(
        ...     test_type='sequential',
        ...     combined_loss=-0.45,
        ...     individual_results=[
        ...         StressTestResult(...),
        ...         StressTestResult(...)
        ...     ],
        ...     transmission_factors={'propagation': 0.3},
        ...     analysis={'max_scenario': '2008_financial_crisis', 'diversification_benefit': 0.15}
        ... )
    """
    test_type: str  # sequential/concurrent/feedback
    combined_loss: float  # 组合损失
    individual_results: List[StressTestResult] = field(default_factory=list)  # 各场景结果
    transmission_factors: Dict[str, float] = field(default_factory=dict)  # 传导因子
    analysis: Dict[str, Any] = field(default_factory=dict)  # 分析结果
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（供5D集成使用）"""
        return {
            'test_type': self.test_type,
            'combined_loss': self.combined_loss,
            'individual_results': [r.to_dict() for r in self.individual_results],
            'transmission_factors': self.transmission_factors,
            'analysis': self.analysis
        }
