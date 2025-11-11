"""
组合数据模型
从 core_bak/portfolio_manager.py 拆分
职责: 定义组合管理相关的枚举和数据结构
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

class AllocationMethod(Enum):
    """资产配置方法枚举"""
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    MARKET_CAP = "market_cap"  # 市值加权
    MIN_VARIANCE = "min_variance"  # 最小方差
    MAX_SHARPE = "max_sharpe"  # 最大夏普比率
    RISK_PARITY = "risk_parity"  # 风险平价
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman模型
    HRP = "hierarchical_risk_parity"  # 分层风险平价
    CLA = "critical_line_algorithm"  # 关键线算法
    CUSTOM = "custom"  # 自定义配置


class RebalanceFrequency(Enum):
    """再平衡频率枚举"""
    DAILY = "daily"  # 每日
    WEEKLY = "weekly"  # 每周
    MONTHLY = "monthly"  # 每月
    QUARTERLY = "quarterly"  # 每季度
    SEMI_ANNUAL = "semi_annual"  # 每半年
    ANNUAL = "annual"  # 每年
    AD_HOC = "ad_hoc"  # 临时调整
    SIGNAL_BASED = "signal_based"  # 基于信号


class RiskModel(Enum):
    """风险模型枚举"""
    SAMPLE_COVARIANCE = "sample_covariance"  # 样本协方差
    LEDOIT_WOLF = "ledoit_wolf"  # Ledoit-Wolf收缩
    ORACLE_APPROXIMATING = "oracle_approximating"  # Oracle近似收缩
    CONSTANT_CORRELATION = "constant_correlation"  # 常数相关
    EXPONENTIALLY_WEIGHTED = "exponentially_weighted"  # 指数加权
    GARCH = "garch"  # GARCH模型
    DCC_GARCH = "dcc_garch"  # DCC-GARCH模型


class PortfolioObjective(Enum):
    """组合优化目标枚举"""
    MAXIMIZE_RETURN = "maximize_return"  # 最大化收益
    MINIMIZE_RISK = "minimize_risk"  # 最小化风险
    MAXIMIZE_SHARPE = "maximize_sharpe"  # 最大化夏普比率
    MAXIMIZE_SORTINO = "maximize_sortino"  # 最大化索提诺比率
    MAXIMIZE_OMEGA = "maximize_omega"  # 最大化Omega比率
    MINIMIZE_CVAR = "minimize_cvar"  # 最小化条件风险价值
    MAXIMIZE_UTILITY = "maximize_utility"  # 最大化效用函数
    TRACK_ERROR = "track_error"  # 最小化跟踪误差


@dataclass
class PortfolioConstraints:
    """组合约束条件"""
    max_asset_weight: float = 0.2  # 单资产最大权重
    min_asset_weight: float = 0.0  # 单资产最小权重
    max_sector_weight: float = 0.3  # 单行业最大权重
    max_turnover: float = 0.2  # 最大换手率
    leverage_limit: float = 1.0  # 杠杆限制
    short_selling_limit: float = 0.0  # 卖空限制
    liquidity_constraints: Dict[str, float] = field(default_factory=dict)  # 流动性约束
    concentration_limit: float = 0.5  # 集中度限制
    risk_budget: Dict[str, float] = field(default_factory=dict)  # 风险预算
    trading_cost: float = 0.001  # 交易成本
    tax_consideration: bool = False  # 税务考虑
    regulatory_constraints: List[str] = field(default_factory=list)  # 监管约束


@dataclass
class PortfolioMetadata:
    """组合元数据"""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_rebalanced: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: AllocationMethod = AllocationMethod.MAX_SHARPE
    risk_model: RiskModel = RiskModel.LEDOIT_WOLF
    objective: PortfolioObjective = PortfolioObjective.MAXIMIZE_SHARPE
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    backtest_period: str = "3y"  # 回看期
    expected_return: float = 0.0  # 预期收益
    expected_risk: float = 0.0  # 预期风险
    sharpe_ratio: float = 0.0  # 夏普比率
    max_drawdown: float = 0.0  # 最大回撤
    diversification: float = 0.0  # 分散度
    turnover_rate: float = 0.0  # 换手率
    risk_parity_score: float = 0.0  # 风险平价得分
    liquidity_score: float = 1.0  # 流动性评分
    stress_test_passed: bool = True  # 压力测试结果
    regulatory_compliant: bool = True  # 监管合规性


@dataclass
class AssetAllocation:
    """资产配置"""
    symbol: str
    weight: float
    target_weight: float
    current_value: float
    target_value: float
    notional: float
    sector: str = ""
    asset_class: str = "equity"
    region: str = "domestic"
    currency: str = "USD"
    liquidity_tier: int = 1  # 流动性分级 1-5, 1最高
    risk_contribution: float = 0.0  # 风险贡献度
    marginal_risk: float = 0.0  # 边际风险
    expected_return: float = 0.0  # 预期收益
    expected_risk: float = 0.0  # 预期风险
    transaction_cost: float = 0.0  # 交易成本
    tax_implication: float = 0.0  # 税务影响
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """组合状态"""
    portfolio_id: str
    total_value: float
    cash_balance: float
    leveraged_value: float
    allocations: Dict[str, AssetAllocation]
    metadata: PortfolioMetadata
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    constraints: PortfolioConstraints
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"
    version: int = 1
    benchmark: str = "SPY"
    tracking_error: float = 0.0
    active_share: float = 0.0
    concentration: float = 0.0
    liquidity_metrics: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    regulatory_compliance: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'portfolio_id': self.portfolio_id,
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'leveraged_value': self.leveraged_value,
            'allocations': {symbol: asdict(alloc) for symbol, alloc in self.allocations.items()},
            'metadata': asdict(self.metadata),
            'performance': self.performance,
            'risk_metrics': self.risk_metrics,
            'constraints': asdict(self.constraints),
            'timestamp': self.timestamp,
            'status': self.status,
            'version': self.version,
            'benchmark': self.benchmark,
            'tracking_error': self.tracking_error,
            'active_share': self.active_share,
            'concentration': self.concentration,
            'liquidity_metrics': self.liquidity_metrics,
            'stress_test_results': self.stress_test_results,
            'regulatory_compliance': self.regulatory_compliance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioState':
        """从字典创建组合状态"""
        allocations = {}
        for symbol, alloc_data in data['allocations'].items():
            allocations[symbol] = AssetAllocation(**alloc_data)

        metadata_data = data['metadata']
        constraints_data = data['constraints']

        return cls(
            portfolio_id=data['portfolio_id'],
            total_value=data['total_value'],
            cash_balance=data['cash_balance'],
            leveraged_value=data['leveraged_value'],
            allocations=allocations,
            metadata=PortfolioMetadata(**metadata_data),
            performance=data['performance'],
            risk_metrics=data['risk_metrics'],
            constraints=PortfolioConstraints(**constraints_data),
            timestamp=data['timestamp'],
            status=data['status'],
            version=data['version'],
            benchmark=data['benchmark'],
            tracking_error=data['tracking_error'],
            active_share=data['active_share'],
            concentration=data['concentration'],
            liquidity_metrics=data['liquidity_metrics'],
            stress_test_results=data['stress_test_results'],
            regulatory_compliance=data['regulatory_compliance']
        )


