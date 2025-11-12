"""
组合管理系统 - 枚举和数据模型
拆分自: core_bak/portfolio_manager.py (line 50-238)
职责: 定义组合管理相关的所有枚举和数据类
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any
from enum import Enum
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
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    AD_HOC = "ad_hoc"
    SIGNAL_BASED = "signal_based"


class RiskModel(Enum):
    """风险模型枚举"""
    SAMPLE_COVARIANCE = "sample_covariance"
    LEDOIT_WOLF = "ledoit_wolf"
    ORACLE_APPROXIMATING = "oracle_approximating"
    CONSTANT_CORRELATION = "constant_correlation"
    EXPONENTIALLY_WEIGHTED = "exponentially_weighted"
    GARCH = "garch"
    DCC_GARCH = "dcc_garch"


class PortfolioObjective(Enum):
    """组合优化目标枚举"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_SORTINO = "maximize_sortino"
    MAXIMIZE_OMEGA = "maximize_omega"
    MINIMIZE_CVAR = "minimize_cvar"
    MAXIMIZE_UTILITY = "maximize_utility"
    TRACK_ERROR = "track_error"


@dataclass
class PortfolioConstraints:
    """组合约束条件"""
    max_asset_weight: float = 0.2
    min_asset_weight: float = 0.0
    max_sector_weight: float = 0.3
    max_turnover: float = 0.2
    leverage_limit: float = 1.0
    short_selling_limit: float = 0.0
    liquidity_constraints: Dict[str, float] = field(default_factory=lambda: {})
    concentration_limit: float = 0.5
    risk_budget: Dict[str, float] = field(default_factory=lambda: {})
    trading_cost: float = 0.001
    tax_consideration: bool = False
    regulatory_constraints: List[str] = field(default_factory=list)


@dataclass
class PortfolioMetadata:
    """组合元数据"""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_rebalanced: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: AllocationMethod = AllocationMethod.MAX_SHARPE
    risk_model: RiskModel = RiskModel.LEDOIT_WOLF
    objective: PortfolioObjective = PortfolioObjective.MAXIMIZE_SHARPE
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    backtest_period: str = "3y"
    expected_return: float = 0.0
    expected_risk: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    diversification: float = 0.0
    turnover_rate: float = 0.0
    risk_parity_score: float = 0.0
    liquidity_score: float = 1.0
    stress_test_passed: bool = True
    regulatory_compliant: bool = True


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
    liquidity_tier: int = 1
    risk_contribution: float = 0.0
    marginal_risk: float = 0.0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    transaction_cost: float = 0.0
    tax_implication: float = 0.0
    constraints: Dict[str, Any] = field(default_factory=lambda: {})
    metadata: Dict[str, Any] = field(default_factory=lambda: {})


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
    liquidity_metrics: Dict[str, float] = field(default_factory=lambda: {})
    stress_test_results: Dict[str, Any] = field(default_factory=lambda: {})
    regulatory_compliance: Dict[str, bool] = field(default_factory=lambda: {})

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
        if 'optimization_method' in metadata_data and isinstance(metadata_data['optimization_method'], str):
            metadata_data['optimization_method'] = AllocationMethod(metadata_data['optimization_method'])
        if 'risk_model' in metadata_data and isinstance(metadata_data['risk_model'], str):
            metadata_data['risk_model'] = RiskModel(metadata_data['risk_model'])
        if 'objective' in metadata_data and isinstance(metadata_data['objective'], str):
            metadata_data['objective'] = PortfolioObjective(metadata_data['objective'])

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
