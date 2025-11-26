from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import uuid

from core_bak_refactored.core.backtest._fragments.event_window_backtester import BacktestResult


@dataclass
class StressTestResult:
    """
    标准化压力测试结果数据结构（5D 接口设计）
    注意：仅定义数据结构与转换助手，不引入新的业务逻辑。
    """
    report_id: str
    portfolio_id: str
    scenario_id: str

    # 风险指标（可选，若当前框架无数据则为None）
    var_normal: Optional[float] = None
    var_stressed: Optional[float] = None

    # 压力损失
    stress_loss_amount: Optional[float] = None  # 金额（当前框架不提供金额口径）
    stress_loss_percentage: Optional[float] = None  # 百分比（来自回测结果 actual_loss）

    # 恢复期（当前框架不提供，预留）
    recovery_period: Optional[int] = None

    # 风险分解（当前框架不提供，预留）
    risk_decomposition: Dict[str, float] = field(default_factory=dict)

    # 动作与建议（当前框架不提供，预留）
    triggered_actions: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    # 合规状态（预留字段，不做业务判断）
    compliance_status: Optional[str] = None

    # 附加元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def from_backtest_result(r: BacktestResult) -> StressTestResult:
    """
    将回测结果转换为标准化压力测试结果（仅映射可用字段）。
    不引入业务判断，不设定默认业务参数。
    """
    return StressTestResult(
        report_id=str(uuid.uuid4()),
        portfolio_id=r.portfolio_id,
        scenario_id=r.event_id,
        var_normal=None,
        var_stressed=None,
        stress_loss_amount=None,
        stress_loss_percentage=float(r.actual_loss) if r is not None else None,
        recovery_period=None,
        risk_decomposition={},
        triggered_actions=[],
        recommended_actions=[],
        compliance_status=None,
        metadata={
            'event_name': r.metadata.get('event_name') if r.metadata else None,
            'period': r.metadata.get('period') if r.metadata else None,
            'predicted_loss': float(r.predicted_loss),
            'actual_loss': float(r.actual_loss),
            'prediction_error': float(r.prediction_error),
            'benchmark_index': r.benchmark_index,
        },
    )
