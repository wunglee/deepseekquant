from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Issue:
    field: str
    message: str
    severity: str  # INFO/WARNING/ERROR


@dataclass
class QualityReport:
    issues: List[Issue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)  # completeness_ok, consistency_ok, timeliness_ok, reliability_ok


@dataclass
class QualityFlags:
    flags: Dict[str, bool] = field(default_factory=dict)  # e.g., {"DATA_QUALITY_WARNING": True}
    reasons: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """统一的数据质量报告类，供所有数据提供者使用
    
    评分维度（采用akshare的加权计算方式）：
    - 完整性（30%）
    - 一致性（30%）
    - 准确性（20%）
    - 异常值（20%）
    """
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    outliers_detected: int = 0
    total_rows: int = 0
    missing_values: int = 0
    overall_score: float = field(init=False)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算综合评分（采用akshare的加权方式）"""
        # 异常值惩罚：异常值比例，最多惩罚100%
        outlier_penalty = min(1.0, self.outliers_detected / max(1, self.total_rows))
        
        # 加权平均：30% 完整性 + 30% 一致性 + 20% 准确性 + 20% 异常值
        self.overall_score = (
            0.3 * self.completeness_score +
            0.3 * self.consistency_score +
            0.2 * self.accuracy_score +
            0.2 * (1.0 - outlier_penalty)
        )
    
    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.9