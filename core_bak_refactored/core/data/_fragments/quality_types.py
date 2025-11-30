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
    overall_score: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    continuity: float = 0.0
    reasonableness: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    outliers_detected: int = 0
    total_rows: int = 0
    missing_values: int = 0
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.9
