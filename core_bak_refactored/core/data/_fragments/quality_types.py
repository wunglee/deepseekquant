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
