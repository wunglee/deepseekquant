from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class DataQualityPolicy:
    # 完整性：必填字段
    completeness_required_fields: List[str] = field(default_factory=list)

    # 一致性：类型与范围规则（配置驱动）
    consistency_type_rules: Dict[str, str] = field(default_factory=dict)  # e.g., {"prices": "List[float]"}
    consistency_range_rules: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # e.g., {"returns": (-1.0, 1.0)}

    # 时效性：最大允许天数
    timeliness_max_age_days: int = 0

    # 可靠性：数据源白名单与错误率阈值
    reliability_source_whitelist: List[str] = field(default_factory=list)
    reliability_error_rate_threshold: float = 0.0
