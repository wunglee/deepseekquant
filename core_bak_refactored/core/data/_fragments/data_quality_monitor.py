from typing import Dict, Any, Tuple
from .data_quality_policy import DataQualityPolicy
from .quality_types import QualityReport, QualityFlags, Issue


class DataQualityMonitor:
    def __init__(self, policy: DataQualityPolicy):
        self.policy = policy

    def validate(self, data: Dict[str, Any]) -> Tuple[QualityReport, QualityFlags]:
        report = QualityReport()
        flags = QualityFlags(flags={"DATA_QUALITY_WARNING": False}, reasons=[])

        # 完整性：必填字段检查（配置驱动）
        completeness_ok = True
        for field in self.policy.completeness_required_fields:
            if field not in data or data[field] is None:
                report.issues.append(Issue(field=field, message="missing required field", severity="WARNING"))
                completeness_ok = False

        # 汇总
        report.summary["completeness_ok"] = completeness_ok
        report.summary["consistency_ok"] = None  # 由后续迭代依据配置执行
        report.summary["timeliness_ok"] = None   # 由后续迭代依据配置执行
        report.summary["reliability_ok"] = None  # 由后续迭代依据配置执行

        # 质量标识：有问题则置位，不设默认阈值
        if report.issues:
            flags.flags["DATA_QUALITY_WARNING"] = True
            flags.reasons.append("completeness")

        return report, flags
