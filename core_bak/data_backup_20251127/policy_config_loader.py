import json
from typing import Any
from .data_quality_policy import DataQualityPolicy


class PolicyConfigLoader:
    @staticmethod
    def load(path: str) -> DataQualityPolicy:
        # 仅支持JSON；YAML由后续迭代补充（避免默认值）
        with open(path, "r", encoding="utf-8") as f:
            config: Any = json.load(f)
        # 严格按配置构造，不设默认值（遵循专家配置）
        return DataQualityPolicy(
            completeness_required_fields=config["completeness"]["required_fields"],
            consistency_type_rules=config.get("consistency", {}).get("type_rules", {}),
            consistency_range_rules=config.get("consistency", {}).get("range_rules", {}),
            timeliness_max_age_days=config["timeliness"]["max_age_days"],
            reliability_source_whitelist=config["reliability"]["source_whitelist"],
            reliability_error_rate_threshold=config["reliability"]["error_rate_threshold"],
        )
