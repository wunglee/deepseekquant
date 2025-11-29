"""
数据验证器（已迁移到 infrastructure/validation）

⚠️ 此模块已废弃，请使用：
    from core_bak_refactored.infrastructure.validation import validate_market_data, validate_data_list, clean_market_data

保留此文件仅为向后兼容，将在未来版本中移除。
"""
import warnings
from core_bak_refactored.infrastructure.validation import (
    validate_market_data,
    validate_data_list,
    clean_market_data
)

warnings.warn(
    "core.data.validation.data_validator 已迁移到 infrastructure.validation，"
    "请更新导入路径：from core_bak_refactored.infrastructure.validation import validate_market_data",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['validate_market_data', 'validate_data_list', 'clean_market_data']