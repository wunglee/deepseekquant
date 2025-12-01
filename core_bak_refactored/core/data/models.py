"""数据模型模块（已废弃）

.. deprecated:: 2025-11-29
    请直接从 `core_bak_refactored.core.share` 导入这些模型和枚举
    
    迁移示例：
    ```python
    # 旧代码
    from core_bak_refactored.core.data.models import MarketData, DataSourceType, DataFrequency
    
    # 新代码
    from core_bak_refactored.core.share import MarketData, DataSourceType, DataFrequency
    ```
"""

# 为了向后兼容，从share模块重新导出（但应直接使用share模块）
from core_bak_refactored.core.share import MarketData, DataSourceType, DataFrequency

__all__ = [
    'MarketData',
    'DataSourceType',
    'DataFrequency',
]
