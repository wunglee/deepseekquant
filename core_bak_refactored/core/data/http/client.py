"""
HTTP客户端设置模块（已迁移到 infrastructure/http）

⚠️ 此模块已废弃，请使用：
    from core_bak_refactored.infrastructure.http import setup_http_client, close_http_client

保留此文件仅为向后兼容，将在未来版本中移除。
"""
import warnings
from core_bak_refactored.infrastructure.http import setup_http_client, close_http_client

warnings.warn(
    "core.data.http.client 已迁移到 infrastructure.http，"
    "请更新导入路径：from core_bak_refactored.infrastructure.http import setup_http_client",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['setup_http_client', 'close_http_client']