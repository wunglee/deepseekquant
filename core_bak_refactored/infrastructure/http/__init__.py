"""
HTTP客户端基础设施模块

职责：
- 提供异步和同步HTTP客户端配置
- 管理连接池和超时设置
- 支持重试策略

从 core/data/http 迁移而来
"""

from .client import setup_http_client, close_http_client

__all__ = [
    'setup_http_client',
    'close_http_client',
]
