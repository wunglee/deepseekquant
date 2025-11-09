"""
DeepSeekQuant 资源管理器模块 - 独立文件
"""

import threading
from datetime import datetime
from typing import Dict, Any
from .interfaces import IResourceManager

try:
    from core.components.resource_monitor import ResourceMonitor
except ImportError:
    from ..components.resource_monitor import ResourceMonitor


class ResourceManager(IResourceManager):
    """统一的资源管理器"""

    def __init__(self, processor_name: str, resource_monitor: ResourceMonitor):
        self.processor_name = processor_name
        self.resource_monitor = resource_monitor
        self.allocated_resources: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    def _check_resource_limits(self, resource_type: str, size: int) -> bool:
        """简单的资源限制检查"""
        try:
            usage = self.resource_monitor.get_usage()
            if resource_type == 'memory':
                max_mb = self.resource_monitor.config.max_memory_mb
                return size <= max(0, max_mb - usage['memory_mb'])
            if resource_type == 'cpu':
                max_cpu = self.resource_monitor.config.max_cpu_percent
                return size <= max(0, max_cpu - usage['cpu_percent'])
            return True
        except Exception:
            # 无法获取使用数据时默认允许，避免阻塞
            return True

    def allocate_resource(self, resource_type: str, resource_id: str, size: int, timeout: int = 30) -> bool:
        """分配资源"""
        with self.lock:
            if not self._check_resource_limits(resource_type, size):
                return False
            self.allocated_resources[resource_id] = {
                'type': resource_type,
                'size': size,
                'timestamp': datetime.now().isoformat()
            }
            return True

    def release_resource(self, resource_id: str):
        """释放资源"""
        with self.lock:
            if resource_id in self.allocated_resources:
                del self.allocated_resources[resource_id]
