"""
DeepSeekQuant 资源管理器模块 - 独立文件
"""

import threading
from datetime import datetime
from typing import Dict, Any
from .interfaces import IResourceManager

# 合并 ResourceMonitor 定义到本模块中，提供统一资源组件
from dataclasses import dataclass, field

@dataclass
class ResourceMonitorConfig:
    monitor_interval: int = 10
    max_memory_mb: int = 512
    max_cpu_percent: int = 80

@dataclass
class ResourceUsage:
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ResourceMonitor:
    def __init__(self, config: ResourceMonitorConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.usage = ResourceUsage()
        self.monitor_thread = None
        self.running = False
        self.lock = threading.RLock()
        try:
            import psutil
            self.process = psutil.Process()
            self.has_psutil = True
        except ImportError:
            self.process = None
            self.has_psutil = False

    def start(self):
        if self.running:
            return
        self.running = True
        import time
        self.monitor_thread = threading.Thread(target=self._monitor_worker, name=f"{self.processor_name}_ResourceMonitor", daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

    def _monitor_worker(self):
        import time
        while self.running:
            try:
                self._update_usage()
                time.sleep(self.config.monitor_interval)
            except Exception:
                time.sleep(self.config.monitor_interval)

    def _update_usage(self):
        with self.lock:
            if self.has_psutil and self.process is not None:
                self.usage.memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.usage.cpu_percent = self.process.cpu_percent()
                self.usage.thread_count = self.process.num_threads()
            else:
                self.usage.memory_mb = 100.0
                self.usage.cpu_percent = 25.0
                self.usage.thread_count = threading.active_count()
            self.usage.timestamp = datetime.now().isoformat()

    def get_usage(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'memory_mb': self.usage.memory_mb,
                'cpu_percent': self.usage.cpu_percent,
                'thread_count': self.usage.thread_count,
                'timestamp': self.usage.timestamp
            }

    def assess_health(self) -> Dict[str, Any]:
        with self.lock:
            penalty = 0
            memory_ratio = self.usage.memory_mb / max(self.config.max_memory_mb, 1)
            if memory_ratio > 0.9:
                penalty += 40
            elif memory_ratio > 0.8:
                penalty += 20
            cpu_ratio = self.usage.cpu_percent / max(self.config.max_cpu_percent, 1)
            if cpu_ratio > 0.9:
                penalty += 30
            elif cpu_ratio > 0.8:
                penalty += 15
            return {
                'penalty': penalty,
                'memory_ratio': memory_ratio,
                'cpu_ratio': cpu_ratio,
                'is_healthy': penalty == 0
            }

    def update_config(self, new_config: ResourceMonitorConfig):
        with self.lock:
            self.config = new_config


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
