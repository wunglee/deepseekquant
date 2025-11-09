"""
资源监控组件 - 负责监控系统资源使用情况
"""

import time
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ResourceMonitorConfig:
    """资源监控配置"""
    monitor_interval: int = 10  # 监控间隔（秒）
    max_memory_mb: int = 512
    max_cpu_percent: int = 80


@dataclass
class ResourceUsage:
    """资源使用情况"""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    active_tasks: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResourceMonitor:
    """资源监控器"""

    def __init__(self, config: ResourceMonitorConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.usage = ResourceUsage()
        self.monitor_thread = None
        self.running = False
        self.lock = threading.RLock()

        # 尝试导入psutil，如果不可用则使用模拟
        try:
            import psutil
            self.process = psutil.Process()
            self.has_psutil = True
        except ImportError:
            self.process = None
            self.has_psutil = False

    def start(self):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_worker,
            name=f"{self.processor_name}_ResourceMonitor",
            daemon=True
        )
        self.monitor_thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

    def _monitor_worker(self):
        """监控工作线程"""
        while self.running:
            try:
                self._update_usage()
                time.sleep(self.config.monitor_interval)
            except Exception as e:
                # 监控错误不应该影响主流程
                time.sleep(self.config.monitor_interval)

    def _update_usage(self):
        """更新资源使用情况"""
        with self.lock:
            if self.has_psutil and self.process:
                self.usage.memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.usage.cpu_percent = self.process.cpu_percent()
                self.usage.thread_count = self.process.num_threads()
            else:
                # 模拟数据
                self.usage.memory_mb = 100.0
                self.usage.cpu_percent = 25.0
                self.usage.thread_count = threading.active_count()

            self.usage.timestamp = datetime.now().isoformat()

    def get_usage(self) -> dict:
        """获取资源使用情况"""
        with self.lock:
            return {
                'memory_mb': self.usage.memory_mb,
                'cpu_percent': self.usage.cpu_percent,
                'thread_count': self.usage.thread_count,
                'timestamp': self.usage.timestamp
            }

    def assess_health(self) -> Dict[str, Any]:
        """评估资源健康状态"""
        with self.lock:
            penalty = 0

            # 内存使用检查
            memory_ratio = self.usage.memory_mb / max(self.config.max_memory_mb, 1)
            if memory_ratio > 0.9:
                penalty += 40
            elif memory_ratio > 0.8:
                penalty += 20

            # CPU使用检查
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
        """更新配置"""
        with self.lock:
            self.config = new_config