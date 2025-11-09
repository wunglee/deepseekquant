"""
性能跟踪组件 - 负责跟踪和报告性能指标
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import threading

from common import PerformanceMetrics


@dataclass
class PerformanceConfig:
    """性能跟踪配置"""
    enable_tracking: bool = True
    max_history_size: int = 1000


class PerformanceTracker:
    """性能跟踪器"""

    def __init__(self, config: PerformanceConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.metrics = PerformanceMetrics()
        self.history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()

    def record_success(self, processing_time: float):
        """记录成功操作（统一入口）"""
        self.record_operation("process", True, processing_time)

    def record_failure(self, processing_time: float):
        """记录失败操作（统一入口）"""
        self.record_operation("process", False, processing_time)

    def record_operation(self, operation_type: str, success: bool,
                         processing_time: float, metadata: Optional[Dict[str, Any]] = None):
        """统一的操作记录方法"""
        if not self.config.enable_tracking:
            return
        with self.lock:
            self.metrics.update(success, processing_time)
            self._add_to_history(operation_type, success, processing_time, metadata or {})

    def _add_to_history(self, operation_type: str, success: bool, processing_time: float, metadata: Dict[str, Any]):
        """添加到历史记录（增强版）"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'success': success,
            'processing_time': processing_time,
            'total_operations': self.metrics.total_operations,
            'metadata': metadata
        }

        self.history.append(record)

        # 限制历史记录大小
        if len(self.history) > self.config.max_history_size:
            self.history.pop(0)

    def get_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.lock:
            return {
                'module': self.processor_name,
                'total_operations': self.metrics.total_operations,
                'successful_operations': self.metrics.successful_operations,
                'failed_operations': self.metrics.failed_operations,
                'error_rate': self.metrics.error_rate,
                'availability': self.metrics.availability,
                'avg_processing_time': self.metrics.avg_processing_time,
                'max_processing_time': self.metrics.max_processing_time,
                'min_processing_time': self.metrics.min_processing_time,
                'throughput': self.metrics.throughput,
                'last_operation_time': self.metrics.last_operation_time
            }

    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            return {
                'total_operations': self.metrics.total_operations,
                'error_rate': self.metrics.error_rate,
                'availability': self.metrics.availability,
                'throughput': self.metrics.throughput
            }

    def get_error_rate(self) -> float:
        """获取错误率"""
        with self.lock:
            return self.metrics.error_rate

    def update_config(self, new_config: PerformanceConfig):
        """更新配置"""
        with self.lock:
            self.config = new_config