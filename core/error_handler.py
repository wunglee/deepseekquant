"""
错误处理组件 - 负责错误记录和管理
"""

import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import threading

@dataclass
class ErrorHandlerConfig:
    """错误处理配置"""
    max_error_history: int = 1000
    enable_error_logging: bool = True

@dataclass
class ErrorRecord:
    """错误记录"""
    timestamp: str
    error_type: str
    error_message: str
    context: str
    stack_trace: str
    error_count: int

class ErrorHandler:
    """错误处理器"""

    def __init__(self, config: ErrorHandlerConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.error_count = 0
        self.last_error: Optional[ErrorRecord] = None
        self.error_history: List[ErrorRecord] = []
        self.lock = threading.RLock()

    # 在 error_handler.py 中修复错误计数逻辑
    # 在 error_handler.py 中修复错误计数逻辑
    def record_error(self, error: Exception, context: str):
        """记录错误 - 确保始终计数"""
        with self.lock:
            # 始终增加错误计数，即使禁用日志
            self.error_count += 1

            if not self.config.enable_error_logging:
                return  # 只计数，不记录详情

            # 记录错误详情
            error_record = ErrorRecord(
                timestamp=datetime.now().isoformat(),
                error_type=type(error).__name__,
                error_message=str(error),
                context=context,
                stack_trace=traceback.format_exc(),
                error_count=self.error_count
            )

            self.last_error = error_record
            self.error_history.append(error_record)

            # 限制错误历史大小
            if len(self.error_history) > self.config.max_error_history:
                self.error_history.pop(0)

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        with self.lock:
            return {
                'total_errors': self.error_count,
                'last_error': self._format_error_record(self.last_error) if self.last_error else None,
                'recent_errors': len([e for e in self.error_history
                                   if self._is_recent_error(e, hours=1)]),
                'error_history_size': len(self.error_history)
            }

    def _format_error_record(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """格式化错误记录"""
        return {
            'timestamp': error_record.timestamp,
            'error_type': error_record.error_type,
            'error_message': error_record.error_message,
            'context': error_record.context
        }

    def _is_recent_error(self, error_record: ErrorRecord, hours: int) -> bool:
        """检查是否为最近错误"""
        try:
            from datetime import datetime, timedelta
            error_time = datetime.fromisoformat(error_record.timestamp)
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return error_time > cutoff_time
        except:
            return False

    def update_config(self, new_config: ErrorHandlerConfig):
        """更新配置"""
        with self.lock:
            self.config = new_config