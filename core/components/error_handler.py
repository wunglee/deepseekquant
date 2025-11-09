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
    """错误记录 - 增强版本带完整上下文"""
    timestamp: str
    error_type: str
    error_message: str
    context: str
    stack_trace: str
    error_count: int
    # 新增字段：完整的错误上下文
    extra_context: Dict[str, Any] = field(default_factory=dict)
    processor_name: str = ""
    severity: str = "ERROR"  # ERROR, WARNING, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'context': self.context,
            'stack_trace': self.stack_trace,
            'error_count': self.error_count,
            'extra_context': self.extra_context,
            'processor_name': self.processor_name,
            'severity': self.severity
        }

class ErrorHandler:
    """错误处理器"""

    def __init__(self, config: ErrorHandlerConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.error_count = 0
        self.last_error: Optional[ErrorRecord] = None
        self.error_history: List[ErrorRecord] = []
        self.lock = threading.RLock()

    def record_error(self, error: Exception, context: str, 
                    extra_context: Optional[Dict[str, Any]] = None,
                    severity: str = "ERROR"):
        """记录错误 - 增强版本带完整上下文支持"""
        with self.lock:
            # 始终增加错误计数，即使禁用日志
            self.error_count += 1

            if not self.config.enable_error_logging:
                return  # 只计数，不记录详情

            # 记录错误详情 - 包含完整上下文
            error_record = ErrorRecord(
                timestamp=datetime.now().isoformat(),
                error_type=type(error).__name__,
                error_message=str(error),
                context=context,
                stack_trace=traceback.format_exc(),
                error_count=self.error_count,
                extra_context=extra_context or {},
                processor_name=self.processor_name,
                severity=severity
            )

            self.last_error = error_record
            self.error_history.append(error_record)

            # 限制错误历史大小
            if len(self.error_history) > self.config.max_error_history:
                self.error_history.pop(0)
    
    def record_error_with_context(self, error: Exception, context: str, 
                                 extra_context: Optional[Dict[str, Any]] = None):
        """记录带完整上下文的错误（别名方法，保持兼容性）"""
        self.record_error(error, context, extra_context)

    def get_error_summary(self, include_context: bool = False) -> Dict[str, Any]:
        """获取错误摘要 - 增强版本支持完整上下文"""
        with self.lock:
            summary = {
                'total_errors': self.error_count,
                'last_error': self._format_error_record(self.last_error, include_context) if self.last_error else None,
                'recent_errors': len([e for e in self.error_history
                                   if self._is_recent_error(e, hours=1)]),
                'error_history_size': len(self.error_history)
            }
            
            # 添加按严重程度分类的统计
            if include_context:
                summary['errors_by_severity'] = self._count_by_severity()
                summary['errors_by_type'] = self._count_by_type()
            
            return summary
    
    def _count_by_severity(self) -> Dict[str, int]:
        """按严重程度统计错误"""
        counts = {'ERROR': 0, 'WARNING': 0, 'CRITICAL': 0}
        for error in self.error_history:
            severity = getattr(error, 'severity', 'ERROR')
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _count_by_type(self) -> Dict[str, int]:
        """按错误类型统计"""
        counts = {}
        for error in self.error_history:
            error_type = error.error_type
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts

    def _format_error_record(self, error_record: ErrorRecord, include_context: bool = False) -> Dict[str, Any]:
        """格式化错误记录 - 支持完整上下文输出"""
        formatted = {
            'timestamp': error_record.timestamp,
            'error_type': error_record.error_type,
            'error_message': error_record.error_message,
            'context': error_record.context,
            'severity': getattr(error_record, 'severity', 'ERROR')
        }
        
        if include_context:
            formatted['extra_context'] = getattr(error_record, 'extra_context', {})
            formatted['processor_name'] = getattr(error_record, 'processor_name', '')
            formatted['stack_trace'] = error_record.stack_trace
        
        return formatted

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