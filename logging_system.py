"""
DeepSeekQuant 日志系统
提供统一的日志记录、格式化和管理功能
支持多日志级别、文件轮转和审计日志
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os
import json
import sys
from datetime import datetime
import threading
from enum import Enum
import gzip
import hashlib
import time
import queue
from dataclasses import dataclass, asdict, field

from common import DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT, DEFAULT_LOG_DATE_FORMAT, MAX_LOG_FILE_SIZE, BACKUP_LOG_COUNT, \
    DEFAULT_ENCODING


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogDestination(Enum):
    """日志输出目的地枚举"""
    CONSOLE = "console"
    FILE = "file"
    SYSLOG = "syslog"
    HTTP = "http"
    DATABASE = "database"
    ALL = "all"


@dataclass
class LogEntry:
    """日志条目数据类"""
    timestamp: str
    level: LogLevel
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    process_id: int
    thread_id: int
    thread_name: str
    exception_info: Optional[str] = None
    stack_trace: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'level': self.level.value,
            'logger_name': self.logger_name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'process_id': self.process_id,
            'thread_id': self.thread_id,
            'thread_name': self.thread_name,
            'exception_info': self.exception_info,
            'stack_trace': self.stack_trace,
            'extra_data': self.extra_data,
            'correlation_id': self.correlation_id,
            'request_id': self.request_id,
            'session_id': self.session_id,
            'user_id': self.user_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """从字典创建日志条目"""
        return cls(
            timestamp=data['timestamp'],
            level=LogLevel(data['level']),
            logger_name=data['logger_name'],
            message=data['message'],
            module=data['module'],
            function=data['function'],
            line_number=data['line_number'],
            process_id=data['process_id'],
            thread_id=data['thread_id'],
            thread_name=data['thread_name'],
            exception_info=data.get('exception_info'),
            stack_trace=data.get('stack_trace'),
            extra_data=data.get('extra_data', {}),
            correlation_id=data.get('correlation_id'),
            request_id=data.get('request_id'),
            session_id=data.get('session_id'),
            user_id=data.get('user_id')
        )


class LogFormat(Enum):
    """日志格式枚举"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    XML = "xml"


class LogRotationStrategy(Enum):
    """日志轮转策略枚举"""
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    SIZE_AND_TIME = "size_and_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class LogConfig:
    """日志配置数据类"""
    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.TEXT
    destinations: List[LogDestination] = field(default_factory=lambda: [LogDestination.CONSOLE, LogDestination.FILE])
    file_path: str = "logs/deepseekquant.log"
    max_file_size_mb: int = 100
    backup_count: int = 10
    rotation_strategy: LogRotationStrategy = LogRotationStrategy.SIZE_BASED
    encoding: str = "utf-8"
    console_output: bool = True
    json_format: bool = False
    include_timestamp: bool = True
    include_module: bool = True
    include_function: bool = True
    include_line_number: bool = True
    include_thread: bool = True
    include_process: bool = True
    audit_log_enabled: bool = True
    audit_log_path: str = "logs/audit.log"
    performance_log_enabled: bool = True
    performance_log_path: str = "logs/performance.log"
    error_log_enabled: bool = True
    error_log_path: str = "logs/error.log"
    syslog_host: Optional[str] = None
    syslog_port: int = 514
    syslog_facility: str = "local0"
    http_endpoint: Optional[str] = None
    http_headers: Dict[str, str] = field(default_factory=dict)
    database_connection: Optional[str] = None
    buffer_size: int = 1000
    flush_interval: int = 30  # seconds
    compression_enabled: bool = True
    retention_days: int = 30

    def __post_init__(self):
        """配置验证"""
        self._validate_config()

    def _validate_config(self):
        """验证配置的合理性"""
        # 基本数值验证
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if self.backup_count < 0:
            raise ValueError("backup_count cannot be negative")
        if self.retention_days < 0:
            raise ValueError("retention_days cannot be negative")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        if self.syslog_port <= 0 or self.syslog_port > 65535:
            raise ValueError("syslog_port must be between 1 and 65535")
        
        # 增强验证：文件路径
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        
        # 增强验证：destinations 类型
        if not isinstance(self.destinations, list):
            raise ValueError("destinations must be a list")
        if any(not isinstance(dest, LogDestination) for dest in self.destinations):
            raise ValueError("destinations must contain only LogDestination instances")
        
        # 文件路径可写性验证
        if LogDestination.FILE in self.destinations:
            try:
                log_dir = os.path.dirname(self.file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                # 测试文件可写性
                with open(self.file_path, 'a') as f:
                    pass  # 只是测试打开
            except (OSError, IOError) as e:
                raise ValueError(f"Invalid file path {self.file_path}: {e}")
        
        # HTTP端点验证
        if LogDestination.HTTP in self.destinations and self.http_endpoint:
            from urllib.parse import urlparse
            try:
                result = urlparse(self.http_endpoint)
                if not all([result.scheme, result.netloc]):
                    raise ValueError("HTTP endpoint must have scheme and netloc")
                if result.scheme not in ('http', 'https'):
                    raise ValueError("HTTP endpoint scheme must be http or https")
            except Exception as e:
                raise ValueError(f"Invalid HTTP endpoint: {e}")
        
        # 数据库连接验证
        if LogDestination.DATABASE in self.destinations and self.database_connection:
            if not isinstance(self.database_connection, str):
                raise ValueError("database_connection must be a string")
            if not self.database_connection.startswith(('sqlite:///', 'postgresql://', 'mysql://', 'mongodb://')):
                raise ValueError("Invalid database connection string format (supported: sqlite, postgresql, mysql, mongodb)")


class DeepSeekQuantFormatter(logging.Formatter):
    """DeepSeekQuant 自定义日志格式化器"""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

        if config.format == LogFormat.JSON:
            self.formatter = self._format_json
        else:
            # 文本格式
            fmt_parts = []
            if config.include_timestamp:
                fmt_parts.append('%(asctime)s')
            fmt_parts.append('%(levelname)s')
            if config.include_module:
                fmt_parts.append('%(name)s')
            if config.include_function:
                fmt_parts.append('%(funcName)s')
            if config.include_line_number:
                fmt_parts.append('%(lineno)d')
            if config.include_thread:
                fmt_parts.append('%(threadName)s')
            fmt_parts.append('%(message)s')

            fmt = ' - '.join(fmt_parts)
            super().__init__(fmt=fmt, datefmt=DEFAULT_LOG_DATE_FORMAT)

    def _format_json(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': record.threadName if hasattr(record, 'threadName') else str(record.thread)
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加额外数据
        if hasattr(record, 'extra_data'):
            log_entry['extra_data'] = record.extra_data  # type: ignore[attr-defined]

        return json.dumps(log_entry, ensure_ascii=False)

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        try:
            if self.config.format == LogFormat.JSON:
                return self._format_json(record)
            else:
                return super().format(record)
        except Exception as e:
            # 格式化失败时的备选方案
            return f"LOG_FORMAT_ERROR - {datetime.now().isoformat()} - {record.levelname} - {record.getMessage()}"


class AuditLogFilter(logging.Filter):
    """审计日志过滤器"""

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤审计日志"""
        return (hasattr(record, 'is_audit') and record.is_audit) or record.levelno >= logging.WARNING  # type: ignore[attr-defined]


class PerformanceLogFilter(logging.Filter):
    """性能日志过滤器"""

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤性能日志"""
        return (hasattr(record, 'is_performance') and record.is_performance) or 'performance' in record.name.lower()  # type: ignore[attr-defined]


class ErrorLogFilter(logging.Filter):
    """错误日志过滤器"""

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤错误日志"""
        return record.levelno >= logging.ERROR


class ThreadSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """线程安全的日志文件轮转处理器"""

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.lock = threading.RLock()  # type: ignore[assignment]

    def emit(self, record):
        """线程安全地输出日志记录"""
        with self.lock:  # type: ignore[attr-defined]
            super().emit(record)


class CompressedRotatingFileHandler(ThreadSafeRotatingFileHandler):
    """支持压缩的日志文件轮转处理器"""

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False, compression=True):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compression = compression

    def doRollover(self):
        """执行日志转转并压缩旧文件"""
        with self.lock:  # type: ignore[attr-defined]
            super().doRollover()

            if self.compression and self.backupCount > 0:
                # 压缩旧的日志文件
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                    sfn_compressed = f"{sfn}.gz"

                    if os.path.exists(sfn) and not os.path.exists(sfn_compressed):
                        try:
                            with open(sfn, 'rb') as f_in:
                                with gzip.open(sfn_compressed, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            os.remove(sfn)
                        except Exception as e:
                            # 压缩失败不影响主流程
                            pass


class DatabaseLogHandler(logging.Handler):
    """数据库日志处理器"""

    def __init__(self, connection_string, table_name='system_logs'):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self.buffer = []
        self.buffer_lock = threading.RLock()
        self.flush_timer = None

    def emit(self, record):
        """将日志记录到数据库"""
        try:
            log_entry = self._format_record(record)

            with self.buffer_lock:
                self.buffer.append(log_entry)

                # 缓冲达到一定大小或定时刷新
                if len(self.buffer) >= 100:
                    self._flush_buffer()
                elif not self.flush_timer:
                    self._start_flush_timer()

        except Exception as e:
            # 数据库记录失败，回退到文件记录
            fallback_handler = logging.FileHandler('logs/db_fallback.log')
            fallback_handler.emit(record)

    def _format_record(self, record):
        """格式化日志记录为数据库记录"""
        return {
            'timestamp': datetime.fromtimestamp(record.created),
            'level': record.levelname,
            'logger_name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'exception_info': self.formatException(record.exc_info) if record.exc_info else None,  # type: ignore[attr-defined]
            'extra_data': getattr(record, 'extra_data', {})
        }

    def _flush_buffer(self):
        """刷新缓冲区到数据库"""
        if not self.buffer:
            return

        try:
            # 这里实现数据库插入逻辑
            # 实际实现中会使用SQLAlchemy或其他ORM
            pass
        except Exception as e:
            # 记录错误但不影响主流程
            pass
        finally:
            with self.buffer_lock:
                self.buffer.clear()

            if self.flush_timer:
                self.flush_timer.cancel()
                self.flush_timer = None

    def _start_flush_timer(self):
        """启动定时刷新计时器"""

        def flush():
            with self.buffer_lock:
                if self.buffer:
                    self._flush_buffer()

        self.flush_timer = threading.Timer(30.0, flush)  # 30秒后刷新
        self.flush_timer.daemon = True
        self.flush_timer.start()


class HttpLogHandler(logging.Handler):
    """HTTP日志处理器"""

    def __init__(self, endpoint, headers=None, timeout=10):
        super().__init__()
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.session = None
        self.buffer = []
        self.buffer_lock = threading.RLock()

    def emit(self, record):
        """通过HTTP发送日志记录"""
        try:
            log_data = self._format_record(record)

            with self.buffer_lock:
                self.buffer.append(log_data)

                # 批量发送以提高性能
                if len(self.buffer) >= 50:
                    self._send_batch()

        except Exception as e:
            # HTTP发送失败，回退到文件记录
            fallback_handler = logging.FileHandler('logs/http_fallback.log')
            fallback_handler.emit(record)

    def _format_record(self, record):
        """格式化日志记录为HTTP请求数据"""
        return {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'extra_data': getattr(record, 'extra_data', {})
        }

    def _send_batch(self):
        """批量发送日志数据"""
        if not self.buffer:
            return

        try:
            # 这里实现HTTP批量发送逻辑
            # 实际实现中会使用requests库
            pass
        except Exception as e:
            # 记录错误但不影响主流程
            pass
        finally:
            with self.buffer_lock:
                self.buffer.clear()


class AsyncLogHandler(logging.Handler):
    """异步日志处理器"""
    
    def __init__(self, base_handler: logging.Handler, max_queue_size: int = 10000):
        super().__init__()
        self.base_handler = base_handler
        self.max_queue_size = max_queue_size
        self.log_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._start_worker()
    
    def _start_worker(self):
        """启动工作线程"""
        self.worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.worker_thread.start()
    
    def _process_logs(self):
        """处理日志队列"""
        while not self._shutdown:
            try:
                # 非阻塞获取，避免无法退出的问题
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # 关闭信号
                    break
                self.base_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # 避免工作线程因异常退出
                try:
                    fallback_handler = logging.StreamHandler(sys.stderr)
                    fallback_handler.emit(logging.makeLogRecord({
                        'msg': f'Async log handler error: {e}',
                        'levelname': 'ERROR',
                        'levelno': logging.ERROR,
                        'pathname': '',
                        'filename': '',
                        'module': '',
                        'lineno': 0,
                        'funcName': '',
                        'created': time.time(),
                        'msecs': 0,
                        'relativeCreated': 0,
                        'thread': 0,
                        'threadName': '',
                        'processName': '',
                        'process': 0
                    }))
                except:
                    pass
    
    def emit(self, record):
        """异步发送日志记录"""
        if self._shutdown:
            return
        
        try:
            # 非阻塞放入队列，如果队列满则丢弃
            self.log_queue.put_nowait(record)
        except queue.Full:
            # 队列满时的降级处理
            try:
                fallback_handler = logging.StreamHandler(sys.stderr)
                fallback_handler.emit(logging.makeLogRecord({
                    'msg': 'Log queue full, message dropped',
                    'levelname': 'WARNING',
                    'levelno': logging.WARNING,
                    'pathname': '',
                    'filename': '',
                    'module': '',
                    'lineno': 0,
                    'funcName': '',
                    'created': time.time(),
                    'msecs': 0,
                    'relativeCreated': 0,
                    'thread': 0,
                    'threadName': '',
                    'processName': '',
                    'process': 0
                }))
            except:
                pass
    
    def flush(self):
        """刷新队列"""
        self.log_queue.join()  # 等待所有任务完成
        self.base_handler.flush()
    
    def close(self):
        """关闭处理器"""
        self._shutdown = True
        # 发送关闭信号
        try:
            self.log_queue.put_nowait(None)
        except:
            pass
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        self.base_handler.close()
        super().close()


class LoggingSystem:
    """DeepSeekQuant 日志系统 - 完整生产实现"""

    def __init__(self, config: LogConfig):
        """
        初始化日志系统

        Args:
            config: 日志配置
        """
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.audit_logger: Optional[logging.Logger] = None
        self.performance_logger: Optional[logging.Logger] = None
        self.error_logger: Optional[logging.Logger] = None

        self._lock = threading.RLock()
        self._initialized = False
        self._shutdown = False

        # 统计信息
        self.stats = {
            'total_logs': 0,
            'debug_logs': 0,
            'info_logs': 0,
            'warning_logs': 0,
            'error_logs': 0,
            'critical_logs': 0,
            'audit_logs': 0,
            'performance_logs': 0,
            'last_flush': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat()
        }

        # 初始化日志系统
        self._initialize()

    def _initialize(self):
        """初始化日志系统"""
        with self._lock:
            if self._initialized:
                return

            try:
                # 创建日志目录
                self._ensure_log_directories()

                # 配置根日志记录器
                self._configure_root_logger()

                # 创建专用日志记录器
                self._create_specialized_loggers()

                # 设置全局异常处理
                self._setup_exception_handling()

                self._initialized = True

                self.get_logger(__name__).info("日志系统初始化完成")

            except Exception as e:
                # 日志系统初始化失败时的备选方案
                self._setup_fallback_logging()
                raise RuntimeError(f"日志系统初始化失败: {e}")

    def _ensure_log_directories(self):
        """确保日志目录存在"""
        directories = [
            os.path.dirname(self.config.file_path),
            os.path.dirname(self.config.audit_log_path),
            os.path.dirname(self.config.performance_log_path),
            os.path.dirname(self.config.error_log_path)
        ]

        for directory in directories:
            if directory:  # 避免空路径
                os.makedirs(directory, exist_ok=True)

    def _configure_root_logger(self):
        """配置根日志记录器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.level.value)

        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 根据配置添加处理器
        if LogDestination.CONSOLE in self.config.destinations or LogDestination.ALL in self.config.destinations:
            self._add_console_handler(root_logger)

        if LogDestination.FILE in self.config.destinations or LogDestination.ALL in self.config.destinations:
            self._add_file_handler(root_logger)

        if LogDestination.SYSLOG in self.config.destinations:
            self._add_syslog_handler(root_logger)

        if LogDestination.HTTP in self.config.destinations:
            self._add_http_handler(root_logger)

        if LogDestination.DATABASE in self.config.destinations:
            self._add_database_handler(root_logger)

    def _add_console_handler(self, logger: logging.Logger):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.level.value)
        console_handler.setFormatter(DeepSeekQuantFormatter(self.config))
        logger.addHandler(console_handler)
        self.handlers['console'] = console_handler

    def _add_file_handler(self, logger: logging.Logger):
        """添加文件处理器"""
        max_bytes = self.config.max_file_size_mb * 1024 * 1024

        file_handler = CompressedRotatingFileHandler(
            filename=self.config.file_path,
            maxBytes=max_bytes,
            backupCount=self.config.backup_count,
            encoding=self.config.encoding,
            compression=self.config.compression_enabled
        )
        file_handler.setLevel(self.config.level.value)
        file_handler.setFormatter(DeepSeekQuantFormatter(self.config))
        logger.addHandler(file_handler)
        self.handlers['file'] = file_handler

    def _add_syslog_handler(self, logger: logging.Logger):
        """添加系统日志处理器"""
        if self.config.syslog_host:
            try:
                syslog_handler = logging.handlers.SysLogHandler(
                    address=(self.config.syslog_host, self.config.syslog_port),
                    facility=self.config.syslog_facility
                )
                syslog_handler.setLevel(self.config.level.value)
                logger.addHandler(syslog_handler)
                self.handlers['syslog'] = syslog_handler
            except Exception as e:
                self._log_setup_error("syslog", e)

    def _add_http_handler(self, logger: logging.Logger):
        """添加HTTP处理器"""
        if self.config.http_endpoint:
            try:
                http_handler = HttpLogHandler(
                    endpoint=self.config.http_endpoint,
                    headers=self.config.http_headers,
                    timeout=10
                )
                http_handler.setLevel(self.config.level.value)
                logger.addHandler(http_handler)
                self.handlers['http'] = http_handler
            except Exception as e:
                self._log_setup_error("http", e)

    def _add_database_handler(self, logger: logging.Logger):
        """添加数据库处理器"""
        if self.config.database_connection:
            try:
                db_handler = DatabaseLogHandler(self.config.database_connection)
                db_handler.setLevel(self.config.level.value)
                logger.addHandler(db_handler)
                self.handlers['database'] = db_handler
            except Exception as e:
                self._log_setup_error("database", e)

    def _log_setup_error(self, handler_type: str, error: Exception):
        """记录处理器设置错误"""
        # 使用基本的日志记录作为备选
        basic_logger = logging.getLogger('LoggingSystem')
        basic_logger.error(f"{handler_type} handler setup failed: {error}")

    def _create_specialized_loggers(self):
        """创建专用日志记录器"""
        # 审计日志记录器
        if self.config.audit_log_enabled:
            self.audit_logger = self._create_audit_logger()

        # 性能日志记录器
        if self.config.performance_log_enabled:
            self.performance_logger = self._create_performance_logger()

        # 错误日志记录器
        if self.config.error_log_enabled:
            self.error_logger = self._create_error_logger()

    def _create_audit_logger(self) -> logging.Logger:
        """创建审计日志记录器"""
        audit_logger = logging.getLogger('audit')
        audit_logger.propagate = False
        audit_logger.setLevel(logging.INFO)

        # 审计日志处理器
        audit_handler = CompressedRotatingFileHandler(
            filename=self.config.audit_log_path,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
            encoding=self.config.encoding
        )
        audit_handler.setFormatter(DeepSeekQuantFormatter(self.config))
        audit_handler.addFilter(AuditLogFilter())
        audit_logger.addHandler(audit_handler)

        return audit_logger

    def _create_performance_logger(self) -> logging.Logger:
        """创建性能日志记录器"""
        performance_logger = logging.getLogger('performance')
        performance_logger.propagate = False
        performance_logger.setLevel(logging.INFO)

        # 性能日志处理器
        performance_handler = CompressedRotatingFileHandler(
            filename=self.config.performance_log_path,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
            encoding=self.config.encoding
        )
        performance_handler.setFormatter(DeepSeekQuantFormatter(self.config))
        performance_handler.addFilter(PerformanceLogFilter())
        performance_logger.addHandler(performance_handler)

        return performance_logger

    def _create_error_logger(self) -> logging.Logger:
        """创建错误日志记录器"""
        error_logger = logging.getLogger('error')
        error_logger.propagate = False
        error_logger.setLevel(logging.ERROR)

        # 错误日志处理器
        error_handler = CompressedRotatingFileHandler(
            filename=self.config.error_log_path,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
            encoding=self.config.encoding
        )
        error_handler.setFormatter(DeepSeekQuantFormatter(self.config))
        error_handler.addFilter(ErrorLogFilter())
        error_logger.addHandler(error_handler)

        return error_logger

    def _setup_exception_handling(self):
        """设置全局异常处理"""

        def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # 不处理键盘中断
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            try:
                # 尝试使用日志系统记录异常
                logger = self.get_logger('UncaughtException')
                logger.critical(
                    "未捕获的异常",
                    exc_info=(exc_type, exc_value, exc_traceback)
                )
            except Exception:
                # 如果日志系统失败，使用标准错误输出
                import traceback
                print("CRITICAL: Uncaught exception (logging failed):", file=sys.stderr)
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

        sys.excepthook = handle_uncaught_exception

    def _setup_fallback_logging(self):
        """设置备选日志记录（当主系统失败时）"""
        # 基本的控制台日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_logger(self, name: str) -> logging.Logger:
        """获取日志记录器，提供安全的错误处理"""
        try:
            with self._lock:
                if self._shutdown:
                    # 返回一个基本的日志记录器而不是抛出异常
                    fallback_logger = logging.getLogger(name)
                    if not fallback_logger.handlers:
                        fallback_logger.addHandler(logging.NullHandler())
                    return fallback_logger

                if name not in self.loggers:
                    logger = logging.getLogger(name)
                    self.loggers[name] = logger

                return self.loggers[name]
        except Exception as e:
            # 确保即使在异常情况下也能返回可用的日志记录器
            fallback_logger = logging.getLogger(name)
            if not fallback_logger.handlers:
                fallback_logger.addHandler(logging.NullHandler())
            return fallback_logger

    def get_audit_logger(self) -> logging.Logger:
        """获取审计日志记录器"""
        if not self.audit_logger:
            raise RuntimeError("审计日志记录器未初始化")
        return self.audit_logger

    def get_performance_logger(self) -> logging.Logger:
        """获取性能日志记录器"""
        if not self.performance_logger:
            raise RuntimeError("性能日志记录器未初始化")
        return self.performance_logger

    def get_error_logger(self) -> logging.Logger:
        """获取错误日志记录器"""
        if not self.error_logger:
            raise RuntimeError("错误日志记录器未初始化")
        return self.error_logger

    def log_audit(self, action: str, user: str, resource: str,
                  status: str, details: Optional[Dict[str, Any]] = None):
        """记录审计日志"""
        if not self.audit_logger:
            return

        try:
            audit_record = logging.LogRecord(
                name='audit',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=f"用户 {user} 执行操作 {action} 于资源 {resource} 状态 {status}",
                args=(),
                exc_info=None
            )

            # 添加审计特定属性
            audit_record.action = action
            audit_record.user = user
            audit_record.resource = resource
            audit_record.status = status
            audit_record.details = details or {}
            audit_record.timestamp = datetime.now().isoformat()
            audit_record.is_audit = True

            self.audit_logger.handle(audit_record)
            self._update_stats('audit_logs')

        except Exception as e:
            # 审计日志记录失败不影响主流程
            self._log_internal_error("audit_log", e)

    def log_performance(self, operation: str, duration: float,
                        success: bool, metrics: Optional[Dict[str, Any]] = None):
        """记录性能日志"""
        if not self.performance_logger:
            return

        try:
            level = logging.INFO if success else logging.WARNING
            status = "成功" if success else "失败"

            performance_record = logging.LogRecord(
                name='performance',
                level=level,
                pathname='',
                lineno=0,
                msg=f"操作 {operation} {status}，耗时 {duration:.3f}秒",
                args=(),
                exc_info=None
            )

            # 添加性能特定属性
            performance_record.operation = operation
            performance_record.duration = duration
            performance_record.success = success
            performance_record.metrics = metrics or {}
            performance_record.timestamp = datetime.now().isoformat()
            performance_record.is_performance = True

            self.performance_logger.handle(performance_record)
            self._update_stats('performance_logs')

        except Exception as e:
            self._log_internal_error("performance_log", e)

    def log_error(self, error_type: str, error_message: str,
                  context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """记录错误日志"""
        if not self.error_logger:
            return

        try:
            error_record = logging.LogRecord(
                name='error',
                level=logging.ERROR,
                pathname='',
                lineno=0,
                msg=f"{error_type}: {error_message}",
                args=(),
                exc_info=sys.exc_info() if exception else None  # type: ignore[arg-type]
            )

            # 添加错误特定属性
            error_record.error_type = error_type
            error_record.error_message = error_message
            error_record.context = context or {}
            error_record.timestamp = datetime.now().isoformat()

            self.error_logger.handle(error_record)
            self._update_stats('error_logs')

        except Exception as e:
            self._log_internal_error("error_log", e)

    def _update_stats(self, stat_type: str):
        """更新统计信息"""
        with self._lock:
            if stat_type in self.stats:
                self.stats[stat_type] += 1
            self.stats['total_logs'] += 1

    def _log_internal_error(self, error_type: str, error: Exception):
        """记录内部错误（使用基本日志记录）"""
        try:
            basic_logger = logging.getLogger('LoggingSystemInternal')
            basic_logger.error(f"日志系统内部错误 [{error_type}]: {error}")
        except:
            # 如果基本日志记录也失败，输出到标准错误
            print(f"CRITICAL: Logging system internal error [{error_type}]: {error}",
                  file=sys.stderr)

    def set_level(self, level: LogLevel):
        """设置日志级别"""
        with self._lock:
            self.config.level = level

            # 更新所有处理器的级别
            for handler in self.handlers.values():
                handler.setLevel(level.value)

            # 更新根日志记录器级别
            logging.getLogger().setLevel(level.value)

            self.get_logger(__name__).info(f"日志级别已设置为: {level.value}")

    def add_destination(self, destination: LogDestination, **kwargs):
        """添加日志输出目的地"""
        with self._lock:
            if destination in self.config.destinations:
                return  # 已存在

            self.config.destinations.append(destination)
            root_logger = logging.getLogger()

            try:
                if destination == LogDestination.CONSOLE:
                    self._add_console_handler(root_logger)
                elif destination == LogDestination.FILE:
                    if 'file_path' in kwargs:
                        self.config.file_path = kwargs['file_path']
                    self._add_file_handler(root_logger)
                elif destination == LogDestination.SYSLOG:
                    if 'host' in kwargs:
                        self.config.syslog_host = kwargs['host']
                    if 'port' in kwargs:
                        self.config.syslog_port = kwargs['port']
                    self._add_syslog_handler(root_logger)
                elif destination == LogDestination.HTTP:
                    if 'endpoint' in kwargs:
                        self.config.http_endpoint = kwargs['endpoint']
                    self._add_http_handler(root_logger)
                elif destination == LogDestination.DATABASE:
                    if 'connection' in kwargs:
                        self.config.database_connection = kwargs['connection']
                    self._add_database_handler(root_logger)

                self.get_logger(__name__).info(f"已添加日志目的地: {destination.value}")

            except Exception as e:
                self.get_logger(__name__).error(f"添加日志目的地失败 {destination.value}: {e}")

    def remove_destination(self, destination: LogDestination):
        """移除日志输出目的地"""
        with self._lock:
            if destination not in self.config.destinations:
                return

            self.config.destinations.remove(destination)
            root_logger = logging.getLogger()

            # 移除对应的处理器
            handler_key = destination.value
            if handler_key in self.handlers:
                handler = self.handlers[handler_key]
                root_logger.removeHandler(handler)
                handler.close()
                del self.handlers[handler_key]

            self.get_logger(__name__).info(f"已移除日志目的地: {destination.value}")

    def flush(self):
        """刷新所有日志处理器"""
        with self._lock:
            for handler in self.handlers.values():
                try:
                    handler.flush()
                except Exception as e:
                    self._log_internal_error("flush", e)

            self.stats['last_flush'] = datetime.now().isoformat()

    def rotate_logs(self):
        """轮转日志文件"""
        with self._lock:
            for handler_name, handler in self.handlers.items():
                if hasattr(handler, 'doRollover'):  # type: ignore[attr-defined]
                    try:
                        handler.doRollover()  # type: ignore[attr-defined]
                        self.get_logger(__name__).info(f"已轮转日志文件: {handler_name}")
                    except Exception as e:
                        self.get_logger(__name__).error(f"日志轮转失败 {handler_name}: {e}")

    def compress_old_logs(self):
        """压缩旧日志文件"""
        with self._lock:
            log_dir = os.path.dirname(self.config.file_path)
            if not os.path.exists(log_dir):
                return

            # 查找未压缩的日志文件
            for filename in os.listdir(log_dir):
                if (filename.endswith('.log') and
                        not filename.endswith('.gz') and
                        filename != os.path.basename(self.config.file_path)):

                    filepath = os.path.join(log_dir, filename)
                    compressed_path = f"{filepath}.gz"

                    try:
                        with open(filepath, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                f_out.writelines(f_in)
                        os.remove(filepath)
                        self.get_logger(__name__).debug(f"已压缩日志文件: {filename}")
                    except Exception as e:
                        self.get_logger(__name__).error(f"日志压缩失败 {filename}: {e}")

    def cleanup_old_logs(self):
        """清理过期日志文件"""
        with self._lock:
            if self.config.retention_days <= 0:
                return

            cutoff_time = datetime.now().timestamp() - (self.config.retention_days * 24 * 3600)
            log_dirs = [
                os.path.dirname(self.config.file_path),
                os.path.dirname(self.config.audit_log_path),
                os.path.dirname(self.config.performance_log_path),
                os.path.dirname(self.config.error_log_path)
            ]

            for log_dir in set(log_dirs):  # 去重
                if not os.path.exists(log_dir):
                    continue

                for filename in os.listdir(log_dir):
                    filepath = os.path.join(log_dir, filename)
                    if (os.path.isfile(filepath) and
                            (filename.endswith('.log') or filename.endswith('.gz'))):

                        file_mtime = os.path.getmtime(filepath)
                        if file_mtime < cutoff_time:
                            try:
                                os.remove(filepath)
                                self.get_logger(__name__).info(f"已清理过期日志: {filename}")
                            except Exception as e:
                                self.get_logger(__name__).error(f"日志清理失败 {filename}: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        with self._lock:
            # 检查处理器状态
            working_handlers = 0
            failed_handlers = []
            
            for name, handler in self.handlers.items():
                try:
                    # 简单测试：尝试格式化一个测试记录
                    test_record = logging.LogRecord(
                        name='health_check',
                        level=logging.INFO,
                        pathname='',
                        lineno=0,
                        msg='health check',
                        args=(),
                        exc_info=None
                    )
                    handler.format(test_record)
                    working_handlers += 1
                except Exception as e:
                    failed_handlers.append({'handler': name, 'error': str(e)})
            
            # 计算运行时间
            start_time = datetime.fromisoformat(self.stats.get('start_time', datetime.now().isoformat()))
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'healthy' if self._initialized and not self._shutdown else 'unhealthy',
                'initialized': self._initialized,
                'shutdown': self._shutdown,
                'uptime_seconds': uptime_seconds,
                'handlers': {
                    'total': len(self.handlers),
                    'working': working_handlers,
                    'failed': len(failed_handlers),
                    'failed_details': failed_handlers
                },
                'loggers_count': len(self.loggers),
                'last_flush': self.stats.get('last_flush'),
                'total_logs': self.stats.get('total_logs', 0),
                'error_rate': self.stats.get('error_rate', 0.0)
            }

    def _get_log_file_sizes(self) -> Dict[str, int]:
        file_sizes = {}
        log_files = [
            self.config.file_path,
            self.config.audit_log_path,
            self.config.performance_log_path,
            self.config.error_log_path
        ]

        for filepath in log_files:
            if os.path.exists(filepath):
                try:
                    file_sizes[os.path.basename(filepath)] = os.path.getsize(filepath)
                except OSError:
                    file_sizes[os.path.basename(filepath)] = 0
            else:
                file_sizes[os.path.basename(filepath)] = 0

        return file_sizes

    def reload_config(self, new_config: LogConfig):
        """
        热重载配置（部分配置支持动态更新）
        
        Args:
            new_config: 新的日志配置
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("日志系统已关闭，无法重载配置")
            
            # 记录配置变更
            self.get_logger(__name__).info("开始热重载日志配置")
            
            # 更新日志级别（可以安全热更新）
            if new_config.level != self.config.level:
                old_level = self.config.level
                self.set_level(new_config.level)
                self.get_logger(__name__).info(
                    f"日志级别已更新: {old_level.value} -> {new_config.level.value}"
                )
            
            # 更新配置对象
            self.config = new_config
            
            self.get_logger(__name__).info("配置热重载完成")

    def _record_performance(self, operation: str, start_time: float):
        """
        记录内部操作性能
        
        Args:
            operation: 操作名称
            start_time: 开始时间
        """
        duration = time.time() - start_time
        
        # 超过1秒的操作记录警告
        if duration > 1.0:
            try:
                self.get_logger(__name__).warning(
                    f"操作 {operation} 耗时 {duration:.3f}s，可能存在性能问题"
                )
            except Exception:
                # 避免性能监控本身导致问题
                pass

    def get_stats(self) -> Dict[str, Any]:
        """获取日志系统统计信息"""
        with self._lock:
            stats = self.stats.copy()
            stats['loggers_count'] = len(self.loggers)
            stats['handlers_count'] = len(self.handlers)
            stats['initialized'] = self._initialized
            stats['shutdown'] = self._shutdown
            stats['current_time'] = datetime.now().isoformat()

            # 添加文件大小信息
            stats['file_sizes'] = self._get_log_file_sizes()

            return stats

    def get_log_entries(self, level: Optional[LogLevel] = None,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        logger_name: Optional[str] = None,
                        limit: int = 1000) -> List[LogEntry]:
        """获取日志条目（需要实现日志查询功能）"""
        # 注意：这是一个高级功能，需要额外的日志存储和查询实现
        # 这里提供基本框架，实际实现可能需要数据库支持

        entries = []

        # 这里应该实现从日志文件或数据库中查询日志条目
        # 目前返回空列表作为占位符
        return entries

    def export_logs(self, filepath: str,
                    level: Optional[LogLevel] = None,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None) -> bool:
        """导出日志到文件"""
        try:
            entries = self.get_log_entries(level, start_time, end_time)

            with open(filepath, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')

            self.get_logger(__name__).info(f"日志已导出到: {filepath}")
            return True

        except Exception as e:
            self.get_logger(__name__).error(f"日志导出失败: {e}")
            return False

    def search_logs(self, query: str,
                    level: Optional[LogLevel] = None,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None) -> List[LogEntry]:
        """搜索日志条目"""
        # 实现日志搜索功能
        # 这里提供基本框架，实际实现可能需要全文搜索引擎

        entries = self.get_log_entries(level, start_time, end_time)
        results = []

        for entry in entries:
            if (query.lower() in entry.message.lower() or
                    query.lower() in entry.logger_name.lower() or
                    any(query.lower() in str(value).lower() for value in entry.extra_data.values())):
                results.append(entry)

        return results

    def shutdown(self):
        """关闭日志系统"""
        start_time = time.time()
        
        with self._lock:
            if self._shutdown:
                return

            try:
                # 刷新所有处理器
                self.flush()

                # 关闭所有处理器
                for handler in self.handlers.values():
                    try:
                        handler.close()
                    except Exception as e:
                        self._log_internal_error("shutdown", e)

                # 清理资源
                self.loggers.clear()
                self.handlers.clear()

                self._shutdown = True

                # 使用基本日志记录最后一条消息
                basic_logger = logging.getLogger('LoggingSystem')
                basic_logger.info("日志系统已关闭")
                
                # 记录关闭性能
                self._record_performance('shutdown', start_time)

            except Exception as e:
                print(f"CRITICAL: Logging system shutdown error: {e}", file=sys.stderr)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()

    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass  # 避免析构函数中的异常

# 全局日志系统实例
_global_logging_system: Optional[LoggingSystem] = None
_global_logging_lock = threading.Lock()

def setup_logging(config: LogConfig) -> LoggingSystem:
    """
    设置全局日志系统

    Args:
        config: 日志配置

    Returns:
        日志系统实例
    """
    global _global_logging_system

    with _global_logging_lock:
        if _global_logging_system is not None:
            raise RuntimeError("全局日志系统已初始化")

        _global_logging_system = LoggingSystem(config)
        return _global_logging_system

def get_logger(name: str) -> logging.Logger:
    """
    获取全局日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    global _global_logging_system

    if _global_logging_system is None:
        # 如果全局系统未初始化，使用基本配置
        logging.basicConfig(
            level=logging.INFO,
            format=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_LOG_DATE_FORMAT
        )
        return logging.getLogger(name)

    return _global_logging_system.get_logger(name)

def get_audit_logger() -> logging.Logger:
    """获取全局审计日志记录器"""
    global _global_logging_system

    if _global_logging_system is None:
        raise RuntimeError("全局日志系统未初始化")

    return _global_logging_system.get_audit_logger()

def get_performance_logger() -> logging.Logger:
    """获取全局性能日志记录器"""
    global _global_logging_system

    if _global_logging_system is None:
        raise RuntimeError("全局日志系统未初始化")

    return _global_logging_system.get_performance_logger()

def get_error_logger() -> logging.Logger:
    """获取全局错误日志记录器"""
    global _global_logging_system

    if _global_logging_system is None:
        raise RuntimeError("全局日志系统未初始化")

    return _global_logging_system.get_error_logger()

def log_audit(action: str, user: str, resource: str,
              status: str, details: Optional[Dict[str, Any]] = None):
    """记录全局审计日志"""
    global _global_logging_system

    if _global_logging_system is not None:
        _global_logging_system.log_audit(action, user, resource, status, details)

def log_performance(operation: str, duration: float,
                    success: bool, metrics: Optional[Dict[str, Any]] = None):
    """记录全局性能日志"""
    global _global_logging_system

    if _global_logging_system is not None:
        _global_logging_system.log_performance(operation, duration, success, metrics)

def log_error(error_type: str, error_message: str,
              context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
    """记录全局错误日志"""
    global _global_logging_system

    if _global_logging_system is not None:
        _global_logging_system.log_error(error_type, error_message, context, exception)

def shutdown_logging():
    """关闭全局日志系统"""
    global _global_logging_system

    with _global_logging_lock:
        if _global_logging_system is not None:
            _global_logging_system.shutdown()
            _global_logging_system = None

def get_logging_stats() -> Dict[str, Any]:
    """获取全局日志系统统计信息"""
    global _global_logging_system

    if _global_logging_system is None:
        return {'status': 'not_initialized'}

    return _global_logging_system.get_stats()

# 默认配置创建函数
def create_default_config() -> LogConfig:
    """创建默认日志配置"""
    return LogConfig()

def create_production_config() -> LogConfig:
    """创建生产环境日志配置"""
    config = LogConfig()
    config.level = LogLevel.INFO
    config.destinations = [LogDestination.FILE, LogDestination.CONSOLE]
    config.audit_log_enabled = True
    config.performance_log_enabled = True
    config.error_log_enabled = True
    config.retention_days = 30
    config.compression_enabled = True
    return config

def create_development_config() -> LogConfig:
    """创建开发环境日志配置"""
    config = LogConfig()
    config.level = LogLevel.DEBUG
    config.destinations = [LogDestination.CONSOLE]
    config.include_timestamp = True
    config.include_module = True
    config.include_function = True
    config.include_line_number = True
    return config

def create_docker_config() -> LogConfig:
    """创建Docker环境配置"""
    config = LogConfig()
    config.destinations = [LogDestination.CONSOLE]  # Docker推荐输出到stdout
    config.format = LogFormat.JSON  # 便于日志收集器处理
    config.level = LogLevel.INFO
    config.include_timestamp = True
    config.include_module = True
    config.include_thread = False  # Docker中线程信息通常不重要
    config.include_process = False
    config.console_output = True
    config.audit_log_enabled = False  # Docker中通常不需要单独的审计日志
    config.performance_log_enabled = False
    config.error_log_enabled = False
    return config

def create_kubernetes_config() -> LogConfig:
    """创建Kubernetes环境配置"""
    config = create_docker_config()
    config.level = LogLevel.DEBUG  # K8s环境中通常需要更详细的日志
    config.include_module = True
    config.include_function = True
    config.include_line_number = True
    return config

# 测试代码
if __name__ == "__main__":
    # 测试日志系统
    config = create_development_config()
    logging_system = LoggingSystem(config)

    logger = logging_system.get_logger("test_logger")
    logger.info("测试信息日志")
    logger.debug("测试调试日志")
    logger.warning("测试警告日志")
    logger.error("测试错误日志")

    # 测试审计日志
    logging_system.log_audit("login", "test_user", "system", "success", {"ip": "127.0.0.1"})

    # 测试性能日志
    logging_system.log_performance("data_fetch", 0.125, True, {"records": 1000})

    # 获取统计信息
    stats = logging_system.get_stats()
    print("日志系统统计:", json.dumps(stats, indent=2))

    # 关闭日志系统
    logging_system.shutdown()