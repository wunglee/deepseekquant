"""
DeepSeekQuant 基础处理器模块 - 重构版本
基于日志系统和配置管理器重构，提供更强大的基类功能
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import traceback
import copy
import logging  # 添加基本日志支持

# 导入公共模块定义
from common import (
    ProcessorState, PerformanceMetrics, TradingMode, RiskLevel,
    SignalType, SignalSource, SignalStatus, DataSourceType,
    DEFAULT_LOG_LEVEL, DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE,
    DEFAULT_INITIAL_CAPITAL, DEFAULT_MAX_POSITION_SIZE,
    ERROR_CONFIG_LOAD, ERROR_CONFIG_VALIDATION, SUCCESS_CONFIG_LOAD,
    DeepSeekQuantEncoder, validate_enum_value, serialize_dict
)

# 导入日志系统
try:
    from logging_system import (
        get_logger, get_audit_logger, get_performance_logger, get_error_logger,
        log_audit, log_performance, log_error, LogLevel
    )
except ImportError:
    # 如果日志系统不可用，使用基本日志作为备选
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # 创建空的备选函数
    get_audit_logger = get_performance_logger = get_error_logger = get_logger
    log_audit = log_performance = log_error = lambda *args, **kwargs: None
    LogLevel = type('LogLevel', (), {'INFO': 'INFO'})

# 导入配置管理器
try:
    from config_manager import ConfigManager, get_global_config_manager
except ImportError:
    # 如果配置管理器不可用，创建模拟类
    class ConfigManager:
        def __init__(self, *args, **kwargs):
            pass
        def get_config(self, key, default=None):
            return default or {}

    def get_global_config_manager():
        return ConfigManager()

# 使用统一的日志记录器
logger = get_logger('DeepSeekQuant.BaseProcessor')


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProcessorConfig:
    """处理器配置数据类 - 增强版本"""
    enabled: bool = True
    module_name: str = ""
    log_level: str = DEFAULT_LOG_LEVEL
    max_threads: int = 8
    processing_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    performance_monitoring: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        'max_memory_mb': 512,
        'max_cpu_percent': 80,
        'max_error_history': 1000,
        'max_task_queue': 10000
    })
    dependencies: List[str] = field(default_factory=list)
    fallback_strategies: Dict[str, str] = field(default_factory=dict)
    health_check_interval: int = 60
    auto_recovery: bool = True
    circuit_breaker: Dict[str, Any] = field(default_factory=lambda: {
        'failure_threshold': 5,
        'recovery_timeout': 300,
        'half_open_max_requests': 3
    })

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessorConfig':
        """从字典创建配置"""
        return cls(**data)


@dataclass
class ResourceUsage:
    """资源使用情况"""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    active_tasks: int = 0
    queue_size: int = 0
    disk_usage_mb: float = 0.0
    network_usage_mb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CircuitBreakerState:
    """熔断器状态"""
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    last_failure_time: Optional[str] = None
    next_retry_time: Optional[str] = None
    consecutive_successes: int = 0


class BaseProcessor(ABC):
    """
    所有处理器的基类 - 重构版本
    集成日志系统和配置管理器，提供更强大的基础功能
    """

    # 类级别的配置和状态
    _global_config_manager: Optional[ConfigManager] = None
    _processors_registry: Dict[str, 'BaseProcessor'] = {}
    _registry_lock = threading.RLock()

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_manager: Optional[ConfigManager] = None,
                 processor_name: Optional[str] = None):
        """
        初始化基础处理器

        Args:
            config: 配置字典（可选，优先使用config_manager）
            config_manager: 配置管理器实例
            processor_name: 处理器名称，如果为None则使用类名
        """
        # 第一步：设置基本属性
        self.processor_name = processor_name or self.__class__.__name__
        self.config_manager = config_manager or self._get_global_config_manager()
        self.raw_config = config or {}

        # 第二步：设置基本日志记录器（用于初始化过程中的错误记录）
        self._setup_basic_logging()

        # 第三步：注册处理器
        with self._registry_lock:
            self._processors_registry[self.processor_name] = self

        # 第四步：初始化基本状态
        self.state = ProcessorState.UNINITIALIZED
        self.health_status = HealthStatus.UNKNOWN
        self.state_lock = threading.RLock()
        self.last_state_change = datetime.now().isoformat()
        self.startup_time = datetime.now().isoformat()

        # 第五步：配置管理（需要谨慎处理，因为可能出错）
        try:
            self.module_config = self._extract_module_config()
            self.processor_config = self._create_processor_config()
        except Exception as e:
            self._log_init_error(f"配置初始化失败: {e}")
            # 使用默认配置继续
            self.module_config = self.raw_config or {}
            self.processor_config = ProcessorConfig(module_name=self.processor_name)

        # 第六步：初始化其他组件
        self.performance_metrics = PerformanceMetrics()
        self.performance_lock = threading.Lock()
        self.performance_history: List[Dict[str, Any]] = []

        self.error_lock = threading.RLock()

        self.resource_usage = ResourceUsage()
        self.thread_pool = None
        self.resource_monitor_thread = None
        self.monitor_running = False

        self.error_count = 0
        self.last_error = None
        self.error_history: List[Dict[str, Any]] = []
        self.circuit_breaker = CircuitBreakerState()

        self.task_queue = []
        self.task_queue_lock = threading.RLock()
        self.active_tasks: Dict[str, Any] = {}

        # 第七步：设置完整的日志系统
        self._setup_full_logging()

        # 第八步：配置观察者
        try:
            self._setup_config_observers()
        except Exception as e:
            self._log_init_error(f"配置观察者设置失败: {e}")

        self.logger.info(f"{self.processor_name} 初始化完成")

    def _setup_basic_logging(self):
        """设置基本日志记录器（用于初始化过程）"""
        # 使用Python标准日志作为备选
        self.logger = logging.getLogger(f'DeepSeekQuant.{self.processor_name}')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 创建基本的审计和错误记录器
        self.audit_logger = self.performance_logger = self.error_logger = self.logger

    def _setup_full_logging(self):
        """设置完整的日志系统"""
        try:
            # 尝试使用完整的日志系统
            self.logger = get_logger(f'DeepSeekQuant.{self.processor_name}')

            # 尝试获取其他日志记录器，如果失败则使用主记录器
            try:
                self.audit_logger = get_audit_logger()
            except:
                self.audit_logger = self.logger

            try:
                self.performance_logger = get_performance_logger()
            except:
                self.performance_logger = self.logger

            try:
                self.error_logger = get_error_logger()
            except:
                self.error_logger = self.logger

        except Exception as e:
            # 如果完整日志系统设置失败，使用基本日志
            self._log_init_error(f"完整日志系统设置失败: {e}")
            # 确保所有日志记录器都有值
            self.audit_logger = self.performance_logger = self.error_logger = self.logger

    def _log_init_error(self, message: str):
        """记录初始化错误"""
        try:
            self.logger.error(message)
        except:
            # 如果日志记录也失败，打印到控制台
            print(f"初始化错误: {message}")

    @classmethod
    def _get_global_config_manager(cls) -> ConfigManager:
        """获取全局配置管理器"""
        if cls._global_config_manager is None:
            cls._global_config_manager = get_global_config_manager()
        return cls._global_config_manager

    @classmethod
    def set_global_config_manager(cls, config_manager: ConfigManager):
        """设置全局配置管理器"""
        cls._global_config_manager = config_manager

    @classmethod
    def get_processor(cls, name: str) -> Optional['BaseProcessor']:
        """获取已注册的处理器实例"""
        with cls._registry_lock:
            return cls._processors_registry.get(name)

    @classmethod
    def get_all_processors(cls) -> Dict[str, 'BaseProcessor']:
        """获取所有已注册的处理器"""
        with cls._registry_lock:
            return copy.deepcopy(cls._processors_registry)

    def _extract_module_config(self) -> Dict[str, Any]:
        """从配置管理器提取模块特定配置"""
        try:
            # 优先使用配置管理器
            if self.config_manager:
                module_config = self.config_manager.get_config(
                    f"processors.{self.processor_name.lower()}", {})

                # 合并原始配置（原始配置优先级更高）
                if self.raw_config:
                    module_config.update(self.raw_config)

                return module_config
            else:
                return self.raw_config

        except Exception as e:
            self._log_init_error(f"配置提取失败: {e}")
            return self.raw_config or {}

    def _create_processor_config(self) -> ProcessorConfig:
        """创建处理器配置对象"""
        try:
            config_data = self.module_config.get('processor_config', {})

            # 设置默认的模块名称
            if 'module_name' not in config_data:
                config_data['module_name'] = self.processor_name

            return ProcessorConfig.from_dict(config_data)

        except Exception as e:
            self._log_init_error(f"处理器配置创建失败: {e}")
            # 返回默认配置
            return ProcessorConfig(module_name=self.processor_name)

    def _setup_config_observers(self):
        """设置配置变更观察者"""
        if not self.config_manager:
            return

        try:
            # 监听处理器配置变更
            config_key = f"processors.{self.processor_name.lower()}"
            self.config_manager.register_observer(
                config_key,
                self._on_config_changed,
                f"{self.processor_name}_config_observer"
            )

            # 监听全局配置变更
            self.config_manager.register_observer(
                None,  # 全局观察者
                self._on_global_config_changed,
                f"{self.processor_name}_global_observer"
            )

            self.logger.debug("配置观察者设置完成")

        except Exception as e:
            self.logger.warning(f"配置观察者设置失败: {e}")

    def _on_config_changed(self, key: str, old_value: Any, new_value: Any):
        """配置变更回调"""
        self.logger.info(f"处理器配置变更: {key}")

        try:
            # 重新加载配置
            self.module_config = self._extract_module_config()
            old_processor_config = self.processor_config
            self.processor_config = self._create_processor_config()

            # 应用配置变更
            self._apply_config_changes(old_processor_config, self.processor_config)

            # 记录审计日志
            self.audit_logger.info(
                f"处理器配置已更新",
                extra={
                    'processor': self.processor_name,
                    'config_key': key,
                    'change_type': 'config_update'
                }
            )

        except Exception as e:
            self.logger.error(f"配置变更处理失败: {e}")
            self.error_logger.error(
                f"配置变更处理失败",
                extra={
                    'processor': self.processor_name,
                    'error': str(e),
                    'config_key': key
                }
            )

    def _on_global_config_changed(self, key: Optional[str], old_value: Any, new_value: Any):
        """全局配置变更回调"""
        if key and 'logging' in key:
            self.logger.info("日志配置变更，重新配置日志系统")
            # 这里可以重新配置日志级别等

    def _apply_config_changes(self, old_config: ProcessorConfig, new_config: ProcessorConfig):
        """应用配置变更"""
        try:
            # 日志级别变更
            if old_config.log_level != new_config.log_level:
                self.logger.info(f"日志级别变更: {old_config.log_level} -> {new_config.log_level}")
                # 实际应用中需要重新配置日志记录器级别

            # 线程池配置变更
            if (old_config.max_threads != new_config.max_threads and
                self.thread_pool is not None):
                self.logger.info(f"线程池大小变更: {old_config.max_threads} -> {new_config.max_threads}")
                self._resize_thread_pool(new_config.max_threads)

            # 性能监控开关
            if old_config.performance_monitoring != new_config.performance_monitoring:
                monitor_status = "启用" if new_config.performance_monitoring else "禁用"
                self.logger.info(f"性能监控{monitor_status}")

        except Exception as e:
            self.logger.error(f"配置变更应用失败: {e}")

    def _resize_thread_pool(self, new_size: int):
        """调整线程池大小"""
        if self.thread_pool is None:
            return

        try:
            # 创建新的线程池
            old_thread_pool = self.thread_pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=new_size,
                thread_name_prefix=f"{self.processor_name}_Worker"
            )

            # 优雅关闭旧线程池
            old_thread_pool.shutdown(wait=True)

            self.logger.info(f"线程池大小已调整为: {new_size}")

        except Exception as e:
            self.logger.error(f"线程池调整失败: {e}")

    def initialize(self) -> bool:
        """
        初始化处理器

        Returns:
            bool: 初始化是否成功
        """
        start_time = time.time()

        with self.state_lock:
            if self.state != ProcessorState.UNINITIALIZED:
                self.logger.warning(f"处理器已经初始化，当前状态: {self.state.value}")
                return False

            self._set_state(ProcessorState.INITIALIZING)

            try:
                # 记录审计日志
                self.audit_logger.info(
                    f"处理器开始初始化",
                    extra={
                        'processor': self.processor_name,
                        'action': 'initialize_start'
                    }
                )

                # 检查依赖
                if not self._check_dependencies():
                    raise RuntimeError("依赖检查失败")

                # 初始化线程池
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.processor_config.max_threads,
                    thread_name_prefix=f"{self.processor_name}_Worker"
                )

                # 启动资源监控
                self._start_resource_monitor()

                # 执行特定初始化
                success = self._initialize_core()

                if success:
                    self._set_state(ProcessorState.READY)
                    self.health_status = HealthStatus.HEALTHY

                    # 记录性能日志
                    initialization_time = time.time() - start_time
                    self.performance_logger.info(
                        f"处理器初始化完成",
                        extra={
                            'processor': self.processor_name,
                            'duration': initialization_time,
                            'success': True
                        }
                    )

                    self.audit_logger.info(
                        f"处理器初始化成功",
                        extra={
                            'processor': self.processor_name,
                            'action': 'initialize_success',
                            'duration': initialization_time
                        }
                    )

                    self.logger.info(f"{self.processor_name} 初始化成功，耗时: {initialization_time:.3f}秒")
                else:
                    self._set_state(ProcessorState.ERROR)
                    self.health_status = HealthStatus.UNHEALTHY
                    self.logger.error(f"{self.processor_name} 初始化失败")

                return success

            except Exception as e:
                self._set_state(ProcessorState.ERROR)
                self.health_status = HealthStatus.UNHEALTHY
                self._record_error(e, "initialize")

                # 记录错误日志
                self.error_logger.error(
                    f"处理器初始化异常",
                    extra={
                        'processor': self.processor_name,
                        'error': str(e),
                        'stack_trace': traceback.format_exc()
                    }
                )

                self.logger.error(f"{self.processor_name} 初始化异常: {e}")
                return False

    @abstractmethod
    def _initialize_core(self) -> bool:
        """
        核心初始化逻辑（由子类实现）

        Returns:
            bool: 初始化是否成功
        """
        pass

    def _check_dependencies(self) -> bool:
        """检查依赖是否可用"""
        try:
            for dep in self.processor_config.dependencies:
                # 这里应该检查实际依赖，简化实现
                self.logger.debug(f"检查依赖: {dep}")

                # 实际实现中可能会检查模块导入、服务连接等
                # 例如: importlib.import_module(dep)

            return True
        except Exception as e:
            self.logger.error(f"依赖检查失败: {e}")
            return False

    def process(self, *args, **kwargs) -> Any:
        """
        处理请求（模板方法）
        """
        start_time = time.time()
        task_id = f"task_{int(start_time * 1000)}_{id(self)}"

        # 检查熔断器
        if not self._check_circuit_breaker():
            error_msg = "处理器处于熔断状态，请求被拒绝"
            try:
                self.logger.warning(error_msg)
            except:
                pass
            return {'error': error_msg, 'circuit_breaker': 'open'}

        with self.state_lock:
            if self.state != ProcessorState.READY:
                error_msg = f"处理器未就绪，当前状态: {self.state.value}"
                try:
                    self.logger.error(error_msg)
                except:
                    pass
                return {'error': error_msg, 'state': self.state.value}

            self._set_state(ProcessorState.PROCESSING)

        try:
            # 记录任务开始
            self._record_task_start(task_id, args, kwargs)

            # 执行实际处理
            result = self._process_core(*args, **kwargs)

            # 检查处理结果是否表示错误
            if isinstance(result, dict) and result.get('status') == 'error':
                # 如果处理核心返回错误结果，也记录为错误
                error_msg = result.get('message', '处理失败')
                error = Exception(error_msg)
                self._record_error(error, "process_core")
                self._update_performance_metrics(False, time.time() - start_time)
                self._record_circuit_breaker_failure()
                return result

            # 更新性能指标和熔断器
            processing_time = time.time() - start_time
            self._update_performance_metrics(True, processing_time)
            self._record_circuit_breaker_success()

            # 记录任务完成
            self._record_task_complete(task_id, result, processing_time)

            return result

        except Exception as e:
            # 错误处理
            processing_time = time.time() - start_time
            self._update_performance_metrics(False, processing_time)

            # 确保错误被正确记录
            self._record_error(e, "process")
            self._record_circuit_breaker_failure()

            # 记录错误日志
            try:
                self.error_logger.error(
                    f"处理任务失败",
                    extra={
                        'processor': self.processor_name,
                        'task_id': task_id,
                        'error': str(e),
                        'processing_time': processing_time
                    }
                )
            except:
                try:
                    self.logger.error(f"处理失败: {e}")
                except:
                    pass

            return {'error': str(e), 'task_id': task_id}

        finally:
            with self.state_lock:
                if self.state == ProcessorState.PROCESSING:
                    self._set_state(ProcessorState.READY)

            # 记录任务结束
            self._record_task_end(task_id)

    @abstractmethod
    def _process_core(self, *args, **kwargs) -> Any:
        """
        核心处理逻辑（由子类实现）

        Returns:
            Any: 处理结果
        """
        pass

    def _check_circuit_breaker(self) -> bool:
        """检查熔断器状态"""
        cb_config = self.processor_config.circuit_breaker

        if self.circuit_breaker.state == "OPEN":
            # 检查是否应该尝试恢复
            if self.circuit_breaker.next_retry_time:
                if datetime.now().isoformat() >= self.circuit_breaker.next_retry_time:
                    self.circuit_breaker.state = "HALF_OPEN"
                    self.circuit_breaker.consecutive_successes = 0
                    self.logger.info("熔断器进入半开状态")
                else:
                    return False

        elif self.circuit_breaker.state == "HALF_OPEN":
            if self.circuit_breaker.consecutive_successes >= cb_config['half_open_max_requests']:
                self.circuit_breaker.state = "CLOSED"
                self.circuit_breaker.failure_count = 0
                self.logger.info("熔断器关闭，恢复正常操作")
            elif self.circuit_breaker.failure_count > 0:
                self.circuit_breaker.state = "OPEN"
                next_retry = datetime.now().timestamp() + cb_config['recovery_timeout']
                self.circuit_breaker.next_retry_time = datetime.fromtimestamp(next_retry).isoformat()
                self.logger.warning("熔断器重新打开")
                return False

        return True

    def _record_circuit_breaker_success(self):
        """记录熔断器成功"""
        if self.circuit_breaker.state == "HALF_OPEN":
            self.circuit_breaker.consecutive_successes += 1
        self.circuit_breaker.failure_count = 0

    def _record_circuit_breaker_failure(self):
        """记录熔断器失败"""
        cb_config = self.processor_config.circuit_breaker

        self.circuit_breaker.failure_count += 1
        self.circuit_breaker.last_failure_time = datetime.now().isoformat()

        if self.circuit_breaker.state == "CLOSED":
            if self.circuit_breaker.failure_count >= cb_config['failure_threshold']:
                self.circuit_breaker.state = "OPEN"
                next_retry = datetime.now().timestamp() + cb_config['recovery_timeout']
                self.circuit_breaker.next_retry_time = datetime.fromtimestamp(next_retry).isoformat()
                self.logger.warning("熔断器打开，进入保护模式")

        elif self.circuit_breaker.state == "HALF_OPEN":
            self.circuit_breaker.consecutive_successes = 0

    def _record_task_start(self, task_id: str, args: tuple, kwargs: dict):
        """记录任务开始"""
        with self.task_queue_lock:
            self.active_tasks[task_id] = {
                'start_time': datetime.now().isoformat(),
                'args': str(args),
                'kwargs': str(kwargs)
            }

    def _record_task_complete(self, task_id: str, result: Any, processing_time: float):
        """记录任务完成"""
        with self.task_queue_lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.update({
                    'end_time': datetime.now().isoformat(),
                    'processing_time': processing_time,
                    'result': str(result)[:200]  # 限制结果长度
                })

    def _record_task_end(self, task_id: str):
        """记录任务结束（清理）"""
        with self.task_queue_lock:
            if task_id in self.active_tasks:
                # 移动到历史记录（可选）
                del self.active_tasks[task_id]

    def _set_state(self, new_state: ProcessorState):
        """更新处理器状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            self.last_state_change = datetime.now().isoformat()

            # 记录状态变更审计日志
            self.audit_logger.info(
                f"处理器状态变更",
                extra={
                    'processor': self.processor_name,
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'timestamp': self.last_state_change
                }
            )

            self.logger.debug(f"状态变更: {old_state.value} -> {new_state.value}")

    # 在 base_processor.py 中修复日志级别问题

    def _update_performance_metrics(self, success: bool, processing_time: float):
        """更新性能指标"""
        with self.performance_lock:
            self.performance_metrics.update(success, processing_time)

            # 记录性能历史
            if len(self.performance_history) >= 1000:  # 限制历史记录大小
                self.performance_history.pop(0)

            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'processing_time': processing_time,
                'total_operations': self.performance_metrics.total_operations
            })

            # 记录性能日志 - 修复日志级别问题
            if self.processor_config.performance_monitoring:
                try:
                    # 使用正确的日志级别（整数）
                    if success:
                        self.performance_logger.info(
                            f"处理完成 - 耗时: {processing_time:.3f}s",
                            extra={
                                'processor': self.processor_name,
                                'duration': processing_time,
                                'success': success,
                                'total_operations': self.performance_metrics.total_operations
                            }
                        )
                    else:
                        self.performance_logger.warning(
                            f"处理失败 - 耗时: {processing_time:.3f}s",
                            extra={
                                'processor': self.processor_name,
                                'duration': processing_time,
                                'success': success,
                                'total_operations': self.performance_metrics.total_operations
                            }
                        )

                except Exception as e:
                    # 避免因日志记录失败而影响主流程
                    try:
                        self.logger.warning(f"性能日志记录失败: {e}")
                    except:
                        pass  # 如果基本日志也失败，则忽略

    # 在 base_processor.py 中彻底修复错误记录问题

    def _record_error(self, error: Exception, context: str):
        """记录错误信息（确保错误计数正确增加）"""
        try:
            # 确保错误计数原子性增加
            with self.error_lock:
                self.error_count += 1

                error_info = {
                    'timestamp': datetime.now().isoformat(),
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'context': context,
                    'processor_state': self.state.value,
                    'error_count': self.error_count  # 记录当前错误计数
                }

                self.last_error = error_info
                self.error_history.append(error_info)

                # 限制错误历史大小
                max_history = self.processor_config.resource_limits.get('max_error_history', 1000)
                if len(self.error_history) > max_history:
                    self.error_history = self.error_history[-max_history:]

        except Exception as e:
            # 如果错误记录过程本身失败，确保至少记录到基本日志
            try:
                self.logger.error(f"错误记录过程失败: {e}")
            except:
                pass  # 如果基本日志也失败，忽略

        finally:
            # 无论如何都要尝试记录错误日志
            try:
                self.error_logger.error(
                    f"处理器错误记录",
                    extra={
                        'processor': self.processor_name,
                        'error_type': type(error).__name__,
                        'error_message': str(error),
                        'context': context,
                        'error_count': self.error_count
                    }
                )
            except Exception as e:
                # 如果错误日志记录失败，使用基本日志
                try:
                    self.logger.error(f"错误日志记录失败: {e}")
                except:
                    pass  # 如果基本日志也失败，忽略

    def _start_resource_monitor(self):
        """启动资源监控线程"""
        if self.monitor_running:
            return

        self.monitor_running = True
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_worker,
            name=f"{self.processor_name}_ResourceMonitor",
            daemon=True
        )
        self.resource_monitor_thread.start()
        self.logger.debug("资源监控线程已启动")

    def _resource_monitor_worker(self):
        """资源监控工作线程"""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            # 如果psutil不可用，使用模拟数据
            process = None

        while self.monitor_running:
            try:
                # 更新资源使用情况
                if process:
                    self.resource_usage.memory_mb = process.memory_info().rss / 1024 / 1024
                    self.resource_usage.cpu_percent = process.cpu_percent()
                    self.resource_usage.thread_count = process.num_threads()
                else:
                    # 模拟数据
                    self.resource_usage.memory_mb = 100.0
                    self.resource_usage.cpu_percent = 25.0
                    self.resource_usage.thread_count = threading.active_count()

                self.resource_usage.timestamp = datetime.now().isoformat()

                # 更新任务队列信息
                with self.task_queue_lock:
                    self.resource_usage.active_tasks = len(self.active_tasks)
                    self.resource_usage.queue_size = len(self.task_queue)

                # 检查资源限制
                self._check_resource_limits()

                # 定期健康检查
                self._perform_health_check()

            except Exception as e:
                self.logger.warning(f"资源监控异常: {e}")

            time.sleep(10)  # 10秒监控间隔

    def _check_resource_limits(self):
        """检查资源限制"""
        limits = self.processor_config.resource_limits

        # 检查内存限制
        if self.resource_usage.memory_mb > limits.get('max_memory_mb', 512):
            self.logger.warning(
                f"内存使用超过限制: {self.resource_usage.memory_mb:.1f}MB > {limits['max_memory_mb']}MB"
            )
            self.health_status = HealthStatus.DEGRADED

        # 检查CPU限制
        if self.resource_usage.cpu_percent > limits.get('max_cpu_percent', 80):
            self.logger.warning(
                f"CPU使用超过限制: {self.resource_usage.cpu_percent:.1f}% > {limits['max_cpu_percent']}%"
            )
            self.health_status = HealthStatus.DEGRADED

    def _perform_health_check(self):
        """执行健康检查"""
        try:
            # 基本状态检查
            if self.state == ProcessorState.ERROR:
                self.health_status = HealthStatus.UNHEALTHY
            elif self.state == ProcessorState.READY:
                # 检查错误率
                total_ops = self.performance_metrics.total_operations
                if total_ops > 0:
                    error_rate = self.performance_metrics.failed_operations / total_ops
                    if error_rate > 0.1:  # 错误率超过10%
                        self.health_status = HealthStatus.DEGRADED
                    else:
                        self.health_status = HealthStatus.HEALTHY
                else:
                    self.health_status = HealthStatus.HEALTHY
            else:
                self.health_status = HealthStatus.UNKNOWN

            # 记录健康状态变更
            if hasattr(self, '_last_health_status') and self._last_health_status != self.health_status:
                self.logger.info(f"健康状态变更: {self._last_health_status.value} -> {self.health_status.value}")

            self._last_health_status = self.health_status

        except Exception as e:
            self.logger.warning(f"健康检查异常: {e}")
            self.health_status = HealthStatus.UNKNOWN

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.performance_lock:
            report = {
                'module': self.processor_name,
                'state': self.state.value,
                'health_status': self.health_status.value,
                'total_operations': self.performance_metrics.total_operations,
                'successful_operations': self.performance_metrics.successful_operations,
                'failed_operations': self.performance_metrics.failed_operations,
                'error_rate': self.performance_metrics.error_rate,
                'availability': self.performance_metrics.availability,
                'total_processing_time': self.performance_metrics.total_processing_time,
                'avg_processing_time': self.performance_metrics.avg_processing_time,
                'max_processing_time': self.performance_metrics.max_processing_time,
                'min_processing_time': self.performance_metrics.min_processing_time,
                'throughput': self.performance_metrics.throughput,
                'last_operation_time': self.performance_metrics.last_operation_time,
                'error_count': self.error_count,
                'last_error': self.last_error,
                'circuit_breaker_state': self.circuit_breaker.state,
                'resource_usage': asdict(self.resource_usage),
                'active_tasks_count': len(self.active_tasks),
                'queue_size': len(self.task_queue)
            }

            # 记录性能报告生成
            self.performance_logger.debug(
                f"性能报告生成",
                extra={
                    'processor': self.processor_name,
                    'report_summary': {
                        'total_operations': report['total_operations'],
                        'error_rate': report['error_rate'],
                        'throughput': report['throughput']
                    }
                }
            )

            return report

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        health_status = {
            'module': self.processor_name,
            'state': self.state.value,
            'health_status': self.health_status.value,
            'last_state_change': self.last_state_change,
            'uptime_seconds': self._get_uptime_seconds(),
            'thread_pool_active': self.thread_pool is not None,
            'resource_usage': asdict(self.resource_usage),
            'performance_metrics': {
                'total_operations': self.performance_metrics.total_operations,
                'error_rate': self.performance_metrics.error_rate,
                'availability': self.performance_metrics.availability
            },
            'circuit_breaker': asdict(self.circuit_breaker),
            'is_healthy': self.health_status == HealthStatus.HEALTHY,
            'last_health_check': datetime.now().isoformat()
        }

        # 添加详细诊断信息
        health_status.update(self._get_detailed_diagnostics())

        return health_status

    def _get_uptime_seconds(self) -> float:
        """获取运行时间（秒）"""
        try:
            start_time = datetime.fromisoformat(self.startup_time)
            current_time = datetime.now()
            return (current_time - start_time).total_seconds()
        except Exception:
            return 0.0

    def _get_detailed_diagnostics(self) -> Dict[str, Any]:
        """获取详细诊断信息"""
        diagnostics = {
            'memory_usage_ratio': self.resource_usage.memory_mb /
                                  max(self.processor_config.resource_limits.get('max_memory_mb', 512), 1),
            'cpu_usage_ratio': self.resource_usage.cpu_percent /
                               max(self.processor_config.resource_limits.get('max_cpu_percent', 80), 1),
            'recent_errors_count': len([e for e in self.error_history
                                        if datetime.fromisoformat(e['timestamp']).timestamp() >
                                        time.time() - 3600]),  # 最近1小时错误数
            'task_queue_health': 'healthy' if len(self.task_queue) < 100 else 'congested',
            'thread_pool_health': 'healthy' if self.thread_pool else 'unavailable'
        }

        # 计算健康评分（0-100）
        health_score = 100

        # 内存使用惩罚
        if diagnostics['memory_usage_ratio'] > 0.8:
            health_score -= 20
        elif diagnostics['memory_usage_ratio'] > 0.9:
            health_score -= 40

        # CPU使用惩罚
        if diagnostics['cpu_usage_ratio'] > 0.8:
            health_score -= 15
        elif diagnostics['cpu_usage_ratio'] > 0.9:
            health_score -= 30

        # 错误率惩罚
        if self.performance_metrics.error_rate > 0.05:
            health_score -= 25
        elif self.performance_metrics.error_rate > 0.1:
            health_score -= 50

        diagnostics['health_score'] = max(0, health_score)

        return diagnostics

    def submit_task(self, task_fn: Callable, *args, **kwargs) -> Any:
        """提交异步任务"""
        if not self.thread_pool:
            raise RuntimeError("线程池未初始化")

        if self.state != ProcessorState.READY:
            raise RuntimeError(f"处理器未就绪，当前状态: {self.state.value}")

        # 检查队列限制
        queue_limit = self.processor_config.resource_limits.get('max_task_queue', 10000)
        with self.task_queue_lock:
            if len(self.task_queue) >= queue_limit:
                raise RuntimeError(f"任务队列已满: {len(self.task_queue)}/{queue_limit}")

        # 提交任务
        future = self.thread_pool.submit(task_fn, *args, **kwargs)

        # 记录任务提交
        self.audit_logger.info(
            f"异步任务提交",
            extra={
                'processor': self.processor_name,
                'task_function': task_fn.__name__,
                'queue_size': len(self.task_queue) + 1
            }
        )

        return future

    def batch_process(self, items: List[Any], batch_size: int = 10) -> List[Any]:
        """批量处理项目"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            try:
                # 使用线程池并行处理批次
                futures = []
                for item in batch:
                    future = self.submit_task(self._process_core, item)
                    futures.append(future)

                # 收集结果
                for future in futures:
                    try:
                        result = future.result(timeout=self.processor_config.processing_timeout)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"批量处理失败: {e}")
                        results.append({'error': str(e)})

            except Exception as e:
                self.logger.error(f"批次处理失败: {e}")
                # 为失败的批次添加错误结果
                results.extend([{'error': str(e)}] * len(batch))

        return results

    def cleanup(self):
        """清理资源"""
        with self.state_lock:
            if self.state == ProcessorState.TERMINATED:
                return

            self._set_state(ProcessorState.SHUTTING_DOWN)
            self.monitor_running = False

        try:
            # 记录清理开始
            self.audit_logger.info(
                f"处理器开始清理",
                extra={
                    'processor': self.processor_name,
                    'action': 'cleanup_start'
                }
            )

            # 停止资源监控
            if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
                self.resource_monitor_thread.join(timeout=5.0)

            # 关闭线程池
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None

            # 取消配置观察者
            if self.config_manager:
                try:
                    self.config_manager.unregister_observer(
                        f"processors.{self.processor_name.lower()}",
                        f"{self.processor_name}_config_observer"
                    )
                    self.config_manager.unregister_observer(
                        None,
                        f"{self.processor_name}_global_observer"
                    )
                except Exception as e:
                    self.logger.warning(f"取消观察者失败: {e}")

            # 执行特定清理
            self._cleanup_core()

            # 从注册表中移除
            with self._registry_lock:
                if self.processor_name in self._processors_registry:
                    del self._processors_registry[self.processor_name]

            with self.state_lock:
                self._set_state(ProcessorState.TERMINATED)

            # 记录清理完成
            self.audit_logger.info(
                f"处理器清理完成",
                extra={
                    'processor': self.processor_name,
                    'action': 'cleanup_success'
                }
            )

            self.logger.info(f"{self.processor_name} 资源清理完成")

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
            with self.state_lock:
                self._set_state(ProcessorState.ERROR)

            # 记录错误日志
            self.error_logger.error(
                f"处理器清理异常",
                extra={
                    'processor': self.processor_name,
                    'error': str(e),
                    'stack_trace': traceback.format_exc()
                }
            )

    @abstractmethod
    def _cleanup_core(self):
        """核心清理逻辑（由子类实现）"""
        pass

    # 在 base_processor.py 中修复重启功能

    def restart(self) -> bool:
        """重启处理器"""
        try:
            self.logger.info(f"{self.processor_name} 开始重启")

            # 记录重启审计
            self.audit_logger.info(
                f"处理器开始重启",
                extra={
                    'processor': self.processor_name,
                    'action': 'restart_start'
                }
            )

            # 先清理
            self.cleanup()

            # 重置状态，以便重新初始化
            with self.state_lock:
                self.state = ProcessorState.UNINITIALIZED
                self.health_status = HealthStatus.UNKNOWN

            # 重新初始化
            success = self.initialize()

            # 记录重启结果
            restart_status = 'success' if success else 'failure'
            self.audit_logger.info(
                f"处理器重启{restart_status}",
                extra={
                    'processor': self.processor_name,
                    'action': f'restart_{restart_status}'
                }
            )

            self.logger.info(f"{self.processor_name} 重启{'成功' if success else '失败'}")
            return success

        except Exception as e:
            self.logger.error(f"重启失败: {e}")
            self.error_logger.error(
                f"处理器重启异常",
                extra={
                    'processor': self.processor_name,
                    'error': str(e),
                    'stack_trace': traceback.format_exc()
                }
            )
            return False

    def emergency_stop(self):
        """紧急停止处理器"""
        try:
            self.logger.warning(f"{self.processor_name} 执行紧急停止")

            # 立即停止监控
            self.monitor_running = False

            # 强制关闭线程池
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)

            # 清理任务队列
            with self.task_queue_lock:
                self.task_queue.clear()
                self.active_tasks.clear()

            # 设置错误状态
            with self.state_lock:
                self._set_state(ProcessorState.ERROR)

            self.audit_logger.warning(
                f"处理器紧急停止",
                extra={
                    'processor': self.processor_name,
                    'action': 'emergency_stop'
                }
            )

        except Exception as e:
            self.logger.error(f"紧急停止失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        if not self.initialize():
            raise RuntimeError(f"{self.processor_name} 初始化失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        try:
            # 只在未清理的情况下执行清理
            if hasattr(self, 'state') and self.state != ProcessorState.TERMINATED:
                self.cleanup()
        except:
            pass  # 避免析构函数中的异常

# 处理器管理器类
class ProcessorManager:
    """处理器管理器 - 用于管理多个处理器实例"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or get_global_config_manager()
        self.processors: Dict[str, BaseProcessor] = {}
        self.manager_lock = threading.RLock()
        self.logger = get_logger('DeepSeekQuant.ProcessorManager')

    def register_processor(self, processor: BaseProcessor) -> bool:
        """注册处理器"""
        with self.manager_lock:
            if processor.processor_name in self.processors:
                self.logger.warning(f"处理器已存在: {processor.processor_name}")
                return False

            self.processors[processor.processor_name] = processor
            self.logger.info(f"处理器注册成功: {processor.processor_name}")
            return True

    def unregister_processor(self, processor_name: str) -> bool:
        """取消注册处理器"""
        with self.manager_lock:
            if processor_name not in self.processors:
                self.logger.warning(f"处理器不存在: {processor_name}")
                return False

            processor = self.processors[processor_name]
            processor.cleanup()
            del self.processors[processor_name]
            self.logger.info(f"处理器取消注册: {processor_name}")
            return True

    def get_processor(self, processor_name: str) -> Optional[BaseProcessor]:
        """获取处理器"""
        with self.manager_lock:
            return self.processors.get(processor_name)

    def initialize_all(self) -> Dict[str, bool]:
        """初始化所有处理器"""
        results = {}
        with self.manager_lock:
            for name, processor in self.processors.items():
                try:
                    results[name] = processor.initialize()
                    self.logger.info(f"处理器初始化 {name}: {'成功' if results[name] else '失败'}")
                except Exception as e:
                    results[name] = False
                    self.logger.error(f"处理器初始化异常 {name}: {e}")

        return results

    def cleanup_all(self):
        """清理所有处理器"""
        with self.manager_lock:
            for name, processor in self.processors.items():
                try:
                    processor.cleanup()
                    self.logger.info(f"处理器清理完成: {name}")
                except Exception as e:
                    self.logger.error(f"处理器清理异常 {name}: {e}")

    def get_health_report(self) -> Dict[str, Any]:
        """获取所有处理器的健康报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_processors': len(self.processors),
            'healthy_processors': 0,
            'degraded_processors': 0,
            'unhealthy_processors': 0,
            'processor_details': {}
        }

        with self.manager_lock:
            for name, processor in self.processors.items():
                try:
                    health_status = processor.get_health_status()
                    report['processor_details'][name] = health_status

                    # 统计健康状态
                    if health_status['is_healthy']:
                        report['healthy_processors'] += 1
                    elif health_status['health_status'] == HealthStatus.DEGRADED.value:
                        report['degraded_processors'] += 1
                    else:
                        report['unhealthy_processors'] += 1

                except Exception as e:
                    self.logger.error(f"获取处理器健康状态失败 {name}: {e}")
                    report['processor_details'][name] = {'error': str(e)}
                    report['unhealthy_processors'] += 1

        return report

    def restart_unhealthy_processors(self) -> Dict[str, bool]:
        """重启不健康的处理器"""
        results = {}
        with self.manager_lock:
            for name, processor in self.processors.items():
                try:
                    health_status = processor.get_health_status()
                    if not health_status['is_healthy']:
                        self.logger.info(f"重启不健康处理器: {name}")
                        results[name] = processor.restart()
                    else:
                        results[name] = True  # 已经是健康的

                except Exception as e:
                    self.logger.error(f"重启处理器失败 {name}: {e}")
                    results[name] = False

        return results

# 全局处理器管理器实例
_global_processor_manager: Optional[ProcessorManager] = None

def get_global_processor_manager() -> ProcessorManager:
    """获取全局处理器管理器"""
    global _global_processor_manager
    if _global_processor_manager is None:
        _global_processor_manager = ProcessorManager()
    return _global_processor_manager

def register_global_processor(processor: BaseProcessor) -> bool:
    """注册处理器到全局管理器"""
    manager = get_global_processor_manager()
    return manager.register_processor(processor)