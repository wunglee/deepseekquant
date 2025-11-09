"""
DeepSeekQuant 基础处理器模块 - 重构优化版本
基于职责分离原则，将功能拆分为专注的组件
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict, field, fields
from enum import Enum
import traceback
import copy

# 导入公共模块
from common import (
    ProcessorState, PerformanceMetrics, TradingMode, RiskLevel,
    SignalType, SignalSource, SignalStatus, DataSourceType,
    DEFAULT_LOG_LEVEL, DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE,
    DEFAULT_INITIAL_CAPITAL, DEFAULT_MAX_POSITION_SIZE,
    ERROR_CONFIG_LOAD, ERROR_CONFIG_VALIDATION, SUCCESS_CONFIG_LOAD,
    DeepSeekQuantEncoder, validate_enum_value, serialize_dict
)

# 导入核心组件
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from resource_monitor import ResourceMonitor, ResourceMonitorConfig, ResourceUsage
from performance_tracker import PerformanceTracker, PerformanceConfig
from error_handler import ErrorHandler, ErrorHandlerConfig, ErrorRecord
from task_manager import TaskManager, TaskManagerConfig, TaskInfo

# 导入日志和配置系统
try:
    from logging_system import (
        get_logger, get_audit_logger, get_performance_logger, get_error_logger,
        log_audit, log_performance, log_error, LogLevel
    )
except ImportError:
    # 简化备选实现
    import logging
    get_logger = lambda name: logging.getLogger(name)
    get_audit_logger = get_performance_logger = get_error_logger = get_logger

try:
    from config_manager import ConfigManager, get_global_config_manager
except ImportError:
    class ConfigManager:
        def __init__(self, *args, **kwargs): pass
        def get_config(self, key, default=None): return default or {}
    get_global_config_manager = lambda: ConfigManager()

logger = get_logger('DeepSeekQuant.BaseProcessor')

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProcessorConfig:
    """处理器配置 - 完整版本"""
    enabled: bool = True
    module_name: str = ""
    log_level: str = "INFO"  # 使用字符串而不是DEFAULT_LOG_LEVEL
    max_threads: int = 8
    processing_timeout: int = 30
    retry_attempts: int = 3
    performance_monitoring: bool = True

    # 组件配置 - 使用默认值而不是field()
    circuit_breaker: Dict[str, Any] = field(default_factory=lambda: {})
    resource_monitor: Dict[str, Any] = field(default_factory=lambda: {})
    performance_tracker: Dict[str, Any] = field(default_factory=lambda: {})
    error_handler: Dict[str, Any] = field(default_factory=lambda: {})
    task_manager: Dict[str, Any] = field(default_factory=lambda: {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'module_name': self.module_name,
            'log_level': self.log_level,
            'max_threads': self.max_threads,
            'processing_timeout': self.processing_timeout,
            'retry_attempts': self.retry_attempts,
            'performance_monitoring': self.performance_monitoring,
            'circuit_breaker': self.circuit_breaker,
            'resource_monitor': self.resource_monitor,
            'performance_tracker': self.performance_tracker,
            'error_handler': self.error_handler,
            'task_manager': self.task_manager
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessorConfig':
        """从字典创建配置，过滤未知字段"""
        # 获取已知字段名称
        known_fields = {f.name for f in fields(cls) if
                        f.name != 'circuit_breaker' and f.name != 'resource_monitor' and f.name != 'performance_tracker' and f.name != 'error_handler' and f.name != 'task_manager'}

        # 过滤数据
        filtered_data = {}
        for key, value in data.items():
            if key in known_fields:
                filtered_data[key] = value
            elif key in ['circuit_breaker', 'resource_monitor', 'performance_tracker', 'error_handler', 'task_manager']:
                # 这些是组件配置，直接传递
                filtered_data[key] = value

        return cls(**filtered_data)

class BaseProcessor(ABC):
    """
    所有处理器的基类 - 重构优化版本
    基于职责分离，使用专用组件管理不同功能
    """

    # 类级别注册表
    _processors_registry: Dict[str, 'BaseProcessor'] = {}
    _registry_lock = threading.RLock()

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 config_manager: Optional[ConfigManager] = None,
                 processor_name: Optional[str] = None):
        """
        初始化基础处理器

        Args:
            config: 处理器特定配置
            config_manager: 配置管理器实例
            processor_name: 处理器名称
        """
        # 基础属性设置
        self.processor_name = processor_name or self.__class__.__name__
        self.config_manager = config_manager or self._get_global_config_manager()
        self.raw_config = config or {}

        # 状态管理
        self.state = ProcessorState.UNINITIALIZED
        self.health_status = HealthStatus.UNKNOWN
        self.state_lock = threading.RLock()
        self.last_state_change = datetime.now().isoformat()
        self.startup_time = datetime.now().isoformat()

        # 配置初始化
        self._setup_configuration()

        # 组件初始化
        self._setup_components()

        # 日志系统设置
        self._setup_logging()

        # 注册处理器
        self._register_processor()

        self.logger.info(f"{self.processor_name} 初始化完成")

    def _setup_configuration(self):
        """配置设置 - 简化版本"""
        try:
            # 直接使用原始配置
            self.module_config = self.raw_config or {}

            # 提取处理器配置
            processor_config_data = self.module_config.get('processor_config', {})

            # 创建处理器配置
            self.processor_config = ProcessorConfig.from_dict(processor_config_data)

            # 设置默认值
            if not self.processor_config.module_name:
                self.processor_config.module_name = self.processor_name

        except Exception as e:
            self._log_error(f"配置初始化失败: {e}")
            # 使用默认配置
            self.processor_config = ProcessorConfig(module_name=self.processor_name)

    def _setup_components(self):
        """初始化各功能组件 - 简化版本"""
        try:
            # 错误处理器
            error_config = self.processor_config.error_handler or {}
            self.error_handler = ErrorHandler(
                ErrorHandlerConfig(**error_config) if error_config else ErrorHandlerConfig(),
                self.processor_name
            )

            # 性能跟踪器
            performance_config = self.processor_config.performance_tracker or {}
            self.performance_tracker = PerformanceTracker(
                PerformanceConfig(**performance_config) if performance_config else PerformanceConfig(),
                self.processor_name
            )

            # 熔断器
            circuit_config = self.processor_config.circuit_breaker or {}
            self.circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(**circuit_config) if circuit_config else CircuitBreakerConfig(),
                self.processor_name
            )

            # 资源监控器
            resource_config = self.processor_config.resource_monitor or {}
            self.resource_monitor = ResourceMonitor(
                ResourceMonitorConfig(**resource_config) if resource_config else ResourceMonitorConfig(),
                self.processor_name
            )

            # 任务管理器
            task_config = self.processor_config.task_manager or {}
            self.task_manager = TaskManager(
                TaskManagerConfig(**task_config) if task_config else TaskManagerConfig(),
                self.processor_name
            )

        except Exception as e:
            self._log_error(f"组件初始化失败: {e}")
            # 使用默认组件
            self.error_handler = ErrorHandler(ErrorHandlerConfig(), self.processor_name)
            self.performance_tracker = PerformanceTracker(PerformanceConfig(), self.processor_name)
            self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(), self.processor_name)
            self.resource_monitor = ResourceMonitor(ResourceMonitorConfig(), self.processor_name)
            self.task_manager = TaskManager(TaskManagerConfig(), self.processor_name)

    def _setup_logging(self):
        """日志系统设置"""
        try:
            self.logger = get_logger(f'DeepSeekQuant.{self.processor_name}')
            self.audit_logger = get_audit_logger()
            self.performance_logger = get_performance_logger()
            self.error_logger = get_error_logger()
        except Exception as e:
            # 备选日志设置
            import logging
            self.logger = self.audit_logger = self.performance_logger = self.error_logger = \
                logging.getLogger(f'DeepSeekQuant.{self.processor_name}')

    def _register_processor(self):
        """注册处理器到全局注册表"""
        with self._registry_lock:
            self._processors_registry[self.processor_name] = self

    def _extract_module_config(self) -> Dict[str, Any]:
        """从配置管理器提取模块配置"""
        try:
            if self.config_manager:
                module_config = self.config_manager.get_config(
                    f"processors.{self.processor_name.lower()}", {})
                # 合并原始配置
                if self.raw_config:
                    module_config.update(self.raw_config)
                return module_config
            return self.raw_config
        except Exception as e:
            self._log_error(f"配置提取失败: {e}")
            return self.raw_config or {}

    # 在 BaseProcessor 类中修复配置提取
    def _create_processor_config(self) -> ProcessorConfig:
        """创建处理器配置对象"""
        try:
            config_data = self.module_config.get('processor_config', {})

            # 处理嵌套配置结构
            if 'resource_limits' in config_data:
                resource_limits = config_data.pop('resource_limits', {})
                if 'max_memory_mb' in resource_limits:
                    config_data.setdefault('resource_monitor', {}).setdefault('max_memory_mb',
                                                                              resource_limits['max_memory_mb'])
                if 'max_cpu_percent' in resource_limits:
                    config_data.setdefault('resource_monitor', {}).setdefault('max_cpu_percent',
                                                                              resource_limits['max_cpu_percent'])

            config_data.setdefault('module_name', self.processor_name)

            # 使用安全的配置创建方法
            return ProcessorConfig.from_dict(config_data)
        except Exception as e:
            self._log_error(f"处理器配置创建失败: {e}")
            # 返回默认配置
            return ProcessorConfig(module_name=self.processor_name)

    def _log_error(self, message: str):
        """错误日志记录"""
        try:
            self.logger.error(message)
        except:
            print(f"错误: {message}")

    def initialize(self) -> bool:
        """
        初始化处理器

        Returns:
            bool: 初始化是否成功
        """
        with self.state_lock:
            if self.state != ProcessorState.UNINITIALIZED:
                self.logger.warning(f"处理器已初始化，当前状态: {self.state.value}")
                return False

            self._set_state(ProcessorState.INITIALIZING)

            try:
                # 审计日志
                self.audit_logger.info("处理器开始初始化", extra={
                    'processor': self.processor_name, 'action': 'initialize_start'
                })

                # 检查依赖
                if not self._check_dependencies():
                    raise RuntimeError("依赖检查失败")

                # 初始化组件
                self._initialize_components()

                # 执行核心初始化
                success = self._initialize_core()

                if success:
                    self._set_state(ProcessorState.READY)
                    self.health_status = HealthStatus.HEALTHY
                    self.audit_logger.info("处理器初始化成功", extra={
                        'processor': self.processor_name, 'action': 'initialize_success'
                    })
                else:
                    self._set_state(ProcessorState.ERROR)
                    self.health_status = HealthStatus.UNHEALTHY

                return success

            except Exception as e:
                self._set_state(ProcessorState.ERROR)
                self.health_status = HealthStatus.UNHEALTHY
                self.error_handler.record_error(e, "initialize")
                return False

    def _initialize_components(self):
        """初始化各功能组件"""
        # 初始化任务管理器（线程池等）
        self.task_manager.initialize()

        # 启动资源监控
        self.resource_monitor.start()

        # 注册配置观察者
        self._setup_config_observers()

    def _setup_config_observers(self):
        """设置配置变更观察者"""
        if not self.config_manager:
            return

        try:
            config_key = f"processors.{self.processor_name.lower()}"
            self.config_manager.register_observer(
                config_key, self._on_config_changed,
                f"{self.processor_name}_config_observer"
            )
        except Exception as e:
            self.logger.warning(f"配置观察者设置失败: {e}")

    def _on_config_changed(self, key: str, old_value: Any, new_value: Any):
        """配置变更回调"""
        self.logger.info(f"处理器配置变更: {key}")
        try:
            # 重新加载配置
            self.module_config = self._extract_module_config()
            old_config = self.processor_config
            self.processor_config = self._create_processor_config()

            # 应用配置变更到各组件
            self._apply_config_changes(old_config, self.processor_config)

        except Exception as e:
            self.error_handler.record_error(e, "config_change")

    def _apply_config_changes(self, old_config: ProcessorConfig, new_config: ProcessorConfig):
        """应用配置变更到各组件"""
        # 各组件处理自己的配置变更
        self.circuit_breaker.update_config(new_config.circuit_breaker)
        self.resource_monitor.update_config(new_config.resource_monitor)
        self.performance_tracker.update_config(new_config.performance_tracker)
        self.error_handler.update_config(new_config.error_handler)
        self.task_manager.update_config(new_config.task_manager)

    def _check_dependencies(self) -> bool:
        """检查依赖是否可用"""
        try:
            for dep in getattr(self.processor_config, 'dependencies', []):
                self.logger.debug(f"检查依赖: {dep}")
            return True
        except Exception as e:
            self.logger.error(f"依赖检查失败: {e}")
            return False

    @abstractmethod
    def _initialize_core(self) -> bool:
        """核心初始化逻辑（由子类实现）"""
        pass

    # 在BaseProcessor中确保错误处理器被正确调用
    def process(self, *args, **kwargs) -> Any:
        """
        处理请求（模板方法）
        """
        start_time = time.time()
        task_id = self.task_manager.generate_task_id()

        # 检查熔断器
        if not self.circuit_breaker.allow_request():
            error_msg = '处理器处于熔断状态'
            self.error_handler.record_error(Exception(error_msg), "circuit_breaker")
            return {'error': error_msg, 'circuit_breaker': 'open'}

        # 检查状态
        with self.state_lock:
            if self.state != ProcessorState.READY:
                error_msg = f'处理器未就绪: {self.state.value}'
                self.error_handler.record_error(Exception(error_msg), "state_check")
                return {'error': error_msg, 'state': self.state.value}
            self._set_state(ProcessorState.PROCESSING)

        # 记录任务开始
        self.task_manager.record_task_start(task_id, args, kwargs)

        try:
            # 执行核心处理
            result = self._process_core(*args, **kwargs)
            processing_time = time.time() - start_time

            # 检查结果中的错误状态
            if isinstance(result, dict) and result.get('status') == 'error':
                # 处理失败但未抛出异常的情况
                error_msg = result.get('message', '处理返回错误状态')
                self.circuit_breaker.record_failure()
                self.performance_tracker.record_failure(processing_time)
                self.error_handler.record_error(Exception(error_msg), "process_error_status")
                self.task_manager.record_task_failure(task_id, error_msg, processing_time)
            else:
                # 更新成功状态
                self.circuit_breaker.record_success()
                self.performance_tracker.record_success(processing_time)
                self.task_manager.record_task_success(task_id, result, processing_time)

            return result

        except Exception as e:
            # 错误处理 - 确保错误处理器被调用
            processing_time = time.time() - start_time
            self.circuit_breaker.record_failure()
            self.performance_tracker.record_failure(processing_time)
            self.error_handler.record_error(e, "process")  # 这行确保错误被记录
            self.task_manager.record_task_failure(task_id, str(e), processing_time)

            return {'error': str(e), 'task_id': task_id}

        finally:
            # 恢复状态
            with self.state_lock:
                if self.state == ProcessorState.PROCESSING:
                    self._set_state(ProcessorState.READY)
            self.task_manager.record_task_end(task_id)

    @abstractmethod
    def _process_core(self, *args, **kwargs) -> Any:
        """核心处理逻辑（由子类实现）"""
        pass

    def _set_state(self, new_state: ProcessorState):
        """更新处理器状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            self.last_state_change = datetime.now().isoformat()

            self.audit_logger.info("处理器状态变更", extra={
                'processor': self.processor_name,
                'old_state': old_state.value,
                'new_state': new_state.value
            })

    def submit_task(self, task_fn: Callable, *args, **kwargs) -> Any:
        """提交异步任务"""
        return self.task_manager.submit_task(task_fn, *args, **kwargs)

    def batch_process(self, items: List[Any], batch_size: int = 10) -> List[Any]:
        """批量处理项目"""
        return self.task_manager.batch_process(
            items, batch_size, self._process_core,
            self.processor_config.processing_timeout
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return self.performance_tracker.get_report()

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        health_status = {
            'module': self.processor_name,
            'state': self.state.value,
            'health_status': self.health_status.value,
            'last_state_change': self.last_state_change,
            'uptime_seconds': self._get_uptime_seconds(),
            'circuit_breaker': self.circuit_breaker.get_status(),
            'performance': self.performance_tracker.get_summary(),
            'resource_usage': self.resource_monitor.get_usage(),
            'task_queue': self.task_manager.get_queue_status()
        }

        # 综合健康评估
        health_status.update(self._assess_health())
        return health_status

    def _assess_health(self) -> Dict[str, Any]:
        """综合健康评估"""
        health_score = 100

        # 基于错误率扣分
        error_rate = self.performance_tracker.get_error_rate()
        if error_rate > 0.1:
            health_score -= 30
        elif error_rate > 0.05:
            health_score -= 15

        # 基于资源使用扣分
        resource_health = self.resource_monitor.assess_health()
        health_score -= resource_health.get('penalty', 0)

        return {
            'health_score': max(0, health_score),
            'is_healthy': health_score >= 70,
            'assessments': {
                'error_rate': error_rate,
                'resource_health': resource_health
            }
        }

    def _get_uptime_seconds(self) -> float:
        """获取运行时间"""
        try:
            start_time = datetime.fromisoformat(self.startup_time)
            return (datetime.now() - start_time).total_seconds()
        except:
            return 0.0

    def cleanup(self):
        """清理资源"""
        with self.state_lock:
            if self.state == ProcessorState.TERMINATED:
                return
            self._set_state(ProcessorState.SHUTTING_DOWN)

        try:
            # 清理各组件
            self.task_manager.cleanup()
            self.resource_monitor.stop()

            # 执行核心清理
            self._cleanup_core()

            # 取消观察者
            if self.config_manager:
                try:
                    config_key = f"processors.{self.processor_name.lower()}"
                    self.config_manager.unregister_observer(
                        config_key, f"{self.processor_name}_config_observer"
                    )
                except Exception as e:
                    self.logger.warning(f"取消观察者失败: {e}")

            # 从注册表移除
            with self._registry_lock:
                if self.processor_name in self._processors_registry:
                    del self._processors_registry[self.processor_name]

            self._set_state(ProcessorState.TERMINATED)
            self.audit_logger.info("处理器清理完成", extra={
                'processor': self.processor_name, 'action': 'cleanup_success'
            })

        except Exception as e:
            self._set_state(ProcessorState.ERROR)
            self.error_handler.record_error(e, "cleanup")

    @abstractmethod
    def _cleanup_core(self):
        """核心清理逻辑（由子类实现）"""
        pass

    def restart(self) -> bool:
        """重启处理器"""
        try:
            self.logger.info(f"{self.processor_name} 开始重启")
            self.cleanup()

            # 重置状态
            with self.state_lock:
                self.state = ProcessorState.UNINITIALIZED
                self.health_status = HealthStatus.UNKNOWN

            # 重新初始化
            return self.initialize()

        except Exception as e:
            self.error_handler.record_error(e, "restart")
            return False

    def emergency_stop(self):
        """紧急停止"""
        try:
            self.task_manager.emergency_stop()
            self.resource_monitor.stop()
            self._set_state(ProcessorState.ERROR)
        except Exception as e:
            self.error_handler.record_error(e, "emergency_stop")

    # 上下文管理器支持
    def __enter__(self):
        if not self.initialize():
            raise RuntimeError(f"{self.processor_name} 初始化失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    # 类方法
    @classmethod
    def _get_global_config_manager(cls) -> ConfigManager:
        """获取全局配置管理器"""
        return get_global_config_manager()

    @classmethod
    def get_processor(cls, name: str) -> Optional['BaseProcessor']:
        """获取已注册的处理器"""
        with cls._registry_lock:
            return cls._processors_registry.get(name)

    @classmethod
    def get_all_processors(cls) -> Dict[str, 'BaseProcessor']:
        """获取所有处理器"""
        with cls._registry_lock:
            return copy.deepcopy(cls._processors_registry)

# 处理器管理器（简化版本）
class ProcessorManager:
    """处理器管理器"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or get_global_config_manager()
        self.processors: Dict[str, BaseProcessor] = {}
        self.manager_lock = threading.RLock()
        self.logger = get_logger('DeepSeekQuant.ProcessorManager')

    def register_processor(self, processor: BaseProcessor) -> bool:
        """注册处理器"""
        with self.manager_lock:
            if processor.processor_name in self.processors:
                return False
            self.processors[processor.processor_name] = processor
            return True

    def initialize_all(self) -> Dict[str, bool]:
        """初始化所有处理器"""
        results = {}
        with self.manager_lock:
            for name, processor in self.processors.items():
                try:
                    results[name] = processor.initialize()
                except Exception as e:
                    results[name] = False
                    self.logger.error(f"处理器初始化失败 {name}: {e}")
        return results

    # 在 base_processor.py 中修复 ProcessorManager
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'processor_details': {},
            'total_processors': 0,  # 添加缺失字段
            'healthy_processors': 0  # 添加缺失字段
        }

        with self.manager_lock:
            report['total_processors'] = len(self.processors)
            healthy_count = 0

            for name, processor in self.processors.items():
                try:
                    health_status = processor.get_health_status()
                    report['processor_details'][name] = health_status

                    # 统计健康处理器数量
                    if health_status.get('is_healthy', False):
                        healthy_count += 1
                except Exception as e:
                    report['processor_details'][name] = {'error': str(e)}

            report['healthy_processors'] = healthy_count

        return report

# 全局处理器管理器
_global_processor_manager: Optional[ProcessorManager] = None

def get_global_processor_manager() -> ProcessorManager:
    """获取全局处理器管理器"""
    global _global_processor_manager
    if _global_processor_manager is None:
        _global_processor_manager = ProcessorManager()
    return _global_processor_manager