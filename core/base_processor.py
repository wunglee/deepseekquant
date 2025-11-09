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

# 导入公共模块 - 使用统一的导入方式
try:
    from common import (
        ProcessorState, PerformanceMetrics, TradingMode, RiskLevel,
        SignalType, SignalSource, SignalStatus, DataSourceType,
        DEFAULT_LOG_LEVEL, DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE,
        DEFAULT_INITIAL_CAPITAL, DEFAULT_MAX_POSITION_SIZE,
        ERROR_CONFIG_LOAD, ERROR_CONFIG_VALIDATION, SUCCESS_CONFIG_LOAD,
        DeepSeekQuantEncoder, validate_enum_value, serialize_dict
    )
except ImportError:
    # 如果直接导入失败，尝试从父目录导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import (
        ProcessorState, PerformanceMetrics, TradingMode, RiskLevel,
        SignalType, SignalSource, SignalStatus, DataSourceType,
        DEFAULT_LOG_LEVEL, DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE,
        DEFAULT_INITIAL_CAPITAL, DEFAULT_MAX_POSITION_SIZE,
        ERROR_CONFIG_LOAD, ERROR_CONFIG_VALIDATION, SUCCESS_CONFIG_LOAD,
        DeepSeekQuantEncoder, validate_enum_value, serialize_dict
    )

# 导入核心组件 - 使用统一的导入方式
try:
    from infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    from infrastructure.resource_manager import ResourceMonitor, ResourceMonitorConfig, ResourceUsage
    from infrastructure.performance_tracker import PerformanceTracker, PerformanceConfig
    from infrastructure.error_handler import ErrorHandler, ErrorHandlerConfig, ErrorRecord
    from infrastructure.task_manager import TaskManager, TaskManagerConfig, TaskInfo
except ImportError:
    # 备用导入已移除，infrastructure 和根目录为标准路径
    raise

# 导入日志和配置系统
try:
    from infrastructure.logging_service import (
        get_logger, get_audit_logger, get_performance_logger, get_error_logger,
        log_audit, log_performance, log_error, LogLevel
    )
except ImportError:
    import logging
    get_logger = lambda name: logging.getLogger(name)
    
    def get_audit_logger():
        return logging.getLogger('DeepSeekQuant.Audit')
    
    def get_performance_logger():
        return logging.getLogger('DeepSeekQuant.Performance')
    
    def get_error_logger():
        return logging.getLogger('DeepSeekQuant.Error')

from infrastructure.interfaces import InfrastructureProvider
get_global_config_manager = lambda: InfrastructureProvider.get('config')

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

try:
    from infrastructure.resource_manager import ResourceManager
except ImportError:
    raise

try:
    from infrastructure.orchestrator import ProcessorOrchestrator, get_global_orchestrator
except ImportError:
    raise


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
                 config_manager: Optional[Any] = None,
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

            # 统一资源管理器
            self.resource_manager = ResourceManager(self.processor_name, self.resource_monitor)

        except Exception as e:
            self._log_error(f"组件初始化失败: {e}")
            # 使用默认组件
            self.error_handler = ErrorHandler(ErrorHandlerConfig(), self.processor_name)
            self.performance_tracker = PerformanceTracker(PerformanceConfig(), self.processor_name)
            self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(), self.processor_name)
            self.resource_monitor = ResourceMonitor(ResourceMonitorConfig(), self.processor_name)
            self.task_manager = TaskManager(TaskManagerConfig(), self.processor_name)

    def _setup_logging(self):
        """统一的日志系统设置 - 增强版本带上下文支持"""
        try:
            # 主日志记录器
            self.logger = get_logger(f'DeepSeekQuant.{self.processor_name}')
            
            # 专用日志记录器
            self.audit_logger = get_audit_logger()
            self.performance_logger = get_performance_logger()
            self.error_logger = get_error_logger()
            
            # 日志上下文 - 统一的日志元数据
            self._log_context = {
                'processor_name': self.processor_name,
                'processor_type': self.__class__.__name__,
                'startup_time': self.startup_time
            }
            
            self.logger.info(f"{self.processor_name} 日志系统初始化完成", extra=self._log_context)
            
        except Exception as e:
            # 备选日志设置
            import logging
            base_logger = logging.getLogger(f'DeepSeekQuant.{self.processor_name}')
            self.logger = self.audit_logger = self.performance_logger = self.error_logger = base_logger
            
            # 备选日志上下文
            self._log_context = {
                'processor_name': self.processor_name,
                'processor_type': self.__class__.__name__,
                'startup_time': self.startup_time
            }
            
            self.logger.warning(f"使用备选日志系统: {e}")

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

    def _log_error(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """统一的错误日志记录 - 带上下文支持"""
        try:
            # 合并日志上下文
            log_data = {**self._log_context}
            if extra_context:
                log_data.update(extra_context)
            
            self.logger.error(message, extra=log_data)
            self.error_logger.error(message, extra=log_data)
        except:
            print(f"错误: {message}")
    
    def _log_info(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """统一的信息日志记录 - 带上下文支持"""
        try:
            log_data = {**self._log_context}
            if extra_context:
                log_data.update(extra_context)
            self.logger.info(message, extra=log_data)
        except:
            print(f"信息: {message}")
    
    def _log_warning(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """统一的警告日志记录 - 带上下文支持"""
        try:
            log_data = {**self._log_context}
            if extra_context:
                log_data.update(extra_context)
            self.logger.warning(message, extra=log_data)
        except:
            print(f"警告: {message}")
    
    def _log_debug(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """统一的调试日志记录 - 带上下文支持"""
        try:
            log_data = {**self._log_context}
            if extra_context:
                log_data.update(extra_context)
            self.logger.debug(message, extra=log_data)
        except:
            pass  # 调试日志失败时不输出
    
    def _log_audit(self, action: str, details: Optional[Dict[str, Any]] = None):
        """统一的审计日志记录"""
        try:
            audit_data = {
                **self._log_context,
                'action': action,
                'timestamp': datetime.now().isoformat()
            }
            if details:
                audit_data.update(details)
            self.audit_logger.info(f"审计: {action}", extra=audit_data)
        except:
            pass
    
    def _log_performance(self, operation: str, duration: float, success: bool, 
                        details: Optional[Dict[str, Any]] = None):
        """统一的性能日志记录"""
        try:
            perf_data = {
                **self._log_context,
                'operation': operation,
                'duration': duration,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            if details:
                perf_data.update(details)
            self.performance_logger.info(
                f"性能: {operation} - {duration:.3f}s - {'成功' if success else '失败'}",
                extra=perf_data
            )
        except:
            pass

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
                # 增强错误记录 - 添加初始化上下文
                self.error_handler.record_error(e, "initialize", extra_context={
                    'processor_name': self.processor_name,
                    'processor_type': self.__class__.__name__,
                    'state_before': ProcessorState.INITIALIZING.value
                }, severity="CRITICAL")
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
            observer_fn = getattr(self.config_manager, 'register_observer', None)
            if callable(observer_fn):
                observer_fn(
                    config_key, self._on_config_changed,
                    f"{self.processor_name}_config_observer"
                )
            else:
                self.logger.warning("配置管理器不支持观察者注册接口")
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
            # 增强错误记录 - 添加配置变更上下文
            self.error_handler.record_error(e, "config_change", extra_context={
                'key': key,
                'processor_name': self.processor_name
            })

    def _apply_config_changes(self, old_config: ProcessorConfig, new_config: ProcessorConfig):
        """应用配置变更到各组件"""
        self.circuit_breaker.update_config(CircuitBreakerConfig(**new_config.circuit_breaker))
        self.resource_monitor.update_config(ResourceMonitorConfig(**new_config.resource_monitor))
        self.performance_tracker.update_config(PerformanceConfig(**new_config.performance_tracker))
        self.error_handler.update_config(ErrorHandlerConfig(**new_config.error_handler))
        self.task_manager.update_config(TaskManagerConfig(**new_config.task_manager))

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
            self.error_handler.record_error(Exception(error_msg), "circuit_breaker", extra_context={
                'task_id': task_id,
                'circuit_breaker_state': self.circuit_breaker.state.state,
                'processor_name': self.processor_name
            }, severity="WARNING")
            return {'error': error_msg, 'circuit_breaker': 'open'}

        # 检查状态
        with self.state_lock:
            if self.state != ProcessorState.READY:
                error_msg = f'处理器未就绪: {self.state.value}'
                self.error_handler.record_error(Exception(error_msg), "state_check", extra_context={
                    'task_id': task_id,
                    'current_state': self.state.value,
                    'processor_name': self.processor_name
                }, severity="WARNING")
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
                self.performance_tracker.record_failure(processing_time, "process", {"task_id": task_id})
                # 增强错误记录
                self.error_handler.record_error(Exception(error_msg), "process_error_status", extra_context={
                    'task_id': task_id,
                    'processing_time': processing_time,
                    'result': result,
                    'processor_name': self.processor_name
                })
                self.task_manager.record_task_failure(task_id, error_msg, processing_time)
            else:
                # 更新成功状态
                self.circuit_breaker.record_success()
                self.performance_tracker.record_success(processing_time, "process", {"task_id": task_id})
                self.task_manager.record_task_success(task_id, result, processing_time)

            return result

        except Exception as e:
            # 错误处理 - 增强版本带完整上下文
            processing_time = time.time() - start_time
            self.circuit_breaker.record_failure()
            self.performance_tracker.record_failure(processing_time, "process", {"task_id": task_id})
            # 记录错误带完整上下文
            self.error_handler.record_error(e, "process", extra_context={
                'task_id': task_id,
                'processing_time': processing_time,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'processor_name': self.processor_name,
                'state': self.state.value
            }, severity="ERROR")
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
                    unregister_fn = getattr(self.config_manager, 'unregister_observer', None)
                    if callable(unregister_fn):
                        unregister_fn(
                            config_key, f"{self.processor_name}_config_observer"
                        )
                    else:
                        self.logger.warning("配置管理器不支持观察者取消接口")
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
            # 增强错误记录 - 添加清理上下文
            self.error_handler.record_error(e, "cleanup", extra_context={
                'processor_name': self.processor_name,
                'state_before': ProcessorState.SHUTTING_DOWN.value
            }, severity="ERROR")

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
    def _get_global_config_manager(cls):
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

