"""
DeepSeekQuant 基础处理器模块
所有处理器的基类，提供统一的初始化、配置管理和性能监控
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import pandas as pd
import psutil
from typing import Callable
import traceback
from collections import defaultdict
import copy

logger = logging.getLogger('DeepSeekQuant.BaseProcessor')


class ProcessorState(Enum):
    """处理器状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class PerformanceMetrics:
    """性能指标数据类"""

    def __init__(self):
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        self.max_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.last_operation_time = None
        self.throughput = 0.0  # 操作数/秒
        self.error_rate = 0.0
        self.availability = 1.0


@dataclass
class ProcessorConfig:
    """处理器配置"""
    enabled: bool = True
    module_name: str = ""
    log_level: str = "INFO"
    max_threads: int = 8
    processing_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    performance_monitoring: bool = True
    resource_limits: Dict[str, Any] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {}
        if self.dependencies is None:
            self.dependencies = []


class BaseProcessor(ABC):
    """
    所有处理器的基类
    提供统一的初始化、配置管理、性能监控和错误处理
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基础处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.module_config = self._extract_module_config(config)
        self.processor_config = self._create_processor_config()

        # 状态管理
        self.state = ProcessorState.UNINITIALIZED
        self.state_lock = threading.RLock()
        self.last_state_change = datetime.now().isoformat()

        # 性能监控
        self.performance_metrics = PerformanceMetrics()
        self.performance_lock = threading.Lock()

        # 资源管理
        self.thread_pool = None
        self.resource_usage = {
            'memory_mb': 0,
            'cpu_percent': 0,
            'thread_count': 0,
            'active_tasks': 0
        }

        # 错误处理
        self.error_count = 0
        self.last_error = None
        self.error_history: List[Dict[str, Any]] = []

        # 初始化日志
        self._initialize_logging()

        logger.info(f"{self.__class__.__name__} 初始化完成")

    def _extract_module_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """提取模块特定配置"""
        module_name = self.__class__.__name__.lower()
        return config.get(module_name, {})

    def _create_processor_config(self) -> ProcessorConfig:
        """创建处理器配置"""
        return ProcessorConfig(
            enabled=self.module_config.get('enabled', True),
            module_name=self.__class__.__name__,
            log_level=self.module_config.get('log_level', 'INFO'),
            max_threads=self.module_config.get('max_threads', 8),
            processing_timeout=self.module_config.get('processing_timeout', 30),
            retry_attempts=self.module_config.get('retry_attempts', 3),
            retry_delay=self.module_config.get('retry_delay', 1.0),
            performance_monitoring=self.module_config.get('performance_monitoring', True),
            resource_limits=self.module_config.get('resource_limits', {}),
            dependencies=self.module_config.get('dependencies', [])
        )

    def _initialize_logging(self):
        """初始化日志配置"""
        log_level = getattr(logging, self.processor_config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    def initialize(self) -> bool:
        """
        初始化处理器

        Returns:
            bool: 初始化是否成功
        """
        with self.state_lock:
            if self.state != ProcessorState.UNINITIALIZED:
                logger.warning(f"处理器已经初始化，当前状态: {self.state.value}")
                return False

            self._set_state(ProcessorState.INITIALIZING)

            try:
                # 检查依赖
                if not self._check_dependencies():
                    raise RuntimeError("依赖检查失败")

                # 初始化线程池
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.processor_config.max_threads,
                    thread_name_prefix=f"{self.__class__.__name__}_Worker"
                )

                # 执行特定初始化
                success = self._initialize_core()

                if success:
                    self._set_state(ProcessorState.READY)
                    logger.info(f"{self.__class__.__name__} 初始化成功")
                else:
                    self._set_state(ProcessorState.ERROR)
                    logger.error(f"{self.__class__.__name__} 初始化失败")

                return success

            except Exception as e:
                self._set_state(ProcessorState.ERROR)
                self._record_error(e, "initialize")
                logger.error(f"{self.__class__.__name__} 初始化异常: {e}")
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
                logger.debug(f"检查依赖: {dep}")
            return True
        except Exception as e:
            logger.error(f"依赖检查失败: {e}")
            return False

    def process(self, *args, **kwargs) -> Any:
        """
        处理请求（模板方法）

        Returns:
            Any: 处理结果
        """
        start_time = time.time()

        with self.state_lock:
            if self.state != ProcessorState.READY:
                error_msg = f"处理器未就绪，当前状态: {self.state.value}"
                logger.error(error_msg)
                return {'error': error_msg}

            self._set_state(ProcessorState.PROCESSING)

        try:
            # 执行实际处理
            result = self._process_core(*args, **kwargs)

            # 更新性能指标
            processing_time = time.time() - start_time
            self._update_performance_metrics(True, processing_time)

            return result

        except Exception as e:
            # 错误处理
            processing_time = time.time() - start_time
            self._update_performance_metrics(False, processing_time)
            self._record_error(e, "process")

            logger.error(f"处理失败: {e}")
            return {'error': str(e)}

        finally:
            with self.state_lock:
                if self.state == ProcessorState.PROCESSING:
                    self._set_state(ProcessorState.READY)

    @abstractmethod
    def _process_core(self, *args, **kwargs) -> Any:
        """
        核心处理逻辑（由子类实现）

        Returns:
            Any: 处理结果
        """
        pass

    def _set_state(self, new_state: ProcessorState):
        """更新处理器状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            self.last_state_change = datetime.now().isoformat()

            logger.debug(f"状态变更: {old_state.value} -> {new_state.value}")

    def _update_performance_metrics(self, success: bool, processing_time: float):
        """更新性能指标"""
        with self.performance_lock:
            self.performance_metrics.total_operations += 1

            if success:
                self.performance_metrics.successful_operations += 1
            else:
                self.performance_metrics.failed_operations += 1

            self.performance_metrics.total_processing_time += processing_time
            self.performance_metrics.avg_processing_time = (
                    self.performance_metrics.total_processing_time /
                    self.performance_metrics.total_operations
            )

            self.performance_metrics.max_processing_time = max(
                self.performance_metrics.max_processing_time,
                processing_time
            )

            self.performance_metrics.min_processing_time = min(
                self.performance_metrics.min_processing_time,
                processing_time
            )

            self.performance_metrics.last_operation_time = datetime.now().isoformat()

            # 计算吞吐量和错误率
            if self.performance_metrics.total_operations > 0:
                total_time = self.performance_metrics.total_processing_time
                if total_time > 0:
                    self.performance_metrics.throughput = (
                            self.performance_metrics.total_operations / total_time
                    )

                self.performance_metrics.error_rate = (
                        self.performance_metrics.failed_operations /
                        self.performance_metrics.total_operations
                )

                self.performance_metrics.availability = (
                        self.performance_metrics.successful_operations /
                        self.performance_metrics.total_operations
                )

    def _record_error(self, error: Exception, context: str):
        """记录错误信息"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'processor_state': self.state.value
        }

        self.error_count += 1
        self.last_error = error_info
        self.error_history.append(error_info)

        # 限制错误历史大小
        max_error_history = self.processor_config.resource_limits.get('max_error_history', 1000)
        if len(self.error_history) > max_error_history:
            self.error_history = self.error_history[-max_error_history:]

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.performance_lock:
            return {
                'module': self.__class__.__name__,
                'state': self.state.value,
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
                'last_error': self.last_error
            }

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'module': self.__class__.__name__,
            'state': self.state.value,
            'last_state_change': self.last_state_change,
            'thread_pool_active': self.thread_pool is not None,
            'resource_usage': self.resource_usage,
            'performance_metrics': self.get_performance_report(),
            'is_healthy': self.state == ProcessorState.READY and self.error_count == 0
        }

    def cleanup(self):
        """清理资源"""
        with self.state_lock:
            if self.state == ProcessorState.TERMINATED:
                return

            self._set_state(ProcessorState.SHUTTING_DOWN)

        try:
            # 关闭线程池
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None

            # 执行特定清理
            self._cleanup_core()

            with self.state_lock:
                self._set_state(ProcessorState.TERMINATED)

            logger.info(f"{self.__class__.__name__} 资源清理完成")

        except Exception as e:
            logger.error(f"资源清理失败: {e}")
            with self.state_lock:
                self._set_state(ProcessorState.ERROR)

    @abstractmethod
    def _cleanup_core(self):
        """核心清理逻辑（由子类实现）"""
        pass

    def __enter__(self):
        """上下文管理器入口"""
        if not self.initialize():
            raise RuntimeError(f"{self.__class__.__name__} 初始化失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 避免析构函数中的异常