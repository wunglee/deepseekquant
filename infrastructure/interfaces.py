from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable

class ITaskManager(ABC):
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def submit_task(self, task_fn: Callable, *args, **kwargs) -> Any: ...

    @abstractmethod
    def record_task_start(self, task_id: str, args: tuple, kwargs: dict) -> None: ...

    @abstractmethod
    def record_task_success(self, task_id: str, result: Any, processing_time: float) -> None: ...

    @abstractmethod
    def record_task_failure(self, task_id: str, error: str, processing_time: float) -> None: ...

    @abstractmethod
    def record_task_end(self, task_id: str) -> None: ...

    @abstractmethod
    def batch_process(self, items: List[Any], batch_size: int, process_fn: Callable, timeout: int) -> List[Any]: ...

    @abstractmethod
    def get_queue_status(self) -> Dict[str, Any]: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @abstractmethod
    def update_config(self, new_config: Any) -> None: ...


class IResourceManager(ABC):
    @abstractmethod
    def allocate_resource(self, resource_type: str, resource_id: str, size: int, timeout: int = 30) -> bool: ...

    @abstractmethod
    def release_resource(self, resource_id: str) -> None: ...


class IOrchestrator(ABC):
    @abstractmethod
    def register_processor(self, processor: Any) -> bool: ...

    @abstractmethod
    def initialize_all(self) -> Dict[str, bool]: ...

    @abstractmethod
    def get_health_report(self) -> Dict[str, Any]: ...


# 基础设施服务接口定义
class ILoggingService(ABC):
    @abstractmethod
    def get_logger(self, name: str): ...

class IConfigService(ABC):
    @abstractmethod
    def process(self, *args, **kwargs) -> Any: ...

class ICacheService(ABC):
    @abstractmethod
    def process(self, *args, **kwargs) -> Any: ...

class IEventBusService(ABC):
    @abstractmethod
    def publish(self, topic: str, event: Any) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, callback: Callable) -> None: ...


# 工厂方法：统一创建基础设施服务（避免循环依赖）
def create_logging_system() -> Any:
    from .logging_service import LoggingSystem, LogConfig
    return LoggingSystem(LogConfig())

def create_config_service() -> Any:
    from .config_service import ConfigService
    return ConfigService()

def create_cache_service() -> Any:
    from .cache_service import CacheService
    return CacheService()

def create_event_bus_service() -> Any:
    from .event_bus_service import EventBusService
    return EventBusService()

def create_task_manager(processor_name: str = 'DefaultProcessor') -> Any:
    from .task_manager import TaskManager, TaskManagerConfig
    return TaskManager(TaskManagerConfig(), processor_name)

def create_resource_manager(processor_name: str = 'DefaultProcessor') -> Any:
    from .resource_manager import ResourceManager, ResourceMonitor, ResourceMonitorConfig
    monitor = ResourceMonitor(ResourceMonitorConfig(), processor_name)
    return ResourceManager(processor_name, monitor)


def create_orchestrator() -> Any:
    from .orchestrator import ProcessorOrchestrator
    return ProcessorOrchestrator()

# 轻量级 Provider 注册表
class InfrastructureProvider:
    _registry: Dict[str, Callable[[], Any]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], Any]) -> None:
        cls._registry[name] = factory

    @classmethod
    def get(cls, name: str) -> Any:
        factory = cls._registry.get(name)
        if factory is None:
            raise KeyError(f"Provider not found: {name}")
        return factory()

# 预注册常用组件（延迟实例化，避免循环）
InfrastructureProvider.register('logging', create_logging_system)
InfrastructureProvider.register('config', create_config_service)
InfrastructureProvider.register('cache', create_cache_service)
InfrastructureProvider.register('event_bus', create_event_bus_service)
InfrastructureProvider.register('task_manager', lambda: create_task_manager())
InfrastructureProvider.register('resource_manager', lambda: create_resource_manager())
InfrastructureProvider.register('orchestrator', create_orchestrator)

