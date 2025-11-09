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


class IProcessorManager(ABC):
    @abstractmethod
    def register_processor(self, processor: Any) -> bool: ...

    @abstractmethod
    def initialize_all(self) -> Dict[str, bool]: ...

    @abstractmethod
    def get_health_report(self) -> Dict[str, Any]: ...
