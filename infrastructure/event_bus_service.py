from typing import Any, Dict, Callable
from core.base_processor import BaseProcessor

class EventBusService(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subscribers: Dict[str, list[Callable]] = {}

    def _initialize_core(self) -> bool:
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        return {'status': 'success'}

    def publish(self, topic: str, event: Any):
        for cb in self._subscribers.get(topic, []):
            try:
                cb(event)
            except Exception as e:
                self.error_handler.record_error(e, 'event_publish', extra_context={'topic': topic})

    def subscribe(self, topic: str, callback: Callable):
        self._subscribers.setdefault(topic, []).append(callback)

    def _cleanup_core(self):
        self._subscribers.clear()
