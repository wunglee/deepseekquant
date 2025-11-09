from typing import Any, Dict, Callable
from core.base_processor import BaseProcessor

class CacheService(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, Any] = {}

    def _initialize_core(self) -> bool:
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        op = kwargs.get('op', 'get')
        key = str(kwargs.get('key', ''))
        if op == 'get':
            return {'status': 'success', 'value': self._cache.get(key)}
        elif op == 'set':
            self._cache[key] = kwargs.get('value')
            return {'status': 'success'}
        return {'status': 'error', 'message': 'unsupported op'}

    def preload_pattern(self, pattern: str, loader: Callable, ttl: int = 300):
        try:
            data = loader()
            if isinstance(data, dict):
                self._cache.update(data)
        except Exception as e:
            self.error_handler.record_error(e, 'cache_preload', extra_context={'pattern': pattern})

    def _cleanup_core(self):
        self._cache.clear()
