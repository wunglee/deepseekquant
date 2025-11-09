from typing import Any, Optional
from core.base_processor import BaseProcessor
from core.config_manager import ConfigManager

class ConfigService(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm: Optional[ConfigManager] = None

    def _initialize_core(self) -> bool:
        try:
            self.cm = ConfigManager(config_path=None)
            return True
        except Exception as e:
            self.error_handler.record_error(e, 'config_service_init')
            return False

    def _process_core(self, *args, **kwargs) -> Any:
        action = kwargs.get('action', 'get')
        key = kwargs.get('key', 'system.name')
        value = kwargs.get('value')
        if action == 'get':
            return {'status': 'success', 'value': self.cm.get_config(key) if self.cm else None}
        if action == 'set':
            ok = self.cm.set_config(key, value) if self.cm else False
            return {'status': 'success' if ok else 'error'}
        return {'status': 'error', 'message': 'unsupported action'}

    def _cleanup_core(self):
        self.cm = None
