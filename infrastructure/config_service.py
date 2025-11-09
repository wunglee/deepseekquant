from typing import Any, Optional
from .config_manager import ConfigManager

class ConfigService:
    def __init__(self):
        self.cm: Optional[ConfigManager] = None

    def initialize(self) -> bool:
        try:
            self.cm = ConfigManager(config_path=None)
            return True
        except Exception as e:
            # 纯服务不依赖 BaseProcessor 的 error_handler，直接返回错误状态或抛出异常
            return False

    def process(self, *args, **kwargs) -> Any:
        action = kwargs.get('action', 'get')
        key = kwargs.get('key', 'system.name')
        value = kwargs.get('value')
        if action == 'get':
            return {'status': 'success', 'value': self.cm.get_config(key) if self.cm else None}
        if action == 'set':
            ok = self.cm.set_config(key, value) if self.cm else False
            return {'status': 'success' if ok else 'error'}
        return {'status': 'error', 'message': 'unsupported action'}

    def cleanup(self):
        self.cm = None
