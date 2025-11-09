from typing import Any
from core.processors.base_processor import BaseProcessor

class ExecProcessor(BaseProcessor):
    def _initialize_core(self) -> bool:
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        order = kwargs.get('order', {'symbol': 'TEST', 'qty': 1})
        return {'status': 'success', 'order': order, 'exec': 'simulated'}

    def _cleanup_core(self):
        pass
