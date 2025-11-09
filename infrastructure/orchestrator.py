"""
DeepSeekQuant 处理器管理器模块 - 独立文件
"""

from datetime import datetime
import threading
from typing import Dict, Any, Optional
from .interfaces import IOrchestrator

# 统一导入日志和配置系统
try:
    from .logging_service import get_logger
except ImportError:
    import logging
    get_logger = lambda name: logging.getLogger(name)

try:
    from config_manager import get_global_config_manager
except ImportError:
    # 标准路径不可用时抛出异常
    raise


class ProcessorOrchestrator(IOrchestrator):
    """处理器管理器"""

    def __init__(self, config_manager: Optional[Any] = None):
        self.config_manager = config_manager or get_global_config_manager()
        self.processors: Dict[str, Any] = {}
        self.manager_lock = threading.RLock()
        self.logger = get_logger('DeepSeekQuant.ProcessorManager')

    def register_processor(self, processor: Any) -> bool:
        """注册处理器"""
        with self.manager_lock:
            if getattr(processor, 'processor_name', None) in self.processors:
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

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'processor_details': {},
            'total_processors': 0,
            'healthy_processors': 0
        }

        with self.manager_lock:
            report['total_processors'] = len(self.processors)
            healthy_count = 0

            for name, processor in self.processors.items():
                try:
                    health_status = processor.get_health_status()
                    report['processor_details'][name] = health_status
                    if health_status.get('is_healthy', False):
                        healthy_count += 1
                except Exception as e:
                    report['processor_details'][name] = {'error': str(e)}

            report['healthy_processors'] = healthy_count

        return report


# 全局处理器管理器
_global_orchestrator: Optional[ProcessorOrchestrator] = None


def get_global_orchestrator() -> ProcessorOrchestrator:
    """获取全局处理器管理器"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ProcessorOrchestrator()
    return _global_orchestrator
