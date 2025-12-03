"""
共享配置管理器（业务层）

职责：
- 提供标准化的配置管理接口
- 支持配置的加载、验证和更新
- 提供配置的类型安全访问
"""

import json
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger('DeepSeekQuant.Core.Share.ConfigManager')


@dataclass
class MonitoringConfig:
    """监控配置数据类"""
    check_interval: int = 300  # 检查间隔（秒）
    max_quality_history: int = 5000  # 最大质量历史记录数
    alert_threshold: float = 0.8  # 告警阈值
    enable_cross_validation: bool = True  # 是否启用交叉验证
    cross_validation_interval: int = 3600  # 交叉验证间隔（秒）


@dataclass
class AlertingConfig:
    """告警配置数据类"""
    enable_email: bool = False
    enable_sms: bool = False
    enable_wechat: bool = True
    enable_dingtalk: bool = False
    enable_slack: bool = False
    enable_webhook: bool = False
    enable_log: bool = True
    min_severity: str = "warning"  # 最小告警严重程度
    deduplication_window: int = 3600  # 去重窗口（秒）


@dataclass
class DataConfig:
    """数据配置数据类"""
    primary_source: str = "yahoo"
    default_index: str = "SPX"
    backup_sources: list = None
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 缓存过期时间（秒）
    max_retries: int = 3  # 最大重试次数
    
    def __post_init__(self):
        if self.backup_sources is None:
            self.backup_sources = ["mock"]


@dataclass
class SystemConfig:
    """系统配置数据类"""
    log_level: str = "INFO"
    max_concurrent_requests: int = 10
    timeout: int = 30  # 请求超时（秒）
    enable_health_check: bool = True
    health_check_interval: int = 60  # 健康检查间隔（秒）


class ConfigManager:
    """
    配置管理器
    
    职责：提供标准化的配置管理接口
    """
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        self.config_file = config_file
        self.environment = environment or os.getenv('DEEPSEEK_ENV', 'dev')  # dev/test/prod
        self._config = {}
        self._load_config()
        # 启动热加载监听（仅在非测试环境）
        if self.environment != 'test':
            self._start_hot_reload_watcher()
    
    def _load_config(self):
        """加载配置（优先从 core_bak_refactored/config/*.yml 读取，找不到则回退默认）"""
        # 优先读取目录化YAML配置
        try:
            import yaml  # 可选依赖
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
            # 环境隔离支持：优先加载环境目录下的配置
            env_dir = os.path.join(base_dir, self.environment)
            monitoring_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'monitoring.yml')
            alerting_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'alerting.yml')
            data_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'data.yml')
            system_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'system.yml')
            event_window_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'event_window.yml')
            regional_data_source_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'regional_data_source.yml')
            dashboard_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'dashboard.yml')
            api_service_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'api_service.yml')
            loaded = {}
            for path, key in [
                (monitoring_path, 'monitoring'),
                (alerting_path, 'alerting'),
                (data_path, 'data'),
                (system_path, 'system'),
                (event_window_path, 'event_window'),
                (regional_data_source_path, 'regional_data_source'),
                (dashboard_path, 'dashboard'),
                (api_service_path, 'api_service'),
            ]:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded[key] = yaml.safe_load(f) or {}
            if loaded:
                self._config = loaded
                logger.info(f"从YAML加载配置[{self.environment}]: {list(loaded.keys())}")
                return
        except Exception as e:
            logger.warning(f"YAML配置加载失败，回退默认: {e}")
        
        # 其次尝试单文件（如果显式提供）
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info(f"配置加载成功: {self.config_file}")
                return
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
                self._config = {}
        
        # 回退默认配置
        self._config = self._get_default_config()
        logger.info("使用默认配置")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置（作为YAML缺失时的回退）"""
        return {
            'monitoring': asdict(MonitoringConfig()),
            'alerting': asdict(AlertingConfig()),
            'data': asdict(DataConfig()),
            'system': asdict(SystemConfig())
        }
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        config_dict = self._config.get('monitoring', {})
        return MonitoringConfig(**config_dict)
    
    def get_alerting_config(self) -> AlertingConfig:
        """获取告警配置"""
        config_dict = self._config.get('alerting', {})
        return AlertingConfig(**config_dict)
    
    def get_data_config(self) -> DataConfig:
        """获取数据配置"""
        config_dict = self._config.get('data', {})
        return DataConfig(**config_dict)
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        config_dict = self._config.get('system', {})
        return SystemConfig(**config_dict)
    
    def _start_hot_reload_watcher(self):
        """启动热加载监听器（仅在非测试环境）"""
        try:
            import threading
            import time
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConfigReloadHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager
                
                def on_modified(self, event):
                    if event.src_path.endswith(('.yml', '.yaml')):
                        logger.info(f"检测到配置文件变更: {event.src_path}")
                        time.sleep(0.1)  # 防止写入未完成
                        self.config_manager._load_config()
            
            # 启动后台监听线程
            def watch_thread():
                base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
                env_dir = os.path.join(base_dir, self.environment)
                watch_path = env_dir if os.path.exists(env_dir) else base_dir
                
                event_handler = ConfigReloadHandler(self)
                observer = Observer()
                observer.schedule(event_handler, watch_path, recursive=False)
                observer.start()
                logger.info(f"启动配置热加载监听: {watch_path}")
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    observer.stop()
                observer.join()
            
            watcher = threading.Thread(target=watch_thread, daemon=True, name='ConfigHotReload')
            watcher.start()
        except ImportError:
            logger.info("未安装watchdog，跳过配置热加载")
        except Exception as e:
            logger.warning(f"配置热加载启动失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, config_file: Optional[str] = None):
        """保存配置到文件"""
        save_path = config_file or self.config_file
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                logger.info(f"配置保存成功: {save_path}")
            except Exception as e:
                logger.error(f"配置保存失败: {e}")
                raise
    
    def update(self, config_dict: Dict[str, Any]):
        """更新配置"""
        def deep_update(original: Dict, updates: Dict):
            for key, value in updates.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
        
        deep_update(self._config, config_dict)