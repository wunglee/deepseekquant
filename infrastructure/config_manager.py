"""
DeepSeekQuant 配置管理器 - 重构版本
基于common.py和logging_system.py重构，提供完整的配置管理功能
"""

import json
import yaml
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import copy
import jsonschema
from dataclasses import dataclass, asdict, field
import toml
import threading
from collections import defaultdict
import re
import tempfile
import shutil
from cryptography.fernet import Fernet
import base64
import uuid
import time

# 从common模块导入共享定义
from common import (
    ConfigFormat, ConfigSource, ProcessorState, TradingMode, RiskLevel,
    DEFAULT_CONFIG_PATH, DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_DATE_FORMAT, MAX_LOG_FILE_SIZE, BACKUP_LOG_COUNT,
    DEFAULT_ENCODING, ERROR_CONFIG_LOAD, ERROR_CONFIG_VALIDATION,
    SUCCESS_CONFIG_LOAD, SUCCESS_CONFIG_VALIDATION,
    WARNING_CONFIG_DEFAULT, DeepSeekQuantEncoder,
    serialize_dict, deserialize_dict, validate_enum_value
)

# 日志系统通过 Provider 获取
from .interfaces import InfrastructureProvider
_logging_system = InfrastructureProvider.get('logging')
get_logger = _logging_system.get_logger

def log_audit(action: str, user: str, resource: str, status: str, details: Dict[str, Any]) -> None:
    _logging_system.get_logger('DeepSeekQuant.Audit').info(
        f"审计: {action}", extra={
            'user': user,
            'resource': resource,
            'status': status,
            **(details or {})
        }
    )

def log_performance(operation: str, duration: float, success: bool, details: Dict[str, Any]) -> None:
    _logging_system.get_logger('DeepSeekQuant.Performance').info(
        f"性能: {operation} - {duration:.3f}s - {'成功' if success else '失败'}",
        extra={'operation': operation, 'duration': duration, 'success': success, **(details or {})}
    )

def log_error(event: str, message: str, details: Dict[str, Any]) -> None:
    _logging_system.get_logger('DeepSeekQuant.Error').error(
        f"错误: {event} - {message}", extra={'event': event, **(details or {})}
    )

# 使用统一的日志记录器
logger = get_logger('DeepSeekQuant.ConfigManager')

# 自定义异常类
class ConfigValidationError(Exception):
    """配置验证错误"""
    pass

class ConfigEncryptionError(Exception):
    """配置加密错误"""
    pass

class ConfigVersionError(Exception):
    """配置版本错误"""
    pass

@dataclass
class ConfigMetadata:
    """配置元数据"""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: ConfigSource = ConfigSource.FILE
    format: ConfigFormat = ConfigFormat.JSON
    checksum: str = ""
    encrypted: bool = False
    description: str = "DeepSeekQuant System Configuration"
    environment: str = "production"
    author: str = "DeepSeekQuant System"
    change_history: List[Dict] = field(default_factory=list)

@dataclass
class ConfigChange:
    """配置变更记录"""
    timestamp: str
    change_type: str  # 'create', 'update', 'delete', 'rollback', 'merge', 'reset'
    changed_by: str
    changes: Dict[str, Any]
    previous_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: str = ""
    version_before: str = ""
    version_after: str = ""

class ConfigManager:
    """配置管理器 - 基于common.py和logging_system.py重构的完整版本"""

    # 使用common.py中的默认配置路径
    DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH

    # 默认配置架构
    DEFAULT_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "system": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "environment": {"type": "string", "enum": ["development", "testing", "staging", "production"]},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "max_memory_mb": {"type": "integer", "minimum": 256},
                    "max_threads": {"type": "integer", "minimum": 1, "maximum": 100},
                    "auto_recovery": {"type": "boolean"},
                    "trading_mode": {"type": "string", "enum": [mode.value for mode in TradingMode]},
                    "snapshots_dir": {"type": "string"},
                    "max_change_history": {"type": "integer", "minimum": 10, "maximum": 10000}
                },
                "required": ["name", "version", "environment", "log_level"]
            },
            "data_sources": {
                "type": "object",
                "properties": {
                    "primary": {"type": "string"},
                    "fallback_sources": {"type": "array", "items": {"type": "string"}},
                    "cache_enabled": {"type": "boolean"},
                    "cache_size_mb": {"type": "integer", "minimum": 10},
                    "request_timeout": {"type": "number", "minimum": 1},
                    "retry_attempts": {"type": "integer", "minimum": 0, "maximum": 10}
                }
            },
            "trading": {
                "type": "object",
                "properties": {
                    "initial_capital": {"type": "number", "minimum": 0},
                    "max_position_size": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_leverage": {"type": "number", "minimum": 1},
                    "commission_rate": {"type": "number", "minimum": 0},
                    "slippage_rate": {"type": "number", "minimum": 0}
                }
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "max_daily_loss": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_position_risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "var_confidence_level": {"type": "number", "minimum": 0.5, "maximum": 0.99},
                    "risk_level": {"type": "string", "enum": [level.value for level in RiskLevel]}
                }
            }
        },
        "required": ["system", "data_sources"]
    }

    # 默认配置 - 使用common.py中的常量
    DEFAULT_CONFIG = {
        "system": {
            "name": "DeepSeekQuant",
            "version": "1.0.0",
            "environment": "development",
            "log_level": DEFAULT_LOG_LEVEL,
            "max_memory_mb": 2048,
            "max_threads": 20,
            "auto_recovery": True,
            "trading_mode": TradingMode.PAPER_TRADING.value,
            "snapshots_dir": "config/snapshots",
            "max_change_history": 1000,
            "performance_monitoring": True,
            "alerting_enabled": True
        },
        "data_sources": {
            "primary": "yahoo_finance",
            "fallback_sources": ["alpha_vantage", "iex_cloud", "polygon"],
            "cache_enabled": True,
            "cache_size_mb": 500,
            "request_timeout": 30,
            "retry_attempts": 3
        },
        "logging": {
            "level": DEFAULT_LOG_LEVEL,
            "file_path": "logs/deepseekquant.log",
            "max_file_size_mb": MAX_LOG_FILE_SIZE // (1024 * 1024),
            "backup_count": BACKUP_LOG_COUNT,
            "console_output": True,
            "json_format": False
        },
        "trading": {
            "initial_capital": 1000000.0,
            "max_position_size": 0.1,
            "max_leverage": 2.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "risk_free_rate": 0.02,
            "min_trade_size": 1000,
            "max_drawdown_limit": 0.2,
            "stop_loss_enabled": True,
            "take_profit_enabled": True
        },
        "risk_management": {
            "max_daily_loss": 0.05,
            "max_position_risk": 0.02,
            "var_confidence_level": 0.95,
            "risk_level": RiskLevel.MODERATE.value
        }
    }

    def __init__(self, config_path: Optional[str] = None,
                 encryption_key: Optional[str] = None,
                 validate_on_load: bool = True,
                 enable_hot_reload: bool = False):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            encryption_key: 加密密钥，用于加密敏感配置
            validate_on_load: 是否在加载时验证配置
            enable_hot_reload: 是否启用配置热重载
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.encryption_key = encryption_key
        self.validate_on_load = validate_on_load
        self.enable_hot_reload = enable_hot_reload
        self.config: Dict[str, Any] = {}
        self.metadata = ConfigMetadata()
        self.change_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._observers: Dict[str, List[tuple[str, Callable]]] = defaultdict(list)
        self._encryption_cipher: Optional[Fernet] = None
        self._file_observer = None
        self._value_cache: Dict[str, Any] = {}  # 配置值缓存

        # 初始化加密
        if encryption_key:
            self._setup_encryption(encryption_key)

        # 加载配置
        self.load_config()
        
        # 启用热重载
        if enable_hot_reload and self.config_path:
            self._setup_hot_reload()

        # 记录配置加载审计日志
        log_audit(
            action="config_initialize",
            user="system",
            resource="config_manager",
            status="success" if self.config else "failure",
            details={"config_path": self.config_path, "encrypted": bool(encryption_key)}
        )

    def _setup_encryption(self, key: str):
        """设置加密"""
        try:
            # 使用common.py中的编码常量
            if len(key) < 32:
                key_hash = hashlib.sha256(key.encode(DEFAULT_ENCODING)).digest()
                key = base64.urlsafe_b64encode(key_hash)
            else:
                key = key.encode(DEFAULT_ENCODING)

            self._encryption_cipher = Fernet(key)
            self.metadata.encrypted = True
            logger.info("配置加密已启用")

        except Exception as e:
            raise ConfigEncryptionError(f"加密设置失败: {e}")

    def _expand_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """展开环境变量 - 支持${VAR_NAME}格式"""
        def expand_recursive(obj):
            if isinstance(obj, dict):
                return {k: expand_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                env_value = os.getenv(var_name)
                if env_value is not None:
                    logger.debug(f"展开环境变量: {var_name} = {env_value}")
                    return env_value
                else:
                    logger.warning(f"环境变量未设置: {var_name}，保持原值")
                    return obj
            return obj
        
        return expand_recursive(config)

    def _setup_hot_reload(self):
        """设置配置热重载 - 监控文件变化自动重载"""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager
                    self.last_modified = 0
                    self.debounce_interval = 2.0  # 防抖间隔（秒）

                def on_modified(self, event):
                    if event.src_path == self.config_manager.config_path:
                        current_time = time.time()
                        if current_time - self.last_modified < self.debounce_interval:
                            return
                        self.last_modified = current_time
                        # 统一处理配置变更
                        self.config_manager._handle_config_change(event.src_path)

            observer = Observer()
            event_handler = ConfigFileHandler(self)
            watch_dir = os.path.dirname(os.path.abspath(self.config_path))
            observer.schedule(event_handler, watch_dir, recursive=False)
            observer.start()
            self._file_observer = observer
            logger.info(f"配置热重载已启用，监控目录: {watch_dir}")

        except ImportError:
            logger.warning("watchdog未安装，配置热重载功能不可用")
        except Exception as e:
            logger.error(f"配置热重载设置失败: {e}")

    def _handle_config_change(self, changed_path: str):
        """统一处理配置文件变更（热重载回调）"""
        try:
            if changed_path != self.config_path:
                return
            old_config = copy.deepcopy(self.config)
            success = self.load_config()
            if success:
                # 通知全局观察者（监听所有变更）
                self._notify_observers(None, old_config, self.config)
                logger.info(f"配置文件热重载完成: {changed_path}")
            else:
                logger.warning(f"配置文件热重载失败，已回退默认配置: {changed_path}")
        except Exception as e:
            logger.error(f"处理配置文件变更时出错: {e}")

    def load_config(self) -> bool:
        """加载配置"""
        with self._lock:
            try:
                if not self.config_path:
                    logger.warning(WARNING_CONFIG_DEFAULT)
                    self.config = copy.deepcopy(self.DEFAULT_CONFIG)
                    self.metadata.source = ConfigSource.DEFAULT
                    return True

                if not os.path.exists(self.config_path):
                    logger.warning(f"配置文件不存在: {self.config_path}，将使用默认配置")
                    self.config = copy.deepcopy(self.DEFAULT_CONFIG)
                    self.metadata.source = ConfigSource.DEFAULT
                    return True

                # 根据文件扩展名确定格式
                file_ext = Path(self.config_path).suffix.lower()
                if file_ext == '.json':
                    self.metadata.format = ConfigFormat.JSON
                    self._load_json_config()
                elif file_ext in ['.yaml', '.yml']:
                    self.metadata.format = ConfigFormat.YAML
                    self._load_yaml_config()
                elif file_ext == '.toml':
                    self.metadata.format = ConfigFormat.TOML
                    self._load_toml_config()
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")

                # 验证配置
                if self.validate_on_load:
                    self.validate_config()

                # 计算校验和
                self.metadata.checksum = self._calculate_checksum()
                self.metadata.source = ConfigSource.FILE
                self.metadata.updated_at = datetime.now().isoformat()

                logger.info(SUCCESS_CONFIG_LOAD)
                log_performance("config_load", 0.0, True, {"config_path": self.config_path})
                return True

            except Exception as e:
                logger.error(f"{ERROR_CONFIG_LOAD}: {e}")
                log_error("config_load_error", str(e), {"config_path": self.config_path})

                # 回退到默认配置
                self.config = copy.deepcopy(self.DEFAULT_CONFIG)
                self.metadata.source = ConfigSource.DEFAULT
                return False

    def _load_json_config(self):
        """加载JSON配置"""
        try:
            with open(self.config_path, 'r', encoding=DEFAULT_ENCODING) as f:
                raw_config = json.load(f)

            if self.metadata.encrypted and self._encryption_cipher:
                self.config = self._decrypt_config(raw_config)
            else:
                self.config = raw_config
            
            # 展开环境变量
            self.config = self._expand_environment_variables(self.config)

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON配置文件格式错误: {e}")

    def _load_yaml_config(self):
        """加载YAML配置"""
        try:
            with open(self.config_path, 'r', encoding=DEFAULT_ENCODING) as f:
                raw_config = yaml.safe_load(f)

            if self.metadata.encrypted and self._encryption_cipher:
                self.config = self._decrypt_config(raw_config)
            else:
                self.config = raw_config
            
            # 展开环境变量
            self.config = self._expand_environment_variables(self.config)

        except yaml.YAMLError as e:
            raise ValueError(f"YAML配置文件格式错误: {e}")

    def _load_toml_config(self):
        """加载TOML配置"""
        try:
            with open(self.config_path, 'r', encoding=DEFAULT_ENCODING) as f:
                raw_config = toml.load(f)

            if self.metadata.encrypted and self._encryption_cipher:
                self.config = self._decrypt_config(raw_config)
            else:
                self.config = raw_config
            
            # 展开环境变量
            self.config = self._expand_environment_variables(self.config)

        except toml.TomlDecodeError as e:
            raise ValueError(f"TOML配置文件格式错误: {e}")

    def _encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """加密配置"""
        if not self._encryption_cipher:
            return config

        try:
            # 使用common.py中的序列化函数
            config_str = serialize_dict(config)
            encrypted_data = self._encryption_cipher.encrypt(config_str.encode(DEFAULT_ENCODING))
            return {
                '_encrypted': True,
                'data': encrypted_data.decode(DEFAULT_ENCODING)
            }
        except Exception as e:
            raise ConfigEncryptionError(f"配置加密失败: {e}")

    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密配置"""
        if not self._encryption_cipher or not config.get('_encrypted'):
            return config

        try:
            encrypted_data = config['data']
            decrypted_data = self._encryption_cipher.decrypt(encrypted_data.encode(DEFAULT_ENCODING))
            return deserialize_dict(decrypted_data.decode(DEFAULT_ENCODING))
        except Exception as e:
            raise ConfigEncryptionError(f"配置解密失败: {e}")

    def save_config(self, filepath: Optional[str] = None,
                    format: Optional[ConfigFormat] = None) -> bool:
        """保存配置到文件"""
        with self._lock:
            try:
                save_path = filepath or self.config_path
                if not save_path:
                    raise ValueError("未指定保存路径")

                save_format = format or self.metadata.format

                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 创建备份
                if os.path.exists(save_path):
                    backup_path = f"{save_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy2(save_path, backup_path)
                    logger.info(f"创建配置备份: {backup_path}")

                # 准备要保存的配置
                save_data = self.config
                if self.metadata.encrypted and self._encryption_cipher:
                    save_data = self._encrypt_config(self.config)

                # 根据格式保存
                if save_format == ConfigFormat.JSON:
                    with open(save_path, 'w', encoding=DEFAULT_ENCODING) as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False, cls=DeepSeekQuantEncoder)
                elif save_format == ConfigFormat.YAML:
                    with open(save_path, 'w', encoding=DEFAULT_ENCODING) as f:
                        yaml.safe_dump(save_data, f, default_flow_style=False, allow_unicode=True)
                elif save_format == ConfigFormat.TOML:
                    with open(save_path, 'w', encoding=DEFAULT_ENCODING) as f:
                        toml.dump(save_data, f)
                else:
                    raise ValueError(f"不支持的保存格式: {save_format}")

                # 更新元数据
                self.metadata.updated_at = datetime.now().isoformat()
                self.metadata.checksum = self._calculate_checksum()

                # 记录审计日志
                log_audit(
                    action="config_save",
                    user="system",
                    resource="config_manager",
                    status="success",
                    details={"file_path": save_path, "format": save_format.value}
                )

                logger.info("配置保存成功")
                return True

            except Exception as e:
                logger.error(f"配置保存失败: {e}")
                log_error("config_save_error", str(e), {"file_path": filepath})
                return False

    def validate_config(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置是否符合架构"""
        with self._lock:
            try:
                # 先验证完整性，再验证jsonschema
                self._validate_schema_completeness()
                validation_schema = schema or self.DEFAULT_CONFIG_SCHEMA
                jsonschema.validate(instance=self.config, schema=validation_schema)
                self._validate_custom_rules()

                logger.info(SUCCESS_CONFIG_VALIDATION)
                return True

            except jsonschema.ValidationError as e:
                # 更友好的验证错误提示
                error_path = '.'.join(str(p) for p in e.path) if e.path else '根配置'
                logger.error(f"{ERROR_CONFIG_VALIDATION}: 路径: {error_path}, 错误: {e.message}")
                raise ConfigValidationError(f"配置验证失败 [{error_path}]: {e.message}")
            except Exception as e:
                logger.error(f"配置验证异常: {e}")
                raise ConfigValidationError(f"配置验证异常: {e}")

    def _validate_schema_completeness(self):
        """验证配置架构定义的完整性"""
        required_sections = ['system', 'data_sources']
        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"缺少必需配置段: {section}")
        
        # 验证system段的必需字段
        system_required = ['name', 'version', 'environment', 'log_level']
        system_config = self.config.get('system', {})
        for field in system_required:
            if field not in system_config:
                raise ConfigValidationError(f"system配置段缺少必需字段: {field}")

    def _validate_custom_rules(self):
        """自定义验证规则 - 使用common.py中的枚举验证"""
        # 验证枚举值
        system_config = self.config.get('system', {})
        if 'trading_mode' in system_config:
            if not validate_enum_value(TradingMode, system_config['trading_mode']):
                raise ConfigValidationError(f"无效的交易模式: {system_config['trading_mode']}")

        # 验证风险等级
        risk_config = self.config.get('risk_management', {})
        if 'risk_level' in risk_config:
            if not validate_enum_value(RiskLevel, risk_config['risk_level']):
                raise ConfigValidationError(f"无效的风险等级: {risk_config['risk_level']}")

        # 验证数值范围
        trading_config = self.config.get('trading', {})
        if trading_config:
            if trading_config.get('initial_capital', 0) <= 0:
                raise ConfigValidationError("初始资金必须大于0")

            if not 0 <= trading_config.get('max_position_size', 0) <= 1:
                raise ConfigValidationError("最大仓位比例必须在0-1之间")

    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """获取配置值 - 带缓存优化"""
        with self._lock:
            if key is None:
                return copy.deepcopy(self.config)

            # 尝试从缓存获取
            cache_key = f"{key}_{self.metadata.checksum}"
            if cache_key in self._value_cache:
                logger.debug(f"从缓存获取配置: {key}")
                return copy.deepcopy(self._value_cache[cache_key])

            try:
                keys = key.split('.')
                value = self.config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                
                # 存入缓存
                self._value_cache[cache_key] = copy.deepcopy(value)
                return copy.deepcopy(value)
            except (KeyError, TypeError, AttributeError):
                return default

    def set_config(self, key: str, value: Any, reason: str = "") -> bool:
        """设置配置值"""
        with self._lock:
            # 增强的键验证
            if not key or not isinstance(key, str) or key.strip() == "":
                logger.error(f"无效的配置键: {key}")
                return False
            
            if '.' in key and (key.startswith('.') or key.endswith('.') or '..' in key):
                logger.error(f"配置键格式错误: {key}")
                return False
                
            try:
                keys = key.split('.')
                current = self.config
                old_value = self.get_config(key)

                # 遍历到最后一个键的父级
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                # 设置新值
                current[keys[-1]] = value
                
                # 清除缓存
                self._value_cache.clear()

                # 记录变更和审计日志
                self._record_change('update', 'user', reason or f"设置配置 {key}",
                                    old_value, value, key)

                # 通知观察者
                self._notify_observers(key, old_value, value)

                log_audit(
                    action="config_update",
                    user="user",
                    resource="config_manager",
                    status="success",
                    details={"key": key, "old_value": str(old_value), "new_value": str(value), "reason": reason}
                )

                logger.debug(f"配置已更新: {key} = {value}")
                return True

            except Exception as e:
                logger.error(f"配置设置失败: {e}")
                log_error("config_set_error", str(e), {"key": key, "value": str(value)})
                return False

    def update_config(self, new_config: Dict[str, Any], reason: str = "") -> bool:
        """更新整个配置"""
        with self._lock:
            try:
                old_config = copy.deepcopy(self.config)

                # 深度合并配置
                def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
                    for key, value in update.items():
                        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                            base[key] = deep_update(base[key], value)
                        else:
                            base[key] = value
                    return base

                self.config = deep_update(self.config, new_config)

                # 验证配置
                if self.validate_on_load:
                    self.validate_config()

                # 记录变更
                self._record_change('update', 'system', reason or "批量更新配置",
                                    old_config, self.config)

                # 通知所有观察者
                self._notify_observers(None, old_config, self.config)

                logger.info("配置批量更新成功")
                return True

            except Exception as e:
                logger.error(f"配置批量更新失败: {e}")
                # 恢复旧配置
                self.config = old_config
                return False

    def _record_change(self, change_type: str, changed_by: str, reason: str,
                       old_value: Any, new_value: Any, key: Optional[str] = None):
        """记录配置变更"""
        change_record = ConfigChange(
            timestamp=datetime.now().isoformat(),
            change_type=change_type,
            changed_by=changed_by,
            changes={key: {'old': old_value, 'new': new_value}} if key else {'full_config': True},
            previous_value=old_value,
            new_value=new_value,
            reason=reason,
            version_before=self.metadata.version,
            version_after=self.metadata.version
        )

        self.change_history.append(asdict(change_record))
        self.metadata.change_history.append(asdict(change_record))

        # 保持变更历史长度
        max_history = self.config.get('system', {}).get('max_change_history', 1000)
        if len(self.change_history) > max_history:
            self.change_history = self.change_history[-max_history:]
        if len(self.metadata.change_history) > max_history:
            self.metadata.change_history = self.metadata.change_history[-max_history:]

    def _calculate_checksum(self) -> str:
        """计算配置校验和"""
        try:
            config_str = serialize_dict(self.config)
            return hashlib.sha256(config_str.encode(DEFAULT_ENCODING)).hexdigest()
        except Exception as e:
            logger.error(f"校验和计算失败: {e}")
            return ""

    def register_observer(self, key: str, callback: Callable,
                          observer_id: Optional[str] = None) -> str:
        """注册配置变更观察者"""
        with self._lock:
            obs_id = observer_id or f"observer_{hash(callback)}_{len(self._observers[key])}"
            self._observers[key].append((obs_id, callback))
            logger.debug(f"注册配置观察者: {key} -> {obs_id}")
            return obs_id

    def unregister_observer(self, key: str, observer_id: str) -> bool:
        """取消注册观察者"""
        with self._lock:
            if key in self._observers:
                original_count = len(self._observers[key])
                self._observers[key] = [(oid, cb) for oid, cb in self._observers[key] if oid != observer_id]
                success = len(self._observers[key]) < original_count
                if success:
                    logger.debug(f"取消注册观察者: {key} -> {observer_id}")
                return success
            return False

    def _notify_observers(self, key: Optional[str], old_value: Any, new_value: Any):
        """通知观察者配置变更"""
        try:
            # 通知特定键的观察者
            if key and key in self._observers:
                for observer_id, callback in self._observers[key]:
                    try:
                        callback(key, old_value, new_value)
                        logger.debug(f"通知观察者 {observer_id}: {key}")
                    except Exception as e:
                        logger.error(f"观察者通知失败 {observer_id}: {e}")

            # 通知全局观察者（监听所有变更）
            if None in self._observers:
                for observer_id, callback in self._observers[None]:
                    try:
                        callback(key, old_value, new_value)
                        logger.debug(f"通知全局观察者 {observer_id}")
                    except Exception as e:
                        logger.error(f"全局观察者通知失败 {observer_id}: {e}")

        except Exception as e:
            logger.error(f"观察者通知系统错误: {e}")

    def _validate_config_value(self, key: str, value: Any) -> bool:
        """
        验证配置值的有效性
        
        Args:
            key: 配置键
            value: 配置值
            
        Returns:
            bool: 是否有效
        """
        # 基础类型验证
        if key in ['max_threads', 'processing_timeout', 'retry_attempts']:
            if not isinstance(value, int) or value <= 0:
                logger.warning(f"配置值无效 {key}: {value}, 应该为正整数")
                return False
        
        # 布尔类型验证
        if key in ['enabled', 'performance_monitoring', 'auto_recovery']:
            if not isinstance(value, bool):
                logger.warning(f"配置值无效 {key}: {value}, 应该为布尔值")
                return False
        
        # 数值范围验证
        if key == 'max_position_size':
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                logger.warning(f"配置值无效 {key}: {value}, 应该在 0-1 之间")
                return False
        
        if key == 'initial_capital':
            if not isinstance(value, (int, float)) or value <= 0:
                logger.warning(f"配置值无效 {key}: {value}, 应该大于 0")
                return False
        
        # 枚举类型验证
        if key == 'trading_mode':
            if not validate_enum_value(TradingMode, value):
                logger.warning(f"配置值无效 {key}: {value}, 不是有效的交易模式")
                return False
        
        if key == 'risk_level':
            if not validate_enum_value(RiskLevel, value):
                logger.warning(f"配置值无效 {key}: {value}, 不是有效的风险等级")
                return False
        
        return True

    def merge_config(self, other_config: Dict[str, Any], reason: str = "") -> bool:
        """合并另一个配置到当前配置 - 增强版本带配置验证"""
        with self._lock:
            try:
                old_config = copy.deepcopy(self.config)

                # 改进的深度合并配置，支持列表合并和配置验证
                def deep_merge_with_validation(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
                    for key, value in update.items():
                        if key in base:
                            # 处理列表合并：去重后追加
                            if isinstance(base[key], list) and isinstance(value, list):
                                base[key] = base[key] + [x for x in value if x not in base[key]]
                            # 处理字典递归合并
                            elif isinstance(base[key], dict) and isinstance(value, dict):
                                base[key] = deep_merge_with_validation(base[key], value)
                            # 其他类型：验证后覆盖
                            else:
                                if self._validate_config_value(key, value):
                                    base[key] = value
                                else:
                                    logger.warning(f"跳过无效配置值: {key} = {value}")
                        else:
                            # 新键：验证后添加
                            if self._validate_config_value(key, value):
                                base[key] = value
                            else:
                                logger.warning(f"跳过无效配置值: {key} = {value}")
                    return base

                self.config = deep_merge_with_validation(self.config, other_config)

                # 验证合并后的配置
                if self.validate_on_load:
                    self.validate_config()

                # 记录变更
                self._record_change('merge', 'system', reason or "合并配置",
                                    old_config, self.config)

                logger.info("配置合并成功")
                return True

            except Exception as e:
                logger.error(f"配置合并失败: {e}")
                # 恢复旧配置
                self.config = old_config
                return False

    def reset_to_default(self, section: Optional[str] = None) -> bool:
        """重置配置到默认值"""
        with self._lock:
            try:
                if section:
                    # 重置特定部分
                    if section in self.DEFAULT_CONFIG:
                        old_value = self.config.get(section, {})
                        self.config[section] = copy.deepcopy(self.DEFAULT_CONFIG[section])
                        self._record_change('reset', 'system', f"重置配置部分: {section}",
                                            old_value, self.config[section], section)
                    else:
                        raise ValueError(f"无效的配置部分: {section}")
                else:
                    # 重置整个配置
                    old_config = copy.deepcopy(self.config)
                    self.config = copy.deepcopy(self.DEFAULT_CONFIG)
                    self._record_change('reset', 'system', "重置整个配置",
                                        old_config, self.config)

                logger.info(f"配置重置成功: {section or '全部'}")
                return True

            except Exception as e:
                logger.error(f"配置重置失败: {e}")
                return False

    def create_snapshot(self, description: str = "") -> str:
        """创建配置快照"""
        with self._lock:
            try:
                snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

                snapshot = {
                    'id': snapshot_id,
                    'timestamp': datetime.now().isoformat(),
                    'description': description,
                    'config': copy.deepcopy(self.config),
                    'metadata': asdict(self.metadata),
                    'checksum': self._calculate_checksum()
                }

                # 保存快照
                snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'config/snapshots')
                os.makedirs(snapshots_dir, exist_ok=True)

                snapshot_file = os.path.join(snapshots_dir, f"{snapshot_id}.json")
                with open(snapshot_file, 'w', encoding=DEFAULT_ENCODING) as f:
                    json.dump(snapshot, f, indent=2, ensure_ascii=False, cls=DeepSeekQuantEncoder)

                logger.info(f"配置快照创建成功: {snapshot_id}")
                return snapshot_id

            except Exception as e:
                logger.error(f"配置快照创建失败: {e}")
                raise

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """从快照恢复配置"""
        with self._lock:
            try:
                snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'config/snapshots')
                snapshot_file = os.path.join(snapshots_dir, f"{snapshot_id}.json")

                if not os.path.exists(snapshot_file):
                    raise FileNotFoundError(f"快照不存在: {snapshot_id}")

                with open(snapshot_file, 'r', encoding=DEFAULT_ENCODING) as f:
                    snapshot = json.load(f)

                # 验证快照完整性
                if snapshot['checksum'] != self._calculate_checksum_for_config(snapshot['config']):
                    raise ValueError("快照校验和验证失败")

                # 恢复配置
                old_config = copy.deepcopy(self.config)
                self.config = snapshot['config']

                # 更新元数据
                self.metadata = ConfigMetadata(**snapshot['metadata'])

                # 记录变更
                self._record_change('restore', 'system', f"从快照恢复: {snapshot_id}",
                                    old_config, self.config)

                logger.info(f"配置从快照恢复成功: {snapshot_id}")
                return True

            except Exception as e:
                logger.error(f"配置快照恢复失败: {e}")
                return False

    def _calculate_checksum_for_config(self, config: Dict[str, Any]) -> str:
        """计算指定配置的校验和"""
        config_str = serialize_dict(config)
        return hashlib.sha256(config_str.encode(DEFAULT_ENCODING)).hexdigest()

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有快照"""
        try:
            snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'config/snapshots')
            if not os.path.exists(snapshots_dir):
                return []

            snapshots = []
            for filename in os.listdir(snapshots_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(snapshots_dir, filename)
                    try:
                        with open(filepath, 'r', encoding=DEFAULT_ENCODING) as f:
                            snapshot = json.load(f)
                            snapshots.append({
                                'id': snapshot.get('id', ''),
                                'timestamp': snapshot.get('timestamp', ''),
                                'description': snapshot.get('description', ''),
                                'checksum': snapshot.get('checksum', ''),
                                'file_path': filepath
                            })
                    except Exception as e:
                        logger.warning(f"读取快照文件失败 {filename}: {e}")

            # 按时间戳排序（最新的在前）
            snapshots.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return snapshots

        except Exception as e:
            logger.error(f"列出快照失败: {e}")
            return []

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除配置快照"""
        try:
            snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'config/snapshots')
            snapshot_file = os.path.join(snapshots_dir, f"{snapshot_id}.json")

            if not os.path.exists(snapshot_file):
                logger.warning(f"快照不存在: {snapshot_id}")
                return False

            os.remove(snapshot_file)
            logger.info(f"快照删除成功: {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"快照删除失败: {e}")
            return False

    def cleanup_old_snapshots(self, max_age_days: int = 30, max_count: int = 100) -> int:
        """清理旧快照"""
        try:
            snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'config/snapshots')
            if not os.path.exists(snapshots_dir):
                return 0

            snapshots = []
            for filename in os.listdir(snapshots_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(snapshots_dir, filename)
                    file_age = time.time() - os.path.getmtime(filepath)
                    snapshots.append((filepath, file_age))

            # 按年龄排序
            snapshots.sort(key=lambda x: x[1], reverse=True)

            deleted_count = 0
            cutoff_time = max_age_days * 24 * 3600  # 转换为秒

            for filepath, file_age in snapshots:
                if file_age > cutoff_time or deleted_count < len(snapshots) - max_count:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.debug(f"删除旧快照: {filepath}")
                    except Exception as e:
                        logger.error(f"删除快照失败 {filepath}: {e}")

            logger.info(f"快照清理完成: 删除 {deleted_count} 个旧快照")
            return deleted_count

        except Exception as e:
            logger.error(f"快照清理失败: {e}")
            return 0

    def get_change_history(self, limit: int = 100,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           change_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取配置变更历史"""
        with self._lock:
            filtered_history = copy.deepcopy(self.change_history)

            # 时间过滤
            if start_time:
                filtered_history = [ch for ch in filtered_history if ch['timestamp'] >= start_time]
            if end_time:
                filtered_history = [ch for ch in filtered_history if ch['timestamp'] <= end_time]

            # 类型过滤
            if change_type:
                filtered_history = [ch for ch in filtered_history if ch['change_type'] == change_type]

            # 限制结果数量
            return filtered_history[-limit:] if limit else filtered_history

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        with self._lock:
            return {
                'total_sections': len(self.config),
                'total_keys': self._count_keys(self.config),
                'last_updated': self.metadata.updated_at,
                'checksum': self.metadata.checksum,
                'format': self.metadata.format.value,
                'source': self.metadata.source.value,
                'encrypted': self.metadata.encrypted,
                'change_count': len(self.change_history),
                'snapshot_count': len(self.list_snapshots()),
                'observer_count': sum(len(observers) for observers in self._observers.values())
            }

    def _count_keys(self, config: Dict[str, Any]) -> int:
        """计算配置键的数量"""
        count = 0
        for key, value in config.items():
            if isinstance(value, dict):
                count += 1 + self._count_keys(value)
            else:
                count += 1
        return count

    def get_config_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """比较配置差异"""

        def find_diff(dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> Dict[str, Any]:
            diff = {}
            all_keys = set(dict1.keys()) | set(dict2.keys())

            for key in all_keys:
                current_path = f"{path}.{key}" if path else key

                if key not in dict1:
                    diff[current_path] = {'action': 'add', 'value': dict2[key]}
                elif key not in dict2:
                    diff[current_path] = {'action': 'remove', 'value': dict1[key]}
                elif dict1[key] != dict2[key]:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        nested_diff = find_diff(dict1[key], dict2[key], current_path)
                        diff.update(nested_diff)
                    else:
                        diff[current_path] = {
                            'action': 'modify',
                            'old_value': dict1[key],
                            'new_value': dict2[key]
                        }
            return diff

        return find_diff(self.config, other_config)

    def validate_config_value(self, key: str, value: Any) -> bool:
        """验证单个配置值"""
        try:
            # 获取配置架构
            schema = self._get_field_schema(key)
            if not schema:
                return True  # 没有架构定义，默认通过

            # 使用jsonschema验证
            jsonschema.validate(value, schema)
            return True

        except jsonschema.ValidationError as e:
            logger.warning(f"配置值验证失败 {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"配置值验证错误 {key}: {e}")
            return False

    def _get_field_schema(self, key: str) -> Optional[Dict[str, Any]]:
        """获取字段的架构定义"""
        try:
            # 支持点分隔的键路径
            keys = key.split('.')
            schema = self.DEFAULT_CONFIG_SCHEMA

            for k in keys:
                if 'properties' in schema and k in schema['properties']:
                    schema = schema['properties'][k]
                else:
                    return None

            return schema
        except Exception:
            return None

    def generate_config_template(self, filepath: str,
                                 format: ConfigFormat = ConfigFormat.JSON,
                                 include_comments: bool = True) -> bool:
        """生成配置模板"""
        try:
            template = copy.deepcopy(self.DEFAULT_CONFIG)

            if include_comments:
                # 添加注释（JSON格式不支持注释，所以使用YAML格式）
                if format == ConfigFormat.YAML:
                    template = self._add_config_comments(template)

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 根据格式生成模板
            if format == ConfigFormat.JSON:
                with open(filepath, 'w', encoding=DEFAULT_ENCODING) as f:
                    json.dump(template, f, indent=2, ensure_ascii=False, cls=DeepSeekQuantEncoder)
            elif format == ConfigFormat.YAML:
                with open(filepath, 'w', encoding=DEFAULT_ENCODING) as f:
                    yaml.safe_dump(template, f, default_flow_style=False, allow_unicode=True)
            elif format == ConfigFormat.TOML:
                with open(filepath, 'w', encoding=DEFAULT_ENCODING) as f:
                    toml.dump(template, f)
            else:
                raise ValueError(f"不支持的模板格式: {format}")

            logger.info(f"配置模板生成成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"配置模板生成失败: {e}")
            return False

    def _add_config_comments(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """为配置添加注释"""
        # 这里实现配置注释的添加逻辑
        comments = {
            'system': {
                'name': '系统名称',
                'version': '系统版本',
                'environment': '运行环境 (development, testing, staging, production)',
                'log_level': '日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
                'max_memory_mb': '最大内存使用限制 (MB)',
                'max_threads': '最大线程数',
                'auto_recovery': '是否启用自动恢复',
                'trading_mode': f'交易模式 ({", ".join([mode.value for mode in TradingMode])})'
            },
            'data_sources': {
                'primary': '主要数据源',
                'fallback_sources': '备用数据源列表',
                'cache_enabled': '是否启用数据缓存',
                'cache_size_mb': '缓存大小 (MB)',
                'request_timeout': '请求超时时间 (秒)',
                'retry_attempts': '重试次数'
            }
        }

        # 在实际实现中，这里会将注释转换为YAML注释格式
        return config

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        with self._lock:
            # 检查配置完整性
            config_valid = True
            try:
                self.validate_config()
            except ConfigValidationError:
                config_valid = False

            # 检查文件系统
            file_system_ok = True
            if self.config_path:
                try:
                    with open(self.config_path, 'r') as f:
                        f.read(1)  # 尝试读取一个字节
                except Exception:
                    file_system_ok = False

            return {
                'status': 'healthy' if config_valid and file_system_ok else 'unhealthy',
                'config_valid': config_valid,
                'file_system_ok': file_system_ok,
                'encryption_enabled': self.metadata.encrypted,
                'total_changes': len(self.change_history),
                'snapshots_count': len(self.list_snapshots()),
                'observers_count': sum(len(observers) for observers in self._observers.values()),
                'last_updated': self.metadata.updated_at,
                'uptime_seconds': (
                            datetime.now() - datetime.fromisoformat(self.metadata.created_at)).total_seconds()
            }

    def reload_config(self, new_config_path: Optional[str] = None) -> bool:
        """重新加载配置"""
        with self._lock:
            try:
                if new_config_path:
                    self.config_path = new_config_path

                return self.load_config()

            except Exception as e:
                logger.error(f"配置重新加载失败: {e}")
                return False

    def export_config(self, include_metadata: bool = False) -> Dict[str, Any]:
        """导出配置数据"""
        with self._lock:
            try:
                export_data = {
                    'config': copy.deepcopy(self.config)
                }

                if include_metadata:
                    export_data['metadata'] = asdict(self.metadata)
                    export_data['metadata']['change_history'] = self.change_history[-10:]  # 最近10个变更

                logger.info("配置导出成功")
                return export_data

            except Exception as e:
                logger.error(f"配置导出失败: {e}")
                raise

    def import_config(self, import_data: Dict[str, Any]) -> bool:
        """导入配置数据"""
        with self._lock:
            try:
                if 'config' not in import_data:
                    raise ValueError("导入数据必须包含config字段")

                return self.update_config(import_data['config'], "导入配置")

            except Exception as e:
                logger.error(f"配置导入失败: {e}")
                return False

    def encrypt_sensitive_data(self) -> bool:
        """加密敏感配置数据 - 使用真正的加密"""
        with self._lock:
            try:
                if not self._encryption_cipher:
                    raise ConfigEncryptionError("加密未启用")

                # 定义敏感字段
                sensitive_fields = [
                    'api.authentication.jwt_secret',
                    'alerts.email.password',
                    'alerts.sms.auth_token',
                    'alerts.sms.account_sid',
                    'database.password',
                    'encryption_key'
                ]

                encrypted_count = 0
                for field in sensitive_fields:
                    value = self.get_config(field)
                    if value and isinstance(value, str) and not value.startswith('_enc_'):
                        # 使用真正的加密
                        encrypted_bytes = self._encryption_cipher.encrypt(value.encode(DEFAULT_ENCODING))
                        encrypted_value = '_enc_' + base64.b64encode(encrypted_bytes).decode(DEFAULT_ENCODING)
                        self.set_config(field, encrypted_value, "加密敏感数据")
                        encrypted_count += 1

                logger.info(f"敏感数据加密完成: {encrypted_count} 个字段已加密")
                return True

            except Exception as e:
                logger.error(f"敏感数据加密失败: {e}")
                return False

    def decrypt_sensitive_data(self) -> bool:
        """解密敏感配置数据 - 使用真正的解密"""
        with self._lock:
            try:
                if not self._encryption_cipher:
                    raise ConfigEncryptionError("加密未启用")

                # 定义敏感字段
                sensitive_fields = [
                    'api.authentication.jwt_secret',
                    'alerts.email.password',
                    'alerts.sms.auth_token',
                    'alerts.sms.account_sid',
                    'database.password',
                    'encryption_key'
                ]

                decrypted_count = 0
                for field in sensitive_fields:
                    value = self.get_config(field)
                    if value and isinstance(value, str) and value.startswith('_enc_'):
                        try:
                            # 使用真正的解密
                            encrypted_data = base64.b64decode(value[5:])  # 移除'_enc_'前缀
                            decrypted_bytes = self._encryption_cipher.decrypt(encrypted_data)
                            decrypted_value = decrypted_bytes.decode(DEFAULT_ENCODING)
                            self.set_config(field, decrypted_value, "解密敏感数据")
                            decrypted_count += 1
                        except Exception as e:
                            logger.warning(f"无法解密字段 {field}: {e}")
                            continue

                logger.info(f"敏感数据解密完成: {decrypted_count} 个字段已解密")
                return True

            except Exception as e:
                logger.error(f"敏感数据解密失败: {e}")
                return False

    def cleanup(self):
        """清理资源"""
        try:
            # 停止热重载观察者
            if self._file_observer:
                self._file_observer.stop()
                self._file_observer.join(timeout=2)
                logger.info("配置热重载观察者已停止")

            # 保存当前配置
            if self.config_path:
                self.save_config()

            # 清理观察者
            self._observers.clear()

            # 清理快照
            self.cleanup_old_snapshots()

            logger.info("配置管理器清理完成")
            log_audit(
                action="config_cleanup",
                user="system",
                resource="config_manager",
                status="success",
                details={}
            )

        except Exception as e:
            logger.error(f"配置管理器清理失败: {e}")
            log_error("config_cleanup_error", str(e), {})

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None
_global_config_lock = threading.Lock()

def get_global_config_manager(config_path: Optional[str] = None,
                              encryption_key: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager

    with _global_config_lock:
        if _global_config_manager is None:
            _global_config_manager = ConfigManager(config_path, encryption_key)

        return _global_config_manager

def set_global_config_manager(config_manager: ConfigManager):
    """设置全局配置管理器实例"""
    global _global_config_manager

    with _global_config_lock:
        if _global_config_manager is not None:
            _global_config_manager.cleanup()

        _global_config_manager = config_manager

def shutdown_global_config_manager():
    """关闭全局配置管理器"""
    global _global_config_manager

    with _global_config_lock:
        if _global_config_manager is not None:
            _global_config_manager.cleanup()
            _global_config_manager = None