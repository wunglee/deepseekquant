"""
DeepSeekQuant 配置管理器
负责系统配置的加载、验证、更新和持久化
"""

import json
import yaml
import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import copy
import jsonschema
from dataclasses import dataclass, asdict, field
import toml
import inspect
from enum import Enum
import threading
from collections import defaultdict
import re
import tempfile
import shutil
from cryptography.fernet import Fernet
import base64
import uuid
import time
from typing import Any

logger = logging.getLogger('DeepSeekQuant.ConfigManager')


class ConfigFormat(Enum):
    """配置文件格式枚举"""
    JSON = 'json'
    YAML = 'yaml'
    TOML = 'toml'
    INI = 'ini'


class ConfigSource(Enum):
    """配置来源枚举"""
    FILE = 'file'
    DATABASE = 'database'
    API = 'api'
    ENVIRONMENT = 'environment'
    DEFAULT = 'default'


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
    change_type: str  # 'create', 'update', 'delete', 'rollback'
    changed_by: str
    changes: Dict[str, Any]
    previous_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: str = ""
    version_before: str = ""
    version_after: str = ""


class ConfigManager:
    """配置管理器 - 完整生产实现"""

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
                    "trading_mode": {"type": "string",
                                     "enum": ["paper_trading", "live_trading", "backtesting", "simulation"]}
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
            # 更多配置项的架构定义...
        },
        "required": ["system", "data_sources"]
    }

    # 默认配置
    DEFAULT_CONFIG = {
        "system": {
            "name": "DeepSeekQuant",
            "version": "1.0.0",
            "environment": "development",
            "log_level": "INFO",
            "max_memory_mb": 2048,
            "max_threads": 20,
            "auto_recovery": True,
            "trading_mode": "paper_trading",
            "max_history_size": 10000,
            "performance_monitoring": True,
            "alerting_enabled": True
        },
        "data_sources": {
            "primary": "yahoo_finance",
            "fallback_sources": ["alpha_vantage", "iex_cloud", "polygon"],
            "cache_enabled": True,
            "cache_size_mb": 500,
            "request_timeout": 30,
            "retry_attempts": 3,
            "rate_limiting": {
                "requests_per_minute": 100,
                "burst_limit": 20
            }
        },
        "database": {
            "type": "sqlite",
            "path": "deepseekquant.db",
            "connection_pool_size": 10,
            "query_timeout": 30,
            "backup_enabled": True,
            "backup_interval_hours": 24
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/deepseekquant.log",
            "max_file_size_mb": 100,
            "backup_count": 10,
            "console_output": True,
            "json_format": False
        },
        "monitoring": {
            "enabled": True,
            "interval_seconds": 60,
            "metrics": {
                "system_metrics": True,
                "performance_metrics": True,
                "business_metrics": True
            },
            "alert_channels": ["email", "slack", "sms"],
            "alert_thresholds": {
                "cpu_usage": 90,
                "memory_usage": 85,
                "disk_usage": 90,
                "error_rate": 5
            }
        },
        "api": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8080,
            "cors_enabled": True,
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 1000,
                "burst_limit": 100
            },
            "authentication": {
                "enabled": True,
                "jwt_secret": "change_this_in_production",
                "token_expiry_hours": 24
            }
        },
        "trading": {
            "initial_capital": 1000000.0,
            "max_position_size": 0.1,  # 10% of portfolio
            "max_leverage": 2.0,
            "commission_rate": 0.001,  # 0.1%
            "slippage_rate": 0.0005,  # 0.05%
            "risk_free_rate": 0.02,  # 2%
            "min_trade_size": 1000,
            "max_drawdown_limit": 0.2,  # 20%
            "stop_loss_enabled": True,
            "take_profit_enabled": True
        },
        "risk_management": {
            "max_daily_loss": 0.05,  # 5%
            "max_position_risk": 0.02,  # 2%
            "var_confidence_level": 0.95,
            "stress_testing": {
                "enabled": True,
                "scenarios": ["flash_crash", "liquidity_crisis", "black_swan"]
            },
            "circuit_breakers": {
                "enabled": True,
                "levels": [0.05, 0.1, 0.2]  # 5%, 10%, 20% drawdown
            }
        },
        "strategies": {
            "default": {
                "enabled": True,
                "weight": 0.5,
                "parameters": {
                    "lookback_period": 20,
                    "entry_threshold": 1.5,
                    "exit_threshold": 0.5,
                    "trailing_stop": 0.02
                }
            }
        },
        "execution": {
            "broker": "interactive_brokers",
            "algorithm": "TWAP",
            "slippage_model": "proportional",
            "commission_model": "tiered",
            "min_order_size": 1,
            "max_order_size": 10000,
            "order_timeout_seconds": 300
        },
        "backtesting": {
            "initial_capital": 1000000,
            "commission": 0.001,
            "slippage": 0.0005,
            "start_date": "2010-01-01",
            "end_date": "2023-12-31",
            "benchmark": "SPY",
            "warmup_period": 252  # 1 year
        },
        "optimization": {
            "method": "bayesian",
            "max_iterations": 100,
            "n_trials": 50,
            "cv_folds": 5,
            "scoring_metric": "sharpe_ratio",
            "parallel_processing": True
        },
        "alerts": {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your_email@gmail.com",
                "password": "your_password",
                "recipients": ["admin@example.com"]
            },
            "slack": {
                "enabled": False,
                "webhook_url": "https://hooks.slack.com/services/...",
                "channel": "#alerts"
            },
            "sms": {
                "enabled": False,
                "provider": "twilio",
                "account_sid": "your_account_sid",
                "auth_token": "your_auth_token",
                "from_number": "+1234567890",
                "to_numbers": ["+1234567890"]
            }
        },
        "security": {
            "encryption": {
                "enabled": False,
                "algorithm": "AES",
                "key_length": 256
            },
            "authentication": {
                "required": True,
                "timeout_minutes": 30,
                "max_attempts": 5
            },
            "audit_logging": {
                "enabled": True,
                "retention_days": 365
            }
        }
    }

    def __init__(self, config_path: Optional[str] = None,
                 encryption_key: Optional[str] = None,
                 validate_on_load: bool = True):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            encryption_key: 加密密钥，用于加密敏感配置
            validate_on_load: 是否在加载时验证配置
        """
        self.config_path = config_path
        self.encryption_key = encryption_key
        self.validate_on_load = validate_on_load
        self.config: Dict[str, Any] = {}
        self.metadata = ConfigMetadata()
        self.change_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._observers: Dict[str, List[tuple[str, Callable]]] = defaultdict(list)
        self._encryption_cipher: Optional[Fernet] = None
        self._schema_cache: Dict[str, Any] = {}

        # 初始化加密
        if encryption_key:
            self._setup_encryption(encryption_key)

        # 加载配置
        self.load_config()

    def _setup_encryption(self, key: str):
        """设置加密"""
        try:
            # 确保密钥长度符合要求
            if len(key) < 32:
                # 使用SHA256哈希来生成32字节密钥
                key_hash = hashlib.sha256(key.encode()).digest()
                key = base64.urlsafe_b64encode(key_hash)
            else:
                key = key.encode()

            self._encryption_cipher = Fernet(key)
            self.metadata.encrypted = True
            logger.info("配置加密已启用")

        except Exception as e:
            raise ConfigEncryptionError(f"加密设置失败: {e}")

    def load_config(self) -> bool:
        """加载配置"""
        with self._lock:
            try:
                if not self.config_path:
                    logger.info("使用默认配置")
                    self.config = copy.deepcopy(self.DEFAULT_CONFIG)
                    self.metadata.source = ConfigSource.DEFAULT
                    return True

                if not os.path.exists(self.config_path):
                    logger.warning(f"配置文件不存在: {self.config_path}, 使用默认配置")
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

                logger.info(f"配置加载成功: {self.config_path}")
                return True

            except Exception as e:
                logger.error(f"配置加载失败: {e}")
                # 回退到默认配置
                self.config = copy.deepcopy(self.DEFAULT_CONFIG)
                self.metadata.source = ConfigSource.DEFAULT
                return False

    def _load_json_config(self):
        """加载JSON配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)

            # 解密配置（如果加密）
            if self.metadata.encrypted and self._encryption_cipher:
                self.config = self._decrypt_config(raw_config)
            else:
                self.config = raw_config

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON配置文件格式错误: {e}")

    def _load_yaml_config(self):
        """加载YAML配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            if self.metadata.encrypted and self._encryption_cipher:
                self.config = self._decrypt_config(raw_config)
            else:
                self.config = raw_config

        except yaml.YAMLError as e:
            raise ValueError(f"YAML配置文件格式错误: {e}")

    def _load_toml_config(self):
        """加载TOML配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = toml.load(f)

            if self.metadata.encrypted and self._encryption_cipher:
                self.config = self._decrypt_config(raw_config)
            else:
                self.config = raw_config

        except toml.TomlDecodeError as e:
            raise ValueError(f"TOML配置文件格式错误: {e}")

    def _encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """加密配置"""
        if not self._encryption_cipher:
            return config

        try:
            # 序列化配置为JSON字符串
            config_str = json.dumps(config)
            # 加密字符串
            encrypted_data = self._encryption_cipher.encrypt(config_str.encode())
            # 返回加密后的数据结构
            return {
                '_encrypted': True,
                'data': encrypted_data.decode()
            }
        except Exception as e:
            raise ConfigEncryptionError(f"配置加密失败: {e}")

    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密配置"""
        if not self._encryption_cipher or not config.get('_encrypted'):
            return config

        try:
            encrypted_data = config['data']
            # 解密数据
            decrypted_data = self._encryption_cipher.decrypt(encrypted_data.encode())
            # 反序列化JSON
            return json.loads(decrypted_data.decode())
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

                # 创建备份（如果文件已存在）
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
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                elif save_format == ConfigFormat.YAML:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(save_data, f, default_flow_style=False, allow_unicode=True)
                elif save_format == ConfigFormat.TOML:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        toml.dump(save_data, f)
                else:
                    raise ValueError(f"不支持的保存格式: {save_format}")

                # 更新元数据
                self.metadata.updated_at = datetime.now().isoformat()
                self.metadata.checksum = self._calculate_checksum()

                # 记录变更
                self._record_change('save', 'system',
                                    f"配置保存到 {save_path}",
                                    {}, self.config)

                logger.info(f"配置保存成功: {save_path}")
                return True

            except Exception as e:
                logger.error(f"配置保存失败: {e}")
                return False

    def validate_config(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置是否符合架构"""
        with self._lock:
            try:
                validation_schema = schema or self.DEFAULT_CONFIG_SCHEMA

                # 使用jsonschema验证
                jsonschema.validate(instance=self.config, schema=validation_schema)

                # 自定义验证规则
                self._validate_custom_rules()

                logger.info("配置验证成功")
                return True

            except jsonschema.ValidationError as e:
                logger.error(f"配置验证失败: {e}")
                raise ConfigValidationError(f"配置验证失败: {e}")
            except Exception as e:
                logger.error(f"配置验证异常: {e}")
                raise ConfigValidationError(f"配置验证异常: {e}")

    def _validate_custom_rules(self):
        """自定义验证规则"""
        # 验证交易配置
        trading_config = self.config.get('trading', {})
        if trading_config:
            if trading_config.get('initial_capital', 0) <= 0:
                raise ConfigValidationError("初始资金必须大于0")

            if not 0 <= trading_config.get('max_position_size', 0) <= 1:
                raise ConfigValidationError("最大仓位比例必须在0-1之间")

            if trading_config.get('max_leverage', 1) < 1:
                raise ConfigValidationError("最大杠杆必须大于等于1")

        # 验证风险配置
        risk_config = self.config.get('risk_management', {})
        if risk_config:
            if not 0 <= risk_config.get('max_daily_loss', 0) <= 1:
                raise ConfigValidationError("最大日亏损必须在0-1之间")

            if not 0 <= risk_config.get('max_position_risk', 0) <= 1:
                raise ConfigValidationError("最大单笔风险必须在0-1之间")

        # 验证API配置
        api_config = self.config.get('api', {})
        if api_config and api_config.get('enabled', False):
            if api_config.get('port', 0) not in range(1024, 65536):
                raise ConfigValidationError("API端口必须在1024-65535之间")

    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            if key is None:
                return copy.deepcopy(self.config)

            try:
                # 支持点分隔的键路径
                keys = key.split('.')
                value = self.config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return copy.deepcopy(value)
            except (KeyError, TypeError, AttributeError):
                return default

    def set_config(self, key: str, value: Any, reason: str = "") -> bool:
        """设置配置值"""
        with self._lock:
            try:
                # 支持点分隔的键路径
                keys = key.split('.')
                current = self.config

                # 获取旧值
                old_value = self.get_config(key)

                # 遍历到最后一个键的父级
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                # 设置新值
                current[keys[-1]] = value

                # 记录变更
                self._record_change('update', 'user', reason or f"设置配置 {key}",
                                    old_value, value, key)

                # 通知观察者
                self._notify_observers(key, old_value, value)

                logger.debug(f"配置已更新: {key} = {value}")
                return True

            except Exception as e:
                logger.error(f"配置设置失败: {e}")
                return False

    def update_config(self, new_config: Dict[str, Any], reason: str = "") -> bool:
        """更新整个配置"""
        with self._lock:
            try:
                old_config = copy.deepcopy(self.config)
                self.config = copy.deepcopy(new_config)

                # 验证新配置
                if self.validate_on_load:
                    self.validate_config()

                # 记录变更
                self._record_change('update', 'system', reason or "批量更新配置",
                                    old_config, new_config)

                # 通知所有观察者
                self._notify_observers(None, old_config, new_config)

                logger.info("配置批量更新成功")
                return True

            except Exception as e:
                logger.error(f"配置批量更新失败: {e}")
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
                self._observers[key] = [(oid, cb) for oid, cb in self._observers[key] if oid != observer_id]
                logger.debug(f"取消注册观察者: {key} -> {observer_id}")
                return True
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

        logger.info(f"配置变更记录: {change_type} by {changed_by} - {reason}")

    def _calculate_checksum(self) -> str:
        """计算配置校验和"""
        try:
            config_str = json.dumps(self.config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"校验和计算失败: {e}")
            return ""

    def get_change_history(self, limit: int = 100,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           change_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取配置变更历史"""
        with self._lock:
            filtered_history = self.change_history

            # 时间过滤
            if start_time:
                filtered_history = [ch for ch in filtered_history
                                    if ch['timestamp'] >= start_time]
            if end_time:
                filtered_history = [ch for ch in filtered_history
                                    if ch['timestamp'] <= end_time]

            # 类型过滤
            if change_type:
                filtered_history = [ch for ch in filtered_history
                                    if ch['change_type'] == change_type]

            # 限制结果数量
            return filtered_history[-limit:] if limit else filtered_history

    def rollback_config(self, change_id: Optional[str] = None,
                        timestamp: Optional[str] = None) -> bool:
        """回滚配置到指定版本"""
        with self._lock:
            try:
                if not change_id and not timestamp:
                    raise ValueError("必须提供变更ID或时间戳")

                # 查找要回滚的变更记录
                target_change = None
                for change in reversed(self.change_history):
                    if (change_id and change.get('id') == change_id) or \
                            (timestamp and change['timestamp'] <= timestamp):
                        target_change = change
                        break

                if not target_change:
                    raise ValueError("未找到指定的配置版本")

                # 恢复旧值
                if 'full_config' in target_change['changes']:
                    # 完整配置回滚
                    old_config = target_change['previous_value']
                    self.config = copy.deepcopy(old_config)
                else:
                    # 部分配置回滚
                    for key, change_data in target_change['changes'].items():
                        self.set_config(key, change_data['old'], f"回滚到版本 {target_change.get('id', 'unknown')}")

                # 记录回滚操作
                self._record_change('rollback', 'system',
                                    f"回滚到变更 {target_change.get('id', 'unknown')}",
                                    self.config, target_change['previous_value'])

                logger.info(f"配置回滚成功: {target_change.get('id', 'unknown')}")
                return True

            except Exception as e:
                logger.error(f"配置回滚失败: {e}")
                return False

    def export_config(self, filepath: str,
                      format: Optional[ConfigFormat] = None,
                      include_metadata: bool = False) -> bool:
        """导出配置到文件"""
        with self._lock:
            try:
                export_format = format or self.metadata.format
                export_data = self.config

                if include_metadata:
                    export_data = {
                        'metadata': asdict(self.metadata),
                        'config': self.config
                    }

                # 确保目录存在
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # 根据格式导出
                if export_format == ConfigFormat.JSON:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                elif export_format == ConfigFormat.YAML:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(export_data, f, default_flow_style=False, allow_unicode=True)
                elif export_format == ConfigFormat.TOML:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        toml.dump(export_data, f)
                else:
                    raise ValueError(f"不支持的导出格式: {export_format}")

                logger.info(f"配置导出成功: {filepath}")
                return True

            except Exception as e:
                logger.error(f"配置导出失败: {e}")
                return False

    def import_config(self, filepath: str,
                      merge: bool = False,
                      validate: bool = True) -> bool:
        """从文件导入配置"""
        with self._lock:
            try:
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"文件不存在: {filepath}")

                # 根据文件扩展名确定格式
                file_ext = Path(filepath).suffix.lower()
                if file_ext == '.json':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        imported_data = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        imported_data = yaml.safe_load(f)
                elif file_ext == '.toml':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        imported_data = toml.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")

                # 提取配置数据
                if 'metadata' in imported_data and 'config' in imported_data:
                    config_data = imported_data['config']
                    metadata = imported_data['metadata']
                else:
                    config_data = imported_data
                    metadata = None

                if merge:
                    # 合并配置
                    self._merge_config(config_data)
                else:
                    # 替换配置
                    old_config = copy.deepcopy(self.config)
                    self.config = config_data

                    if validate:
                        self.validate_config()

                    # 记录变更
                    self._record_change('import', 'system',
                                        f"导入配置从 {filepath}",
                                        old_config, self.config)

                # 更新元数据
                if metadata:
                    self.metadata = ConfigMetadata(**metadata)

                logger.info(f"配置导入成功: {filepath}")
                return True

            except Exception as e:
                logger.error(f"配置导入失败: {e}")
                return False

    def _merge_config(self, new_config: Dict[str, Any]):
        """合并配置"""

        def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        old_config = copy.deepcopy(self.config)
        self.config = deep_merge(self.config, new_config)

        # 验证合并后的配置
        if self.validate_on_load:
            self.validate_config()

        # 记录变更
        self._record_change('merge', 'system',
                            "合并配置更新",
                            old_config, self.config)

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
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(template, f, indent=2, ensure_ascii=False)
            elif format == ConfigFormat.YAML:
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(template, f, default_flow_style=False, allow_unicode=True)
            elif format == ConfigFormat.TOML:
                with open(filepath, 'w', encoding='utf-8') as f:
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
                'environment': '运行环境 (development, testing, production)',
                'log_level': '日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
                'max_memory_mb': '最大内存使用限制 (MB)',
                'max_threads': '最大线程数',
                'auto_recovery': '是否启用自动恢复',
                'trading_mode': '交易模式 (paper_trading, live_trading, backtesting, simulation)'
            },
            'data_sources': {
                'primary': '主要数据源',
                'fallback_sources': '备用数据源列表',
                'cache_enabled': '是否启用数据缓存',
                'cache_size_mb': '缓存大小 (MB)',
                'request_timeout': '请求超时时间 (秒)',
                'retry_attempts': '重试次数'
            }
            # 更多配置项的注释...
        }

        # 在实际实现中，这里会将注释转换为YAML注释格式
        return config

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

    def encrypt_sensitive_data(self) -> bool:
        """加密敏感配置数据"""
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
                    if value and not isinstance(value, dict):  # 避免加密已经加密的数据
                        encrypted_value = self._encryption_cipher.encrypt(str(value).encode())
                        self.set_config(field, encrypted_value.decode(), "加密敏感数据")
                        encrypted_count += 1

                logger.info(f"敏感数据加密完成: {encrypted_count} 个字段已加密")
                return True

            except Exception as e:
                logger.error(f"敏感数据加密失败: {e}")
                return False

    def decrypt_sensitive_data(self) -> bool:
        """解密敏感配置数据"""
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
                    if value and isinstance(value, str) and value.startswith('gAAAA'):  # Fernet加密前缀
                        try:
                            decrypted_value = self._encryption_cipher.decrypt(value.encode()).decode()
                            self.set_config(field, decrypted_value, "解密敏感数据")
                            decrypted_count += 1
                        except:
                            # 可能不是加密数据，跳过
                            continue

                logger.info(f"敏感数据解密完成: {decrypted_count} 个字段已解密")
                return True

            except Exception as e:
                logger.error(f"敏感数据解密失败: {e}")
                return False

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
                'sensitive_data_encrypted': self._has_encrypted_sensitive_data(),
                'validation_status': self._get_validation_status()
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

    def _has_encrypted_sensitive_data(self) -> bool:
        """检查是否有加密的敏感数据"""
        sensitive_fields = [
            'api.authentication.jwt_secret',
            'alerts.email.password',
            'alerts.sms.auth_token'
        ]

        for field in sensitive_fields:
            value = self.get_config(field)
            if value and isinstance(value, str) and value.startswith('gAAAA'):
                return True
        return False

    def _get_validation_status(self) -> Dict[str, Any]:
        """获取验证状态"""
        try:
            self.validate_config()
            return {'status': 'valid', 'errors': []}
        except ConfigValidationError as e:
            return {'status': 'invalid', 'errors': [str(e)]}
        except Exception as e:
            return {'status': 'error', 'errors': [str(e)]}

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
                snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'snapshots')
                os.makedirs(snapshots_dir, exist_ok=True)

                snapshot_file = os.path.join(snapshots_dir, f"{snapshot_id}.json")
                with open(snapshot_file, 'w', encoding='utf-8') as f:
                    json.dump(snapshot, f, indent=2, ensure_ascii=False)

                logger.info(f"配置快照创建成功: {snapshot_id}")
                return snapshot_id

            except Exception as e:
                logger.error(f"配置快照创建失败: {e}")
                raise

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """从快照恢复配置"""
        with self._lock:
            try:
                snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'snapshots')
                snapshot_file = os.path.join(snapshots_dir, f"{snapshot_id}.json")

                if not os.path.exists(snapshot_file):
                    raise FileNotFoundError(f"快照不存在: {snapshot_id}")

                with open(snapshot_file, 'r', encoding='utf-8') as f:
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
                self._record_change('restore', 'system',
                                    f"从快照恢复: {snapshot_id}",
                                    old_config, self.config)

                logger.info(f"配置从快照恢复成功: {snapshot_id}")
                return True

            except Exception as e:
                logger.error(f"配置快照恢复失败: {e}")
                return False

    def _calculate_checksum_for_config(self, config: Dict[str, Any]) -> str:
        """计算指定配置的校验和"""
        try:
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()
        except Exception:
            return ""

    def cleanup_old_snapshots(self, max_age_days: int = 30,
                              max_count: int = 100) -> int:
        """清理旧快照"""
        try:
            snapshots_dir = self.config.get('system', {}).get('snapshots_dir', 'snapshots')
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
                    os.remove(filepath)
                    deleted_count += 1
                    logger.debug(f"删除旧快照: {filepath}")

            logger.info(f"快照清理完成: 删除 {deleted_count} 个旧快照")
            return deleted_count

        except Exception as e:
            logger.error(f"快照清理失败: {e}")
            return 0

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def cleanup(self):
        """清理资源"""
        try:
            # 保存当前配置
            if self.config_path:
                self.save_config()

            # 清理观察者
            self._observers.clear()

            # 清理缓存
            self._schema_cache.clear()

            logger.info("配置管理器清理完成")

        except Exception as e:
            logger.error(f"配置管理器清理失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 避免析构函数中的异常

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

if __name__ == "__main__":
    # 测试代码
    import tempfile

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ConfigManager.DEFAULT_CONFIG, f, indent=2)
        test_config_path = f.name

    try:
        # 测试配置管理器
        config_manager = ConfigManager(test_config_path)

        # 测试配置获取
        print("系统名称:", config_manager.get_config('system.name'))
        print("日志级别:", config_manager.get_config('system.log_level'))

        # 测试配置设置
        config_manager.set_config('system.log_level', 'DEBUG', '测试配置更新')
        print("更新后的日志级别:", config_manager.get_config('system.log_level'))

        # 测试配置验证
        print("配置验证:", config_manager.validate_config())

        # 测试配置导出
        export_path = tempfile.mktemp(suffix='.json')
        config_manager.export_config(export_path)
        print("配置导出成功:", export_path)

        # 清理
        os.unlink(export_path)

    finally:
        # 清理临时文件
        os.unlink(test_config_path)