"""
配置管理工具
"""
import glob
from typing import Any, Dict, List, Optional, Union, TypeVar
from dataclasses import dataclass
import logging
import os

from core_bak_refactored.core.share.market.market_config import MarketConfig  # 导入 MarketConfig 类

logger = logging.getLogger('DeepSeekQuant.Infrastructure.ConfigManager')

T = TypeVar('T')


class ConfigExtractor:
    """配置提取器（统一373处.get()调用）"""
    
    @staticmethod
    def get_nested(
        config: Dict[str, Any],
        path: str,
        default: Any = None,
        separator: str = '.'
    ) -> Any:
        """
        提取嵌套配置
        
        Args:
            config: 配置字典
            path: 配置路径（用separator分隔）
            default: 默认值
            separator: 路径分隔符
        
        Returns:
            配置值或默认值
        
        示例:
            alpha = ConfigExtractor.get_nested(
                config,
                'market_configs.CN.price_impact_alpha',
                default=0.4
            )
        """
        try:
            keys = path.split(separator)
            value = config
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                    if value is None:
                        logger.debug(f"配置路径'{path}'在'{key}'处中断，使用默认值{default}")
                        return default
                else:
                    logger.debug(f"配置路径'{path}'在'{key}'处类型错误，使用默认值{default}")
                    return default
            
            return value
        except Exception as e:
            logger.debug(f"配置提取失败'{path}': {e}，使用默认值{default}")
            return default
    
    @staticmethod
    def get_typed(
        config: Dict[str, Any],
        key: str,
        expected_type: type,
        default: T
    ) -> T:
        """
        提取指定类型的配置
        
        Args:
            config: 配置字典
            key: 配置键
            expected_type: 期望类型
            default: 默认值
        
        Returns:
            指定类型的配置值或默认值
        
        示例:
            threshold = ConfigExtractor.get_typed(
                config,
                'threshold',
                float,
                0.95
            )
        """
        try:
            value = config.get(key, default)
            
            if not isinstance(value, expected_type):
                logger.debug(
                    f"配置'{key}'类型错误: 期望{expected_type.__name__}, "
                    f"实际{type(value).__name__}, 使用默认值{default}"
                )
                return default
            
            return value
        except Exception as e:
            logger.debug(f"配置提取失败'{key}': {e}，使用默认值{default}")
            return default
    
    @staticmethod
    def get_market_config(
        config: Dict[str, Any],
        market_type: str,
        param_name: str,
        default: Any
    ) -> Any:
        """
        提取市场特定配置（简化常见模式）
        
        Args:
            config: 配置字典
            market_type: 市场类型（CN/US/HK等）
            param_name: 参数名
            default: 默认值
        
        Returns:
            市场配置值或默认值
        
        示例:
            alpha = ConfigExtractor.get_market_config(
                config,
                'CN',
                'price_impact_alpha',
                0.4
            )
        """
        return ConfigExtractor.get_nested(
            config,
            f'market_configs.{market_type}.{param_name}',
            default=default
        )
    
    @staticmethod
    def get_with_fallback(
        config: Dict[str, Any],
        keys: List[str],
        default: Any
    ) -> Any:
        """
        按优先级获取配置（fallback链）
        
        Args:
            config: 配置字典
            keys: 配置键列表（按优先级）
            default: 默认值
        
        Returns:
            第一个存在的配置值或默认值
        
        示例:
            # 先尝试advanced_var_enabled，再尝试var_enabled
            enabled = ConfigExtractor.get_with_fallback(
                config,
                ['advanced_var_enabled', 'var_enabled'],
                default=False
            )
        """
        for key in keys:
            if key in config:
                return config[key]
        
        logger.debug(f"配置键{keys}均不存在，使用默认值{default}")
        return default
    
    @staticmethod
    def get_numeric_range(
        config: Dict[str, Any],
        key: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        提取带范围限制的数值配置
        
        Args:
            config: 配置字典
            key: 配置键
            default: 默认值
            min_value: 最小值限制
            max_value: 最大值限制
        
        Returns:
            范围内的配置值或默认值
        
        示例:
            confidence = ConfigExtractor.get_numeric_range(
                config,
                'var_confidence_level',
                default=0.95,
                min_value=0.5,
                max_value=0.999
            )
        """
        try:
            value = float(config.get(key, default))
            
            if min_value is not None and value < min_value:
                logger.warning(
                    f"配置'{key}'={value}低于最小值{min_value}，使用最小值"
                )
                return min_value
            
            if max_value is not None and value > max_value:
                logger.warning(
                    f"配置'{key}'={value}超过最大值{max_value}，使用最大值"
                )
                return max_value
            
            return value
        except Exception as e:
            logger.debug(f"配置提取失败'{key}': {e}，使用默认值{default}")
            return default


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_required_keys(
        config: Dict[str, Any],
        required_keys: Dict[str, type],
        config_name: str = "配置"
    ) -> tuple[bool, List[str]]:
        """
        验证必需配置键
        
        Args:
            config: 配置字典
            required_keys: {键名: 期望类型}
            config_name: 配置名称
        
        Returns:
            (是否有效, 错误消息列表)
        
        示例:
            valid, errors = ConfigValidator.validate_required_keys(
                config,
                {
                    'market_type': str,
                    'var_confidence_level': float
                },
                "风险配置"
            )
        """
        errors = []
        
        for key, expected_type in required_keys.items():
            if key not in config:
                errors.append(f"{config_name}缺少必需键'{key}'")
                continue
            
            if not isinstance(config[key], expected_type):
                errors.append(
                    f"{config_name}键'{key}'类型错误: "
                    f"期望{expected_type.__name__}, 实际{type(config[key]).__name__}"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"{config_name}验证失败: {'; '.join(errors)}")
        
        return is_valid, errors
    
    @staticmethod
    def fill_defaults(
        config: Dict[str, Any],
        defaults: Dict[str, Any],
        deep_merge: bool = False
    ) -> Dict[str, Any]:
        """
        填充默认配置
        
        Args:
            config: 原始配置
            defaults: 默认配置
            deep_merge: 是否深度合并（嵌套字典）
        
        Returns:
            合并后的配置
        
        示例:
            config = ConfigValidator.fill_defaults(
                user_config,
                DEFAULT_RISK_CONFIG,
                deep_merge=True
            )
        """
        result = config.copy()
        
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
                logger.debug(f"填充默认配置: {key} = {default_value}")
            elif deep_merge and isinstance(default_value, dict) and isinstance(result[key], dict):
                # 递归合并
                result[key] = ConfigValidator.fill_defaults(
                    result[key],
                    default_value,
                    deep_merge=True
                )
        
        return result


class ThresholdManager:
    """阈值管理器（统一阈值配置和动态调整）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化阈值管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self._cache = {}
    
    def get_threshold(
        self,
        threshold_name: str,
        market_type: Optional[str] = None,
        default: float = 1.0
    ) -> float:
        """
        获取阈值（支持市场特定阈值）
        
        Args:
            threshold_name: 阈值名称
            market_type: 市场类型（可选，如果提供则查找市场特定阈值）
            default: 默认值
        
        Returns:
            阈值
        
        示例:
            threshold = manager.get_threshold(
                'normal_vol_max',
                market_type='CN',
                default=1.2
            )
        """
        # 构建缓存键
        cache_key = f"{threshold_name}_{market_type}" if market_type else threshold_name
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 尝试获取市场特定阈值
        if market_type:
            value = ConfigExtractor.get_market_config(
                self.config,
                market_type,
                threshold_name,
                default=None
            )
            if value is not None:
                self._cache[cache_key] = float(value)
                return float(value)
        
        # 回退到通用阈值
        value = ConfigExtractor.get_nested(
            self.config,
            f'thresholds.{threshold_name}',
            default=default
        )
        
        self._cache[cache_key] = float(value)
        return float(value)
    
    def get_dynamic_threshold(
        self,
        threshold_name: str,
        adjustment_factor: float = 1.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        获取动态调整的阈值
        
        Args:
            threshold_name: 阈值名称
            adjustment_factor: 调整因子
            min_value: 最小值限制
            max_value: 最大值限制
        
        Returns:
            动态阈值
        
        示例:
            # 在压力期调高阈值
            threshold = manager.get_dynamic_threshold(
                'leverage_limit',
                adjustment_factor=0.8,  # 降低20%
                min_value=1.0
            )
        """
        base_value = self.get_threshold(threshold_name)
        adjusted_value = base_value * adjustment_factor
        
        if min_value is not None:
            adjusted_value = max(adjusted_value, min_value)
        
        if max_value is not None:
            adjusted_value = min(adjusted_value, max_value)
        
        return adjusted_value


# ==================== 配置数据类 ====================

@dataclass
class MonitoringConfig:
    """监控配置数据类
    
    Note:
        所有字段必须从config文件读取，不提供默认值
        配置文件路径: core_bak_refactored/config/{env}/monitoring.yml
    """
    check_interval: int  # 检查间隔（秒）
    max_quality_history: int  # 最大质量历史记录数
    alert_threshold: float  # 告警阈值
    enable_cross_validation: bool  # 是否启用交叉验证
    cross_validation_interval: int  # 交叉验证间隔（秒）


@dataclass
class AlertingConfig:
    """告警配置数据类
    
    Note:
        所有字段必须从config文件读取，不提供默认值
        配置文件路径: core_bak_refactored/config/{env}/alerting.yml
    """
    enable_email: bool
    enable_sms: bool
    enable_wechat: bool
    enable_dingtalk: bool
    enable_slack: bool
    enable_webhook: bool
    enable_log: bool
    min_severity: str  # 最小告警严重程度
    deduplication_window: int  # 去重窗口（秒）

# MarketConfig 已从 core.share.market.market_config 导入，不再需要 dataclass 定义

@dataclass
class ProvidersConfig:
    """数据配置数据类
    
    Note:
        - 大部分字段从 core_bak_refactored/config/{env}/data_provider.yml 读取
        - use_proxy 配置已融入到 providers 列表中，每个 provider 对象包含 use_proxy 字段
        
        配置示例:
        providers:
        - id: akshare
          use_proxy: false
        - id: yahoo
          use_proxy: false
        - id: finnhub
          use_proxy: true
    """
    default_index: str
    cache_enabled: bool
    cache_ttl: int  # 缓存TTL（秒）
    max_retries: int  # 最大重试次数
    providers: list  # 数据源列表（包含 use_proxy 配置）

    def __post_init__(self):
        # 确保 providers 是列表类型
        if not isinstance(self.providers, list):
            raise ValueError("providers 必须是列表类型")
        
        # 验证每个 provider 是否包含必要字段
        for provider in self.providers:
            if not isinstance(provider, dict):
                raise ValueError("每个 provider 必须是字典类型")
            if 'id' not in provider:
                raise ValueError("每个 provider 必须包含 'id' 字段")
            # 确保 use_proxy 字段存在，默认为 false
            if 'use_proxy' not in provider:
                provider['use_proxy'] = False
    
    def get_provider_proxy_config(self, provider_id: str) -> bool:
        """获取指定数据源的代理配置
        
        Args:
            provider_id: 数据源ID（如 'akshare', 'yahoo', 'finnhub'）
        
        Returns:
            bool: 是否使用代理
        
        Examples:
            >>> config = ConfigManager().get_provider_config()
            >>> config.get_provider_proxy_config('finnhub')
            True
            >>> config.get_provider_proxy_config('akshare')
            False
        """
        for provider in self.providers:
            if provider.get('id') == provider_id:
                return provider.get('use_proxy', False)
        return False


@dataclass
class SystemConfig:
    """系统配置数据类
    
    Note:
        所有字段必须从config文件读取，不提供默认值
        配置文件路径: core_bak_refactored/config/{env}/system.yml
    """
    log_level: str
    max_concurrent_requests: int
    timeout: int  # 请求超时（秒）
    enable_health_check: bool
    health_check_interval: int  # 健康检查间隔（秒）
    proxies: Optional[dict] = None  # 代理配置，例如 {"http": "http://127.0.0.1:8002", "socks5": "socks5://127.0.0.1:1081"}


@dataclass
class CacheConfig:
    """缓存配置数据类
    
    Note:
        所有字段从 core_bak_refactored/config/{env}/cache.yml 读取
    """
    cache_mode: str  # 'memory' 或 'redis'
    window_size: int  # 窗口大小
    memory_max_windows: int  # 内存最大窗口数
    memory_ttl: int  # 内存TTL（秒）
    redis_ttl: int  # Redis TTL（秒）
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


# ==================== 配置管理器 ====================

class ConfigManager:
    """
    配置管理器（单例模式）
    
    职责：提供标准化的配置管理接口
    
    Note:
        使用单例模式确保全局只有一个实例，避免多个watchdog监听器冲突
    """
    
    _instance = None
    _lock = None
    _observer = None  # 全局watchdog observer
    environment: str
    
    @staticmethod
    def _get_environment() -> str:
        """获取当前环境配置"""
        return os.getenv('DEEPSEEK_ENV', 'dev')
    
    def __new__(cls, config_file: Optional[str] = None, environment: Optional[str] = None):
        """单例模式实现"""
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()
            
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._initialized = False
        
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        # 避免重复初始化，但允许environment参数变更时重新加载
        requested_env = environment or ConfigManager._get_environment()
        
        if self._initialized:
            # 如果environment参数与当前环境不同，重新加载配置
            if requested_env != self.environment:
                logger.info(f"检测到环境变更: {self.environment} -> {requested_env}，重新加载配置")
                self.environment = requested_env
                self._load_config()
            return
        
        self.config_file = config_file
        self.environment = requested_env
        self._config = {}
        self._load_config()
        
        # 启动热加载监听（仅在非测试环境且尚未启动）
        if self.environment != 'test' and ConfigManager._observer is None:
            self._start_hot_reload_watcher()
        
        self._initialized = True
    
    def _load_config(self):
        """加载配置（优先从 core_bak_refactored/config/*.yml 读取）"""
        try:
            import yaml
            import glob
            # core_bak_refactored/core/share/config_manager.py -> core_bak_refactored/config
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
            env_dir = os.path.join(base_dir, self.environment)
            config_dir = env_dir if os.path.exists(env_dir) else base_dir
                
            # 自动扫描所有 .yml 文件
            loaded = {}
            yml_files = glob.glob(os.path.join(config_dir, '*.yml'))
                
            for yml_path in yml_files:
                config_key = os.path.splitext(os.path.basename(yml_path))[0]
                try:
                    with open(yml_path, 'r', encoding='utf-8') as f:
                        loaded[config_key] = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"跳过无效配置文件 {yml_path}: {e}")
                        
            if loaded:
                self._config = loaded
                logger.info(f"从 YAML 加载配置 [{self.environment}]: {sorted(loaded.keys())}")
                return
        except Exception as e:
            logger.warning(f"YAML配置加载失败: {e}")
            raise RuntimeError(
                f"配置文件加载失败，请检查配置文件是否存在: "
                f"core_bak_refactored/config/{self.environment}/*.yml"
            )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        config_dict = self._config.get('monitoring', {})
        return MonitoringConfig(**config_dict)
    
    def get_alerting_config(self) -> AlertingConfig:
        """获取告警配置"""
        config_dict = self._config.get('alerting', {})
        return AlertingConfig(**config_dict)

    def get_market_config(self) -> MarketConfig:
        """获取市场配置
        
        Returns:
            MarketConfig: 市场配置管理器实例，从 market.yml 加载配置
        """
        return MarketConfig(environment=self.environment)
    
    def get_risk_config(self):
        """获取风险配置
        
        Returns:
            RiskConfig: 风险配置管理器实例，从 risk.yml 加载配置
            
        Examples:
            >>> cm = ConfigManager()
            >>> risk_config = cm.get_risk_config()
            >>> cn_risk = risk_config.get_risk_parameters('CN')
            >>> print(cn_risk['risk_free_rate'])  # 0.03
        """
        from core_bak_refactored.core.risk.config import RiskConfig
        return RiskConfig(environment=self.environment)
    
    def get_trade_config(self):
        """获取交易配置
        
        Returns:
            TradeConfig: 交易配置管理器实例，从 trade.yml 加载配置
            
        Examples:
            >>> cm = ConfigManager()
            >>> trade_config = cm.get_trade_config()
            >>> cn_cost = trade_config.get_trade_cost('CN')
            >>> print(cn_cost['price_impact_alpha'])  # 0.55
        """
        from core_bak_refactored.core.exec.config import TradeConfig
        return TradeConfig(environment=self.environment)

    def get_provider_config(self) -> ProvidersConfig:
        """获取数据配置"""
        config_dict = dict(self._config.get('data_provider', {}))
        return ProvidersConfig(**config_dict)
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        config_dict = self._config.get('system', {})
        return SystemConfig(**config_dict)
    
    def get_cache_config(self) -> CacheConfig:
        """获取缓存配置✨ 新增"""
        cache_dict = self._config.get('cache', {})
        
        # 提取配置值
        result = {
            'cache_mode': cache_dict.get('cache_mode', 'memory'),
            'window_size': cache_dict.get('window_strategy', {}).get('window_size', 1),
            'memory_max_windows': cache_dict.get('memory', {}).get('max_windows', 1000),
            'memory_ttl': cache_dict.get('memory', {}).get('ttl', 300),
            'redis_ttl': cache_dict.get('redis', {}).get('ttl', 3600),
            'redis_host': cache_dict.get('redis', {}).get('host', 'localhost'),
            'redis_port': cache_dict.get('redis', {}).get('port', 6379),
            'redis_db': cache_dict.get('redis', {}).get('db', 0),
            'redis_password': cache_dict.get('redis', {}).get('password'),
        }
        
        return CacheConfig(**result)
    
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
    
    def update(self, config_dict: Dict[str, Any]):
        """批量更新配置
        
        Args:
            config_dict: 要更新的配置字典，支持嵌套结构
        
        Examples:
            >>> cm = ConfigManager()
            >>> cm.update({'data': {'primary_source': 'akshare'}})
            >>> cm.update({'credentials': {'tushare': {'token': 'xxx'}}})
        
        Note:
            此方法仅更新内存中的配置，不会持久化到文件。
            如需持久化，请使用 save_config() 方法（如果存在）。
        """
        for key, value in config_dict.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                # 对于字典类型，递归更新
                self._config[key].update(value)
            else:
                # 直接覆盖
                self._config[key] = value
        
        logger.info(f"配置已更新: {list(config_dict.keys())}")
    
    def get_trading_hours(self, market_code: str) -> Dict[str, Any]:
        """获取指定市场的交易时段配置
        
        Args:
            market_code: 市场代码（CN, US, HK, JP, EU, SG）
        
        Returns:
            Dict包含交易时段配置:
                - open: 开盘时间（HH:MM格式）
                - close: 收盘时间
                - lunch_start: 午休开始时间（如有）
                - lunch_end: 午休结束时间（如有）
                - has_lunch_break: 是否有午休
                - timezone: 时区
                - description: 描述
        
        Examples:
            >>> cm = ConfigManager()
            >>> hours = cm.get_trading_hours('CN')
            >>> print(hours['open'])  # '09:30'
            >>> print(hours['close']) # '15:00'
        """
        market_config = self._config.get('market', {})
        trading_hours = market_config.get('trading_hours', {})
        hours = trading_hours.get(market_code, {})
        
        if not hours:
            logger.warning(f"未找到市场 {market_code} 的交易时段配置，使用默认值")
            # 默认配置（中国市场）
            return {
                'open': '09:30',
                'close': '15:00',
                'lunch_start': '11:30',
                'lunch_end': '13:00',
                'has_lunch_break': True,
                'timezone': 'Asia/Shanghai',
                'description': '默认交易时段'
            }
        
        return hours
    
    def get_provider_for_symbol(self, symbol: str) -> Optional[str]:
        """根据股票代码获取数据源ID"""
        from core_bak_refactored.core.share.market.market_utils import MarketUtils
        
        market_code = MarketUtils.infer_market_from_symbol(symbol)
        market_config = self._config.get('market', {})
        market_sources = market_config.get('market_sources', {})
        provider_id = market_sources.get(market_code.value)
        
        if not provider_id:
            logger.warning(f"未找到市场 {market_code.value} 的数据源配置")
            return None
        
        return provider_id
    
    def _start_hot_reload_watcher(self):
        """启动热加载监听器（全局单例）"""
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
                        time.sleep(0.1)
                        self.config_manager._load_config()
            
            def watch_thread():
                # core_bak_refactored/core/share/config_manager.py -> core_bak_refactored/config
                base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
                env_dir = os.path.join(base_dir, self.environment)
                watch_path = env_dir if os.path.exists(env_dir) else base_dir
                
                event_handler = ConfigReloadHandler(self)
                ConfigManager._observer = Observer()
                ConfigManager._observer.schedule(event_handler, watch_path, recursive=False)
                ConfigManager._observer.start()
                logger.info(f"启动配置热加载监听: {watch_path}")
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    if ConfigManager._observer:
                        ConfigManager._observer.stop()
                if ConfigManager._observer:
                    ConfigManager._observer.join()
            
            watcher = threading.Thread(target=watch_thread, daemon=True, name='ConfigHotReload')
            watcher.start()
        except ImportError:
            logger.info("未安装watchdog，跳过配置热加载")
        except Exception as e:
            logger.warning(f"配置热加载启动失败: {e}")

    def get_config_path(self, name: str) -> str:
        """获取配置文件绝对路径
        
        Args:
            name: 配置文件名（如 'market' 或 'market.yml'）
        
        Returns:
            配置文件的绝对路径
        
        Examples:
            >>> cm = ConfigManager()
            >>> cm.get_config_path('market')
            '/path/to/core_bak_refactored/config/dev/market.yml'
        """
        # core_bak_refactored/core/share/config_manager.py -> core_bak_refactored/config
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        env_dir = os.path.join(base_dir, self.environment)
        config_dir = env_dir if os.path.exists(env_dir) else base_dir
        
        # 自动添加 .yml 后缀
        if not name.endswith(('.yml', '.yaml')):
            name = f"{name}.yml"
        
        return os.path.join(config_dir, name)
