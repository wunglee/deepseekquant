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

from core_bak_refactored.core.share.market.market_enums import MarketCode

logger = logging.getLogger('DeepSeekQuant.Core.Share.ConfigManager')


@dataclass
class MonitoringConfig:
    """监控配置数据类
    
    Note:
        所有字段必须从配置文件读取，不提供默认值
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
        所有字段必须从配置文件读取，不提供默认值
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


@dataclass
class DataConfig:
    """数据配置数据类
    
    Note:
        - 大部分字段从 core_bak_refactored/config/{env}/data_provider.yml 读取
        - market_sources 从 core_bak_refactored/config/{env}/market.yml 读取
        
    ❌ 已废弃：primary_source（不向后兼容）
    ✅ 新字段：market_sources - 每个市场单独配置数据源（从 market.yml 读取）
    """
    market_sources: Dict[str, str]  # 市场数据源映射：{market_code: provider_id}，来自 market.yml
    default_index: str
    cache_enabled: bool
    cache_ttl: int  # 缓存过期时间（秒）
    max_retries: int  # 最大重试次数
    providers: list  # 数据源列表（用于 Providers 页面展示）
    data_providers: Optional[dict] = None  # 数据源代理配置，例如 {"yahoo_finance": {"use_proxy": true}}

    def __post_init__(self):
        # 确保 providers 是列表类型
        if not isinstance(self.providers, list):
            raise ValueError("providers 必须是列表类型")
        # 确保 market_sources 是字典类型
        if not isinstance(self.market_sources, dict):
            raise ValueError("market_sources 必须是字典类型")


@dataclass
class SystemConfig:
    """系统配置数据类
    
    Note:
        所有字段必须从配置文件读取，不提供默认值
        配置文件路径: core_bak_refactored/config/{env}/system.yml
    """
    log_level: str
    max_concurrent_requests: int
    timeout: int  # 请求超时（秒）
    enable_health_check: bool
    health_check_interval: int  # 健康检查间隔（秒）
    proxies: Optional[dict] = None  # 代理配置，例如 {"http": "http://127.0.0.1:8002", "socks5": "socks5://127.0.0.1:1081"}


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
    environment:str
    
    @staticmethod
    def _get_environment() -> str:
        """
        获取当前环境配置（内部方法）
        
        Returns:
            环境名称（dev/test/prod）
            
        Note:
            这是内部方法，外部代码不应直接调用
            环境是 ConfigManager 的实现细节，外部应通过 ConfigManager 实例获取配置
        """
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
        requested_env = environment or ConfigManager._get_environment()  # 使用私有方法
        
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
        """加载配置（优先从 core_bak_refactored/config/*.yml 读取，找不到则回退默认）"""
        # 优先读取目录化YAML配置
        try:
            import yaml  # 可选依赖
            import glob
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
            # 环境隔离支持：优先加载环境目录下的配置
            env_dir = os.path.join(base_dir, self.environment)
            config_dir = env_dir if os.path.exists(env_dir) else base_dir
                
            # 自动扫描所有 .yml 文件（不需要手动维护列表）
            loaded = {}
            yml_files = glob.glob(os.path.join(config_dir, '*.yml'))
                
            for yml_path in yml_files:
                # 从文件名提取配置 key（去掉 .yml 后缀）
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
        """配置文件缺失时抛出异常（不再提供硬编码默认值）
        
        Raises:
            RuntimeError: 配置文件不存在或加载失败
        
        Note:
            所有配置必须从配置文件读取，不提供硬编码默认值
            请确保配置文件存在: core_bak_refactored/config/{env}/*.yml
        """
        raise RuntimeError(
            f"配置文件加载失败，请检查配置文件是否存在: "
            f"core_bak_refactored/config/{self.environment}/*.yml\n"
            f"必需的配置文件: monitoring.yml, alerting.yml, data_provider.yml, system.yml"
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        config_dict = self._config.get('monitoring', {})
        return MonitoringConfig(**config_dict)
    
    def get_alerting_config(self) -> AlertingConfig:
        """获取告警配置"""
        config_dict = self._config.get('alerting', {})
        return AlertingConfig(**config_dict)
    
    def get_data_config(self) -> DataConfig:
        """获取数据配置
        
        Note:
            - 大部分字段从 data_provider.yml 读取
            - market_sources 从 market.yml 读取
        """
        # 从 data_provider.yml 读取基础配置
        config_dict = dict(self._config.get('data_provider', {}))
        
        # 从 market.yml 读取 market_sources
        market_config = self._config.get('market', {})
        config_dict['market_sources'] = market_config.get('market_sources', {})
        
        return DataConfig(**config_dict)
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        config_dict = self._config.get('system', {})
        return SystemConfig(**config_dict)
    
    def get_proxies_from_config(self) -> Optional[dict]:
        """
        从系统配置中获取代理设置
        
        Returns:
            代理配置字典，如果没有配置则返回None
        """
        try:
            system_config = self.get_system_config()
            return system_config.proxies
        except Exception as e:
            # 如果配置加载失败，返回None
            return None
    
    def configure_provider_proxy(self, provider, proxy_config: Optional[dict] = None):
        """
        为数据提供者配置代理
        
        Args:
            provider: 数据提供者实例
            proxy_config: 代理配置字典
        """
        if proxy_config is None:
            proxy_config = self.get_proxies_from_config()
        
        if proxy_config and hasattr(provider, 'proxy'):
            provider.proxy = proxy_config
            # 注意：timeout属性可能不存在于所有provider中，需要检查
            if hasattr(provider, 'timeout'):
                provider.timeout = 20
    
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
                        time.sleep(0.1)  # 防止写入未完成
                        self.config_manager._load_config()
            
            # 启动后台监听线程
            def watch_thread():
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
    
    def validate_market_sources(self, market_sources: Dict[str, str]) -> tuple[bool, Optional[str]]:
        """验证市场数据源映射配置
        
        Args:
            market_sources: 市场到数据源的映射字典
        
        Returns:
            (是否有效, 错误信息)
        
        Examples:
            >>> config_manager = ConfigManager()
            >>> valid, error = config_manager.validate_market_sources({'CN': 'akshare', 'US': 'yahoo'})
            >>> valid
            False  # 缺少必需市场
            >>> error
            '市场 HK 未配置数据源'
        
        验证规则:
            - 所有必需市场（CN/HK/US/EU/JP/SG）都必须配置
            - 数据源 ID 不能为空
        """
        if not isinstance(market_sources, dict):
            return False, 'market_sources 必须是字典类型'
        
        # 验证所有必需市场都有配置
        required_markets = [
            MarketCode.CN,
            MarketCode.HK,
            MarketCode.US,
            MarketCode.EU,
            MarketCode.JP,
            MarketCode.SG
        ]
        
        for market in required_markets:
            market_value = market.value
            if market_value not in market_sources:
                return False, f'市场 {market_value} 未配置数据源'
            
            provider_id = market_sources[market_value]
            if not provider_id or not isinstance(provider_id, str) or not provider_id.strip():
                return False, f'市场 {market_value} 的数据源 ID 无效'
        
        return True, None
    
    def save_market_sources(self, market_sources: Dict[str, str], env: str = None) -> bool:
        """保存市场数据源映射到配置文件
        
        Args:
            market_sources: 市场到数据源的映射字典
            env: 环境名称（默认使用当前环境）
        
        Returns:
            bool: 是否保存成功
        
        Examples:
            >>> config_manager = ConfigManager()
            >>> market_sources = {'CN': 'akshare', 'US': 'yahoo', 'HK': 'akshare', ...}
            >>> config_manager.save_market_sources(market_sources)
            True
        
        Raises:
            ValueError: 如果 market_sources 验证失败
            
        Note:
            保存到 market.yml 文件的 market_sources 字段
        """
        # 1. 验证配置
        valid, error = self.validate_market_sources(market_sources)
        if not valid:
            raise ValueError(f"市场配置验证失败: {error}")
        
        # 2. 确定保存路径 - 保存到 market.yml
        import yaml
        target_env = env or self.environment
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        env_dir = os.path.join(base_dir, target_env)
        market_yml_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, 'market.yml')
        
        try:
            # 3. 读取现有配置
            if os.path.exists(market_yml_path):
                with open(market_yml_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}
            
            # 4. 更新 market_sources
            config_data['market_sources'] = market_sources
            
            # 5. 写入文件
            with open(market_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"保存市场配置成功: {market_sources} -> {market_yml_path}")
            
            # 6. 重新加载配置（更新内存中的配置）
            self._load_config()
            
            return True
            
        except Exception as e:
            logger.error(f"保存市场配置失败: {e}")
            raise
    
    def get_config_path(self, config_type: str, env: str = None) -> str:
        """获取配置文件路径
        
        Args:
            config_type: 配置类型（如 'data', 'credentials', 'monitoring'）
            env: 环境名称（默认使用当前环境）
        
        Returns:
            str: 配置文件绝对路径
        
        Examples:
            >>> config_manager = ConfigManager()
            >>> config_manager.get_config_path('data')
            '/path/to/core_bak_refactored/config/dev/data_provider.yml'
            >>> config_manager.get_config_path('credentials', env='prod')
            '/path/to/core_bak_refactored/config/prod/credentials.yml'
        """
        target_env = env or self.environment
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        env_dir = os.path.join(base_dir, target_env)
        
        config_file = f"{config_type}.yml"
        config_path = os.path.join(env_dir if os.path.exists(env_dir) else base_dir, config_file)
        
        return config_path
    
    def get_trading_hours(self, market_code: MarketCode) -> dict:
        """获取市场交易时段配置
        
        Args:
            market_code: 市场代码枚举 (MarketCode.CN/HK/US/EU/JP/SG)
        
        Returns:
            dict: {
                'open': '开盘时间 HH:MM',
                'close': '收盘时间 HH:MM',
                'lunch_start': '午休开始 HH:MM' (optional),
                'lunch_end': '午休结束 HH:MM' (optional),
                'has_lunch_break': bool,
                'timezone': '时区'
            }
        
        Examples:
            >>> config_manager = ConfigManager()
            >>> config_manager.get_trading_hours(MarketCode.CN)
            {'open': '09:30', 'close': '15:00', 'lunch_start': '11:30', 'lunch_end': '13:00', 'has_lunch_break': True, 'timezone': 'Asia/Shanghai'}
        
        Note:
            配置从 market.yml 的 trading_hours 字段读取
        """
        # 默认配置（A股）
        default_hours = {
            'open': '09:30',
            'close': '15:00',
            'lunch_start': '11:30',
            'lunch_end': '13:00',
            'has_lunch_break': True,
            'timezone': 'Asia/Shanghai'
        }
        
        # 🔧 确保smarket_code是MarketCode枚举
        if not isinstance(market_code, MarketCode):
            logger.warning(f"market_code应为MarketCode枚举，当前类型: {type(market_code)}，尝试解析")
            market_code = MarketCode.parse(market_code)
        
        market_config = self._config.get('market', {})
        trading_hours = market_config.get('trading_hours', {})
        market_hours = trading_hours.get(market_code.value, {})
        
        if not market_hours:
            logger.warning(f"未找到市场 {market_code.value} 的交易时段配置，使用默认配置")
            return default_hours
        
        return market_hours