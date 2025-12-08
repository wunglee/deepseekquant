"""
数据提供者工厂 - 统一创建和管理所有数据源

职责：
- 注册和管理所有内置数据提供者
- 支持外部注入自定义数据提供者（依赖注入）
- 提供统一的创建接口
- 确保所有provider实现HistoricalDataProvider接口

设计原则：
- 工厂模式：集中管理provider创建逻辑
- 依赖注入：支持外部注入自定义实现（测试Mock等）
- 类型安全：所有provider必须实现统一接口
- 懒加载：按需创建provider实例

使用示例：
    # 基本使用
    factory = DataProviderFactory()
    provider = factory.create('yahoo')
    data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-12-31')
    
    # 注入自定义provider
    factory.register('custom', MyCustomProvider)
    provider = factory.create('custom', **config)
"""

from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger('DeepSeekQuant.ProviderFactory')


class DataProviderFactory:
    """
    数据提供者工厂
    
    功能：
    - 注册内置数据提供者（yahoo, tushare, mock等）
    - 支持外部注入自定义provider（依赖注入）
    - 统一创建provider实例
    - 类型验证和错误处理
    """
    
    def __init__(self):
        """初始化工厂，注册内置providers"""
        self._providers: Dict[str, Type] = {}
        self._register_builtin_providers()
    
    def _register_builtin_providers(self):
        """注册内置数据提供者（统一处理异常）"""
        # 定义内置providers列表：(name, module_path, class_name)
        builtin_providers = [
            ('yahoo', 'core_bak_refactored.core.data.providers.yahoo_finance', 'YahooFinanceDataProvider'),
            ('tushare', 'core_bak_refactored.core.data.providers.tushare', 'TushareDataProvider'),
            ('real', 'core_bak_refactored.core.data.providers.historical_data_provider', 'RealHistoricalDataProvider'),
            ('akshare', 'core_bak_refactored.core.data.providers.akshare_provider', 'AKShareDataProvider'),
            ('alpha_vantage', 'core_bak_refactored.core.data.providers.alpha_vantage', 'AlphaVantageProvider'),
            ('polygon', 'core_bak_refactored.core.data.providers.polygon', 'fetch_polygon_data'),
            ('iex_cloud', 'core_bak_refactored.core.data.providers.iex_cloud', 'fetch_iex_cloud_data'),
            ('finnhub', 'core_bak_refactored.core.data.providers.finnhub', 'fetch_finnhub_data'),
            ('twelve_data', 'core_bak_refactored.core.data.providers.twelve_data', 'fetch_twelve_data'),
        ]
        
        for name, module_path, class_name in builtin_providers:
            try:
                # 动态导入模块
                module = __import__(module_path, fromlist=[class_name])
                provider_class = getattr(module, class_name)
                self._providers[name] = provider_class
                logger.debug(f"成功注册 {name} provider")
            except Exception as e:
                logger.warning(f"{name} provider 注册失败: {e}")
        
        # Mock数据提供者仅在测试中使用，不在生产工厂中注册
        # 如需在测试中使用，请手动注册：
        # factory.register('mock', MockHistoricalDataProvider)
        
        logger.info(f"已注册 {len(self._providers)} 个内置数据提供者: {list(self._providers.keys())}")
    
    def register(self, name: str, provider_class: Type):
        """
        注册自定义数据提供者（依赖注入入口）
        
        Args:
            name: provider名称（如'custom_mock', 'my_provider'）
            provider_class: provider类（必须实现HistoricalDataProvider接口）
        
        Example:
            >>> factory = DataProviderFactory()
            >>> factory.register('custom', MyCustomProvider)
            >>> provider = factory.create('custom')
        """
        if name in self._providers:
            logger.warning(f"覆盖已存在的provider: {name}")
        
        self._providers[name] = provider_class
        logger.info(f"注册自定义provider: {name} -> {provider_class.__name__}")
    
    def create(self, name: str, **kwargs) -> Any:
        """
        创建数据提供者实例
        
        Args:
            name: provider名称（'yahoo', 'tushare', 'akshare', 'real', 或自定义名称）
            **kwargs: 传递给provider构造函数的参数
        
        Returns:
            HistoricalDataProvider实例
        
        Raises:
            ValueError: provider不存在
        
        Example:
            >>> factory = DataProviderFactory()
            >>> provider = factory.create('akshare')
            >>> data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-12-31')
        """
        if name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(
                f"未知的provider: '{name}'\n"
                f"可用的providers: {available}\n"
                f"提示: 使用 factory.register('{name}', YourProviderClass) 注册自定义provider"
            )
        
        provider_class = self._providers[name]
        
        try:
            instance = provider_class(**kwargs)
            logger.debug(f"创建provider实例: {name} ({provider_class.__name__})")
            return instance
        except Exception as e:
            logger.error(f"创建provider失败: {name} - {e}")
            raise RuntimeError(f"Failed to create provider '{name}': {e}") from e
    
    def list_providers(self) -> list:
        """
        列出所有已注册的provider名称
        
        Returns:
            provider名称列表
        """
        return list(self._providers.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        检查provider是否已注册
        
        Args:
            name: provider名称
        
        Returns:
            是否已注册
        """
        return name in self._providers
    
    def unregister(self, name: str):
        """
        移除已注册的provider
        
        Args:
            name: provider名称
        
        Note:
            内置provider也可以被移除（不推荐）
        """
        if name in self._providers:
            del self._providers[name]
            logger.info(f"移除provider: {name}")
        else:
            logger.warning(f"尝试移除不存在的provider: {name}")


# 全局单例工厂（可选）
_global_factory: Optional[DataProviderFactory] = None


def get_global_factory() -> DataProviderFactory:
    """
    获取全局单例工厂
    
    Returns:
        全局DataProviderFactory实例
    
    Example:
        >>> factory = get_global_factory()
        >>> provider = factory.create('yahoo')
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = DataProviderFactory()
    return _global_factory


def reset_global_factory():
    """
    重置全局工厂（主要用于测试）
    """
    global _global_factory
    _global_factory = None
