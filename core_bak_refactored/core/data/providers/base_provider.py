"""
Provider 基类 - 三层数据架构封装

三层数据获取策略:
1. 内存缓存（最快）- 毫秒级
2. 数据库缓存（次快）- 0.1-0.3秒
3. 外部API（最慢）- 4-8秒

对外透明: 调用者无需关心数据来源，Provider自动选择最优策略
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

import yaml
import pandas as pd

from core_bak_refactored.core.share import ConfigManager

logger = logging.getLogger('DeepSeekQuant.DataProviders')


class BaseDataProvider(ABC):
    """
    数据提供者基类 - 三层数据架构
    
    核心责任:
    1. 封装三层数据获取逻辑（内存、数据库、API）
    2. 对外提供统一的数据接口
    3. 自动处理缓存读写，对调用者透明
    
    数据获取流程:
    get_index_prices() → _get_with_cache()
        ↓
    1. 检查内存缓存 (毫秒级)
        └─ 命中 → 返回
        ↓
    2. 检查数据库缓存 (0.1-0.3秒)
        └─ 命中 → 写入内存 → 返回
        ↓
    3. 调用外部API (4-8秒)
        └─ 成功 → 写入数据库 → 写入内存 → 返回
    """
    
    def __init__(self):
        """初始化数据提供者"""
        # 💚 内存缓存（粀单字典，可后续替换为 LRU Cache）
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # 💚 数据库服务（延迟初始化）
        self._db_service = None
        self._db_initialized = False
        
        # 缓存配置
        self._cache_ttl = 300  # 内存缓存TTL（秒）
        self._enable_memory_cache = True
        self._enable_db_cache = True
        
        # 加载配置
        self._load_cache_config()
    
    def _load_cache_config(self):
        """加载缓存配置"""
        try:
            config_manager = ConfigManager()
            # 💚 修复：使用 config 属性而不是 get_config 方法
            if hasattr(config_manager, 'config'):
                data_config = config_manager.config.get('data', {})
                
                if data_config:
                    self._cache_ttl = data_config.get('cache_ttl', 300)
                    self._enable_memory_cache = data_config.get('cache_enabled', True)
                
                # 检查数据库配置
                db_config = config_manager.config.get('database', {})
                if db_config:
                    cache_strategy = db_config.get('cache_strategy', {})
                    self._enable_db_cache = cache_strategy.get('enabled', True)
        except Exception as e:
            logger.debug(f"加载缓存配置失败，使用默认值: {e}")
    
    def _get_db_service(self):
        """
        获取数据库服务（延迟初始化）
        
        Returns:
            DatabaseService 或 None
        """
        if not self._enable_db_cache:
            return None
        
        if not self._db_initialized:
            try:
                from core_bak_refactored.infrastructure.database_service import get_database_service
                self._db_service = get_database_service()
                logger.info("数据库服务已启用")
            except Exception as e:
                logger.warning(f"数据库服务初始化失败，将不使用数据库缓存: {e}")
                self._db_service = None
            finally:
                self._db_initialized = True
        
        return self._db_service
    
    def _make_cache_key(self, index_id: str, start_date: str, end_date: str) -> str:
        """
        生成缓存键
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            缓存键字符串
        """
        return f"{index_id}:{start_date}:{end_date}"
    
    def _get_from_memory_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        从内存缓存获取数据
        
        Args:
            cache_key: 缓存键
        
        Returns:
            DataFrame 或 None
        """
        if not self._enable_memory_cache:
            return None
        
        cached = self._memory_cache.get(cache_key)
        if cached:
            # 检查是否过期
            if time.time() - cached['timestamp'] < self._cache_ttl:
                logger.debug(f"✅ 内存缓存命中: {cache_key}")
                return cached['data']
            else:
                # 过期，删除
                del self._memory_cache[cache_key]
        
        return None
    
    def _set_to_memory_cache(self, cache_key: str, data: pd.DataFrame):
        """
        写入内存缓存
        
        Args:
            cache_key: 缓存键
            data: 数据
        """
        if not self._enable_memory_cache or data is None or data.empty:
            return
        
        self._memory_cache[cache_key] = {
            'data': data.copy(),
            'timestamp': time.time()
        }
        logger.debug(f"✅ 写入内存缓存: {cache_key}")
    
    def _get_from_db_cache(self, index_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从数据库缓存获取数据
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame 或 None
        """
        db_service = self._get_db_service()
        if not db_service:
            return None
        
        try:
            df = db_service.get_cached_data(
                index_id,
                start_date,
                end_date,
                source=self.__class__.__name__
            )
            
            if df is not None and not df.empty:
                logger.info(f"✅ 数据库缓存命中: {index_id} ({len(df)} 条)")
                return df
        except Exception as e:
            logger.warning(f"从数据库获取缓存失败: {e}")
        
        return None
    
    def _set_to_db_cache(self, index_id: str, data: pd.DataFrame):
        """
        写入数据库缓存
        
        Args:
            index_id: 指数代码
            data: 数据
        """
        db_service = self._get_db_service()
        if not db_service or data is None or data.empty:
            return
        
        try:
            db_service.cache_data(
                index_id,
                data,
                source=self.__class__.__name__
            )
            logger.info(f"✅ 数据已缓存到数据库: {index_id} ({len(data)} 条)")
        except Exception as e:
            logger.warning(f"缓存数据到数据库失败: {e}")
    
    def _get_with_cache(self, index_id: str, start_date: str, end_date: str):
        """
        三层数据获取（核心方法）
        
        数据获取顺序:
        1. 内存缓存 → 命中则返回
        2. 数据库缓存 → 命中则写入内存并返回
        3. 外部API → 写入数据库和内存后返回
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            PriceData 对象
        """
        # 1. 尝试内存缓存
        cache_key = self._make_cache_key(index_id, start_date, end_date)
        cached_df = self._get_from_memory_cache(cache_key)
        
        if cached_df is not None:
            # 转换为 PriceData
            return self._dataframe_to_price_data(cached_df, index_id)
        
        # 2. 尝试数据库缓存
        cached_df = self._get_from_db_cache(index_id, start_date, end_date)
        
        if cached_df is not None:
            # 写入内存缓存
            self._set_to_memory_cache(cache_key, cached_df)
            # 转换为 PriceData
            return self._dataframe_to_price_data(cached_df, index_id)
        
        # 3. 调用外部API（子类实现）
        logger.info(f"🌐 缓存未命中，调用外部API: {index_id}")
        price_data = self._fetch_from_external_api(index_id, start_date, end_date)
        
        if price_data and price_data.count > 0:
            # 转换为 DataFrame
            df = price_data.to_dataframe()
            
            # 写入数据库缓存
            self._set_to_db_cache(index_id, df)
            
            # 写入内存缓存
            self._set_to_memory_cache(cache_key, df)
        
        return price_data
    
    @staticmethod
    def _dataframe_to_price_data(df: pd.DataFrame, symbol: str):
        """
        将 DataFrame 转换为 PriceData 对象
        
        Args:
            df: DataFrame
            symbol: 股票/指数代码
        
        Returns:
            PriceData 对象
        """
        from core_bak_refactored.core.data.providers.protocols import PriceData
        return PriceData.from_dataframe(df, symbol)
    
    # ========================================================================
    # 数据获取接口（对外提供，自动使用缓存）
    # ========================================================================
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str):
        """
        获取指数价格数据（对外接口，自动使用三层缓存）
        
        💚 三层数据策略:
        1. 内存缓存 → 毫秒级
        2. 数据库缓存 → 0.1-0.3秒
        3. 外部API → 4-8秒
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            PriceData: 价格数据对象
        """
        return self._get_with_cache(index_id, start_date, end_date)
    
    def get_stock_prices(self, stock_id: str, start_date: str, end_date: str):
        """
        获取股票价格数据（对外接口，自动使用三层缓存）
        
        Args:
            stock_id: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            PriceData: 价格数据对象
        """
        # 股票数据也使用相同的缓存策略
        return self._get_with_cache(stock_id, start_date, end_date)
    
    # ========================================================================
    # 内部接口（子类必须实现）
    # ========================================================================
    
    @abstractmethod
    def _fetch_from_external_api(self, symbol: str, start_date: str, end_date: str):
        """
        从外部API获取数据（抽象方法，子类必须实现）
        
        💚 注意:
        - 此方法仅供内部使用，不对外暴露
        - 外部调用者应使用 get_index_prices() 或 get_stock_prices()
        - 基类会自动处理缓存，子类只需实现API调用
        
        Args:
            symbol: 股票/指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            PriceData: 价格数据对象
        
        Raises:
            Exception: API调用失败时抛出异常
        """
        pass
    
    def get_test_symbol(self) -> str:
        """
        获取测试符号（子类可重写）
        
        Returns:
            str: 测试用的股票/指数代码
        """
        return '^GSPC'  # 默认测试符号：标普500
    
    def initialize(self, **kwargs):
        """
        初始化方法（可选实现）
        
        子类可以重写此方法来进行额外的初始化工作，
        例如根据传入的参数初始化客户端连接等。
        
        Args:
            **kwargs: 初始化参数
        """
        pass
    
    # ========================================================================
    # 配置管理接口（具体方法，基类统一实现）
    # ========================================================================
    
    @staticmethod
    @classmethod
    def _get_config_path(cls, filename: str) -> Path:
        """
        获取配置文件路径
        
        Args:
            filename: 配置文件名
            
        Returns:
            Path: 配置文件完整路径
            
        Note:
            使用 ConfigManager.get_config_path() 统一获取配置路径
        """
        from core_bak_refactored.core.share.config_manager import ConfigManager
        config_manager = ConfigManager()
        # 使用 ConfigManager 的封装方法获取配置路径
        config_path_str = config_manager.get_config_path(filename.replace('.yml', ''))
        return Path(config_path_str)
    
    @classmethod
    def test_provider(cls, provider_id: str, credential:str) -> Dict[str, Any]:
        """
        测试数据源连接
        
        Args:
            provider_id: 数据源ID
            credential: 临时凭证（用于测试）
            
        Returns:
            测试结果字典
        """
        logger.info(f"Testing connection for provider: {provider_id}")
        
        try:
            # 获取数据源配置
            config_manager = ConfigManager()
            data_config = config_manager.get_data_config()
            providers = data_config.providers
            
            provider_config = next((p for p in providers if p.get('id') == provider_id or p.get('name') == provider_id), None)
            
            if not provider_config:
                return {
                    'status': 'error',
                    'test_result': 'failed',
                    'available': False,
                    'message': f'数据源不存在: {provider_id}',
                    'error_code': 'PROVIDER_NOT_FOUND'
                }
            
            # 动态创建适配器实例
            adapter_module = provider_config.get('adapter_module')
            adapter_class = provider_config.get('adapter_class')
            
            if not adapter_module or not adapter_class:
                return {
                    'status': 'error',
                    'test_result': 'failed',
                    'available': False,
                    'message': f'{provider_id} 适配器未实现',
                    'error_code': 'ADAPTER_NOT_IMPLEMENTED'
                }
            
            try:
                # 动态导入类
                module = __import__(adapter_module, fromlist=[adapter_class])
                provider_class = getattr(module, adapter_class)
                
                # 创建临时实例（各Provider从配置读取proxy，无需传参）
                test_instance = provider_class()
                
                # 如果Provider实现了initialize方法，调用它来初始化客户端
                if hasattr(test_instance, 'initialize'):
                    if credential:
                        test_instance.initialize(credential=credential)
                    else:
                        test_instance.initialize()
                
                # 使用适配器自定义的测试符号
                if hasattr(test_instance, 'get_test_symbol'):
                    test_symbol = test_instance.get_test_symbol()
                else:
                    test_symbol = '^GSPC'
                
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                start_time = time.time()
                
                # 执行测试查询
                test_data = test_instance.get_index_prices(test_symbol, start_date, end_date)
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # 处理PriceData对象
                if hasattr(test_data, 'to_dataframe'):
                    test_data_df = test_data.to_dataframe()
                    is_empty = test_data_df.empty
                    data_count = len(test_data_df)
                else:
                    is_empty = test_data.empty if test_data is not None else True
                    data_count = len(test_data) if test_data is not None else 0
                
                if test_data is None or is_empty:
                    # 测试失败：连接成功但返回空数据
                    is_available = False
                    message = f'{provider_id} 连接成功，但返回空数据'
                    logger.warning(f"{provider_id} 测试警告: {message}")
                    
                    result = {
                        'status': 'error',
                        'test_result': 'failed',
                        'available': is_available,
                        'message': message,
                        'details': {
                            'test_symbol': test_symbol,
                            'date_range': f'{start_date} to {end_date}',
                            'latency_ms': latency_ms
                        }
                    }
                else:
                    # 测试成功
                    is_available = True
                    message = f'{provider_id} 连接测试通过'
                    logger.info(f"{provider_id} 测试成功: {data_count} 条数据, {latency_ms}ms")
                    
                    result = {
                        'status': 'success',
                        'test_result': 'passed',
                        'available': is_available,
                        'message': message,
                        'details': {
                            'test_symbol': test_symbol,
                            'data_count': data_count,
                            'date_range': f'{start_date} to {end_date}',
                            'latency_ms': latency_ms
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 测试成功后，保存凭证到文件
                    if credential:
                        cls.save_credentials(provider_id, credential)
                        logger.info(f"{provider_id} 凭证已保存")
                    
                    # 💚 保存测试状态到配置文件（关键修复）
                    cls.save_test_status(provider_id, 'passed')
                    logger.info(f"{provider_id} 测试状态已保存: passed")
                
                return result
                
            except Exception as test_error:
                logger.error(f"测试 {provider_id} 连接失败: {test_error}")
                return {
                    'status': 'error',
                    'test_result': 'failed',
                    'available': False,
                    'message': f'{provider_id} 连接测试失败: {str(test_error)}',
                    'error_code': 'CONNECTION_TEST_FAILED'
                }
                
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            return {
                'status': 'error',
                'test_result': 'failed',
                'available': False,
                'message': str(e),
                'error_code': 'TEST_CONNECTION_FAILED'
            }
    
    @classmethod
    def save_credentials(
        cls,
        provider_id: str,
        credential: str,
    ) -> bool:
        """
        保存数据源凭证
        
        Args:
            provider_id: 数据源ID
            credential: 凭证数据
            
        Returns:
            bool: 是否成功
            
        Note:
            凭证保存后不再重置测试状态，状态由下次系统启动时的实时测试决定
        """
        try:
            credentials_yml_path = cls._get_config_path('credentials.yml')
            
            # 读取现有凭证
            if credentials_yml_path.exists():
                with open(credentials_yml_path, 'r', encoding='utf-8') as f:
                    credentials_data = yaml.safe_load(f) or {}
            else:
                credentials_data = {}
            
            # 更新凭证
            credentials_data[provider_id] = credential
            
            # 写入凭证文件
            credentials_yml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(credentials_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(credentials_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"保存 {provider_id} 凭证成功: {credential}")
            return True
            
        except Exception as e:
            logger.error(f"保存凭证失败: {e}")
            return False
    
    @classmethod
    def delete_credentials(
        cls,
        provider_id: str,
    ) -> bool:
        """
        删除数据源凭证
        
        Args:
            provider_id: 数据源ID
            
        Returns:
            bool: 是否成功
        
        Examples:
            >>> BaseDataProvider.delete_credentials('yahoo')
            True
            >>> BaseDataProvider.delete_credentials('nonexistent_provider')
            True  # 即使凭证不存在也返回 True
        
        Note:
            - 如果凭证文件不存在，返回 True（视为已删除）
            - 如果 provider_id 不存在，也返回 True（视为已删除）
            - 只有当文件操作失败时才返回 False
        """
        try:
            credentials_yml_path = cls._get_config_path('credentials.yml')
            
            # 如果凭证文件不存在，视为已删除
            if not credentials_yml_path.exists():
                logger.info(f"凭证文件不存在，视为已删除: {credentials_yml_path}")
                return True
            
            # 读取现有凭证
            with open(credentials_yml_path, 'r', encoding='utf-8') as f:
                credentials_data = yaml.safe_load(f) or {}
            
            # 如果 provider_id 不存在，视为已删除
            if provider_id not in credentials_data:
                logger.info(f"{provider_id} 凭证不存在，视为已删除")
                return True
            
            # 删除凭证
            del credentials_data[provider_id]
            
            # 写入文件
            with open(credentials_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(credentials_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"删除 {provider_id} 凭证成功")
            return True
            
        except Exception as e:
            logger.error(f"删除凭证失败: {e}")
            return False
    
    @classmethod
    def save_test_status(
        cls,
        provider_id: str,
        status: str,
    ) -> bool:
        """
        保存数据源测试状态到配置文件
        
        Args:
            provider_id: 数据源ID
            status: 测试状态 ('passed' | 'failed' | 'untested')
            
        Returns:
            bool: 是否成功
            
        Examples:
            >>> BaseDataProvider.save_test_status('yahoo', 'passed')
            True
            >>> BaseDataProvider.save_test_status('akshare', 'failed')
            True
        
        Note:
            - 状态保存到 data_provider.yml 中对应 provider 的 status 字段
            - 直接写入文件,确保持久化
        """
        try:
            from datetime import datetime
            from core_bak_refactored.core.share.config_manager import ConfigManager
            import yaml
            import os
            
            config_manager = ConfigManager()
            
            # 获取 data_provider.yml 的路径
            data_yml_path = config_manager.get_config_path('data')
            
            # 读取现有配置
            if os.path.exists(data_yml_path):
                with open(data_yml_path, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f) or {}
            else:
                logger.error(f"配置文件不存在: {data_yml_path}")
                return False
            
            # 查找并更新 provider 状态
            providers = data_config.get('providers', [])
            provider_found = False
            
            for provider in providers:
                if provider.get('id') == provider_id:
                    provider['status'] = status
                    provider['last_test'] = datetime.now().isoformat()
                    provider_found = True
                    logger.info(f"更新 provider 状态: {provider_id} -> {status}")
                    break
            
            if not provider_found:
                logger.warning(f"Provider {provider_id} 不存在于配置文件中")
                return False
            
            # 💚 关键修复: 写入文件,确保持久化
            with open(data_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"{provider_id} 测试状态已保存: {status}")
            
            # 重新加载配置(更新内存)
            config_manager._load_config()
            
            return True
            
        except Exception as e:
            logger.error(f"保存测试状态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False






