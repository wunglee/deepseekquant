"""
Provider 基类 - 确保所有 Provider 实现统一的配置管理接口
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from core_bak_refactored.core.share import ConfigManager

logger = logging.getLogger('DeepSeekQuant.DataProviders')


class BaseDataProvider(ABC):
    """
    数据提供者基类
    
    定义所有 Provider 必须实现的接口，包括：
    1. 数据获取接口（抽象方法，子类必须实现）
    2. 配置管理接口（具体方法，基类统一实现）
    """
    
    # ========================================================================
    # 数据获取接口（抽象方法，子类必须实现）
    # ========================================================================
    @abstractmethod
    def get_index_prices(self, index_id: str, start_date: str, end_date: str):
        """
        获取指数价格数据（抽象方法）

        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 包含 date, close, volume 等列的数据
        """
        pass
    @abstractmethod
    def get_stock_prices(self, index_id: str, start_date: str, end_date: str):
        """
        获取指数价格数据（抽象方法）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 包含 date, close, volume 等列的数据
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
            - 状态保存到 data.yml 中对应 provider 的 status 字段
            - 直接写入文件,确保持久化
        """
        try:
            from datetime import datetime
            from core_bak_refactored.core.share.config_manager import ConfigManager
            import yaml
            import os
            
            config_manager = ConfigManager()
            
            # 获取 data.yml 的路径
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


