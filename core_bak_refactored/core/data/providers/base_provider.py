"""
Provider 基类 - 确保所有 Provider 实现统一的配置管理接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from datetime import datetime

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
    def _get_config_path(filename: str, env: str = 'dev') -> Path:
        """
        获取配置文件路径
        
        Args:
            filename: 配置文件名
            env: 环境（dev/prod/test）
            
        Returns:
            Path: 配置文件完整路径
        """
        # 从当前文件向上找到 core_bak_refactored/config
        current_file = Path(__file__).resolve()
        core_bak_dir = current_file.parent.parent.parent.parent
        config_dir = core_bak_dir / 'config'
        
        # 优先使用环境目录
        env_dir = config_dir / env
        env_path = env_dir / filename
        if env_path.exists():
            return env_path
        return config_dir / filename
    
    @classmethod
    def test_provider(
        cls,
        provider_id: str,
        env: str = 'dev',
        **test_args
    ) -> Dict[str, Any]:
        """
        测试数据源连接（简化版，移除 RealHistoricalDataProvider 特殊逻辑）
        
        Args:
            provider_id: 数据源ID
            env: 环境（dev/prod/test）
            **test_args: 测试参数，例如api_key等
            
        Returns:
            Dict: 测试结果
                - status: 'success'/'error'
                - test_result: 'passed'/'failed'
                - available: bool (可用/不可用)
                - message: 提示信息
                - details: 详细信息
        
        Note:
            此方法已废弃，现在测试逻辑由 API 端点直接处理
        """
        try:
            # 创建临时实例用于测试
            test_instance = cls()
            
            # 如果有测试参数，调用initialize方法进行初始化
            if test_args:
                test_instance.initialize(**test_args)
            
            # 获取测试符号
            if hasattr(test_instance, 'get_test_symbol'):
                test_symbol = test_instance.get_test_symbol()
            else:
                test_symbol = '^GSPC'
            
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            try:
                test_data = test_instance.get_index_prices(test_symbol, start_date, end_date)
                
                if hasattr(test_data, 'to_dataframe'):
                    test_data_df = test_data.to_dataframe()
                    is_empty = test_data_df.empty
                    data_count = len(test_data_df)
                else:
                    is_empty = test_data.empty if test_data is not None else True
                    data_count = len(test_data) if test_data is not None else 0
                
                if test_data is None or is_empty:
                    result = {
                        'status': 'error',
                        'test_result': 'failed',
                        'available': False,
                        'message': f'{provider_id} 连接成功，但返回空数据',
                        'details': {
                            'test_symbol': test_symbol,
                            'date_range': f'{start_date} to {end_date}'
                        }
                    }
                else:
                    logger.info(f"测试 {provider_id} 连接成功，返回 {data_count} 条数据")
                    result = {
                        'status': 'success',
                        'test_result': 'passed',
                        'available': True,
                        'message': f'{provider_id} 连接测试通过',
                        'details': {
                            'test_symbol': test_symbol,
                            'data_count': data_count,
                            'date_range': f'{start_date} to {end_date}'
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                
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
        credentials: Dict[str, Any],
        env: str = 'dev'
    ) -> bool:
        """
        保存数据源凭证
        
        Args:
            provider_id: 数据源ID
            credentials: 凭证数据
            env: 环境（dev/prod/test）
            
        Returns:
            bool: 是否成功
            
        Note:
            凭证保存后不再重置测试状态，状态由下次系统启动时的实时测试决定
        """
        try:
            credentials_yml_path = cls._get_config_path('credentials.yml', env)
            
            # 读取现有凭证
            if credentials_yml_path.exists():
                with open(credentials_yml_path, 'r', encoding='utf-8') as f:
                    credentials_data = yaml.safe_load(f) or {}
            else:
                credentials_data = {}
            
            # 更新凭证
            credentials_data[provider_id] = credentials
            
            # 写入凭证文件
            credentials_yml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(credentials_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(credentials_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"保存 {provider_id} 凭证成功: {list(credentials.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"保存凭证失败: {e}")
            return False
    
    @classmethod
    def delete_credentials(
        cls,
        provider_id: str,
        env: str = 'dev'
    ) -> bool:
        """
        删除数据源凭证
        
        Args:
            provider_id: 数据源ID
            env: 环境（dev/prod/test）
            
        Returns:
            bool: 是否成功
        
        Examples:
            >>> BaseDataProvider.delete_credentials('yahoo', env='dev')
            True
            >>> BaseDataProvider.delete_credentials('nonexistent_provider')
            True  # 即使凭证不存在也返回 True
        
        Note:
            - 如果凭证文件不存在，返回 True（视为已删除）
            - 如果 provider_id 不存在，也返回 True（视为已删除）
            - 只有当文件操作失败时才返回 False
        """
        try:
            credentials_yml_path = cls._get_config_path('credentials.yml', env)
            
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
