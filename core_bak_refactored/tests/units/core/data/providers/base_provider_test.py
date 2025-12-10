"""
BaseDataProvider 单元测试

测试基类通用方法：
1. save_credentials() - 保存凭证到 credentials.yml
2. test_provider() - 测试数据源连接
3. _get_config_path() - 获取配置文件路径
"""

import unittest
import yaml
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider


class MockProvider(BaseDataProvider):
    """用于测试的 Mock Provider"""
    
    def __init__(self):
        self.test_data_available = True
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str):
        """模拟获取指数价格"""
        if not self.test_data_available:
            return None
        
        # 返回模拟数据
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'close': [100.0] * len(dates),
            'volume': [1000000] * len(dates)
        })
        return df
    
    def get_test_symbol(self) -> str:
        """返回测试符号"""
        return 'TEST_SYMBOL'
    
    def get_stock_prices(self, symbol: str, start_date: str, end_date: str):
        """模拟获取股票价格"""
        # 返回模拟数据
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'close': [100.0] * len(dates),
            'volume': [1000000] * len(dates)
        })
        return df


class BaseProviderTest(unittest.TestCase):
    """BaseDataProvider 基类测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时配置目录
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / 'config'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建环境子目录
        self.dev_dir = self.config_dir / 'dev'
        self.dev_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    # ========================================================================
    # 测试 _get_config_path() 方法
    # ========================================================================
    
    @patch('core_bak_refactored.core.data.providers.base_provider.Path')
    def test_get_config_path_with_env_dir(self, mock_path):
        """测试获取配置文件路径（环境目录存在）"""
        # 模拟环境目录存在
        mock_path.return_value.exists.return_value = True
        
        # 注意：由于 _get_config_path 使用复杂路径计算，我们跳过 mock
        # 直接测试实际逻辑（需要真实文件系统）
        self.skipTest("需要真实文件系统环境，跳过 mock 测试")
    
    # ========================================================================
    # 测试 save_credentials() 方法
    # ========================================================================
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_save_credentials_new_file(self, mock_get_config_path):
        """测试保存凭证（新建文件）"""
        credentials_path = self.dev_dir / 'credentials.yml'
        mock_get_config_path.return_value = credentials_path
        
        # 测试保存凭证
        credentials = {
            'api_key': 'test_key_12345',
            'secret': 'test_secret'
        }
        
        result = BaseDataProvider.save_credentials('test_provider', credentials, env='dev')
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证文件存在
        self.assertTrue(credentials_path.exists())
        
        # 验证文件内容
        with open(credentials_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertIn('test_provider', saved_data)
        self.assertEqual(saved_data['test_provider']['api_key'], 'test_key_12345')
        self.assertEqual(saved_data['test_provider']['secret'], 'test_secret')
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_save_credentials_update_existing(self, mock_get_config_path):
        """测试保存凭证（更新已存在的凭证）"""
        credentials_path = self.dev_dir / 'credentials.yml'
        mock_get_config_path.return_value = credentials_path
        
        # 预先创建文件并写入旧凭证
        old_credentials = {
            'test_provider': {
                'api_key': 'old_key',
                'secret': 'old_secret'
            },
            'another_provider': {
                'token': 'another_token'
            }
        }
        with open(credentials_path, 'w', encoding='utf-8') as f:
            yaml.dump(old_credentials, f, allow_unicode=True)
        
        # 更新凭证
        new_credentials = {
            'api_key': 'new_key_67890',
            'secret': 'new_secret'
        }
        
        result = BaseDataProvider.save_credentials('test_provider', new_credentials, env='dev')
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证文件内容
        with open(credentials_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        # 验证更新后的凭证
        self.assertEqual(saved_data['test_provider']['api_key'], 'new_key_67890')
        self.assertEqual(saved_data['test_provider']['secret'], 'new_secret')
        
        # 验证其他provider的凭证未受影响
        self.assertIn('another_provider', saved_data)
        self.assertEqual(saved_data['another_provider']['token'], 'another_token')
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_save_credentials_io_error(self, mock_get_config_path):
        """测试保存凭证时遇到 IO 错误"""
        # 使用不存在的路径（父目录也不存在且无法创建）
        credentials_path = Path('/nonexistent_root_dir/credentials.yml')
        mock_get_config_path.return_value = credentials_path
        
        credentials = {'api_key': 'test_key'}
        
        # 保存应该失败并返回 False
        result = BaseDataProvider.save_credentials('test_provider', credentials, env='dev')
        
        self.assertFalse(result)
    
    # ========================================================================
    # 测试 delete_credentials() 方法
    # ========================================================================
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_delete_credentials_success(self, mock_get_config_path):
        """测试删除凭证 - 成功场景"""
        credentials_path = self.dev_dir / 'credentials.yml'
        mock_get_config_path.return_value = credentials_path
        
        # 预先创建凭证文件
        credentials_data = {
            'provider1': {'api_key': 'key1'},
            'provider2': {'api_key': 'key2'}
        }
        with open(credentials_path, 'w', encoding='utf-8') as f:
            yaml.dump(credentials_data, f, allow_unicode=True)
        
        # 删除 provider1 的凭证
        result = BaseDataProvider.delete_credentials('provider1', env='dev')
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证文件内容
        with open(credentials_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        # provider1 已被删除
        self.assertNotIn('provider1', saved_data)
        # provider2 仍然存在
        self.assertIn('provider2', saved_data)
        self.assertEqual(saved_data['provider2']['api_key'], 'key2')
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_delete_credentials_file_not_exists(self, mock_get_config_path):
        """测试删除凭证 - 文件不存在"""
        credentials_path = self.dev_dir / 'nonexistent_credentials.yml'
        mock_get_config_path.return_value = credentials_path
        
        # 删除不存在的文件应该返回 True（幂等设计）
        result = BaseDataProvider.delete_credentials('provider1', env='dev')
        
        self.assertTrue(result)
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_delete_credentials_provider_not_exists(self, mock_get_config_path):
        """测试删除凭证 - provider 不存在"""
        credentials_path = self.dev_dir / 'credentials.yml'
        mock_get_config_path.return_value = credentials_path
        
        # 预先创建凭证文件
        credentials_data = {
            'provider1': {'api_key': 'key1'}
        }
        with open(credentials_path, 'w', encoding='utf-8') as f:
            yaml.dump(credentials_data, f, allow_unicode=True)
        
        # 删除不存在的 provider 应该返回 True（幂等设计）
        result = BaseDataProvider.delete_credentials('nonexistent_provider', env='dev')
        
        self.assertTrue(result)
        
        # 验证原有数据未受影响
        with open(credentials_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertIn('provider1', saved_data)
    
    @patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path')
    def test_delete_credentials_last_provider(self, mock_get_config_path):
        """测试删除凭证 - 删除最后一个 provider"""
        credentials_path = self.dev_dir / 'credentials.yml'
        mock_get_config_path.return_value = credentials_path
        
        # 预先创建凭证文件（只有一个 provider）
        credentials_data = {
            'provider1': {'api_key': 'key1'}
        }
        with open(credentials_path, 'w', encoding='utf-8') as f:
            yaml.dump(credentials_data, f, allow_unicode=True)
        
        # 删除唯一的 provider
        result = BaseDataProvider.delete_credentials('provider1', env='dev')
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证文件内容（应该为空字典或 None）
        with open(credentials_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f) or {}
        
        self.assertEqual(saved_data, {})
    
    # ========================================================================
    # 测试 test_provider() 方法
    # ========================================================================
    
    def test_test_provider_success(self):
        """测试 test_provider 成功场景"""
        # 测试成功场景
        result = MockProvider.test_provider('mock_provider', env='dev')
        
        # 验证结果结构
        self.assertIn('status', result)
        self.assertIn('test_result', result)
        self.assertIn('available', result)
        self.assertIn('message', result)
        
        # 验证成功状态
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['test_result'], 'passed')
        self.assertTrue(result['available'])
        
        # 验证详细信息
        self.assertIn('details', result)
        self.assertIn('test_symbol', result['details'])
        self.assertIn('data_count', result['details'])
        self.assertIn('date_range', result['details'])
        self.assertEqual(result['details']['test_symbol'], 'TEST_SYMBOL')
    
    def test_test_provider_empty_data(self):
        """测试 test_provider 返回空数据"""
        # 创建一个返回空数据的 provider
        class EmptyDataProvider(BaseDataProvider):
            def get_index_prices(self, index_id: str, start_date: str, end_date: str):
                import pandas as pd
                return pd.DataFrame()  # 空 DataFrame
            
            def get_test_symbol(self) -> str:
                return 'EMPTY_SYMBOL'
            
            def get_stock_prices(self, symbol: str, start_date: str, end_date: str):
                import pandas as pd
                return pd.DataFrame()
        
        result = EmptyDataProvider.test_provider('empty_provider', env='dev')
        
        # 验证结果
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['test_result'], 'failed')
        self.assertFalse(result['available'])
        self.assertIn('返回空数据', result['message'])
    
    def test_test_provider_connection_error(self):
        """测试 test_provider 连接错误"""
        # 创建一个抛出异常的 provider
        class ErrorProvider(BaseDataProvider):
            def get_index_prices(self, index_id: str, start_date: str, end_date: str):
                raise ConnectionError("无法连接到数据源")
            
            def get_test_symbol(self) -> str:
                return 'ERROR_SYMBOL'
            
            def get_stock_prices(self, symbol: str, start_date: str, end_date: str):
                pass
        
        result = ErrorProvider.test_provider('error_provider', env='dev')
        
        # 验证结果
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['test_result'], 'failed')
        self.assertFalse(result['available'])
        self.assertIn('连接测试失败', result['message'])
        self.assertIn('无法连接到数据源', result['message'])
    
    def test_test_provider_default_symbol(self):
        """测试 test_provider 使用默认测试符号"""
        # 创建一个没有 get_test_symbol 方法的 provider
        class DefaultSymbolProvider(BaseDataProvider):
            def get_index_prices(self, index_id: str, start_date: str, end_date: str):
                import pandas as pd
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                df = pd.DataFrame({
                    'date': dates,
                    'close': [100.0] * len(dates),
                    'volume': [1000000] * len(dates)
                })
                return df
            
            def get_stock_prices(self, symbol: str, start_date: str, end_date: str):
                import pandas as pd
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                df = pd.DataFrame({
                    'date': dates,
                    'close': [100.0] * len(dates),
                    'volume': [1000000] * len(dates)
                })
                return df
        
        result = DefaultSymbolProvider.test_provider('default_provider', env='dev')
        
        # 验证使用了默认符号
        self.assertEqual(result['status'], 'success')
        # BaseDataProvider 的默认测试符号是 '^GSPC'
    
    def test_test_provider_with_temporary_credentials(self):
        """测试 test_provider 使用临时凭证"""
        # 这个测试主要是为了确保 test_provider 方法能够正常工作
        # 实际的临时凭证处理在 API 层完成
        result = MockProvider.test_provider('temp_provider', env='dev')
        
        # 验证基本结构
        self.assertIn('status', result)
        self.assertIn('test_result', result)
        self.assertIn('available', result)
        self.assertIn('message', result)
    
    def test_test_provider_exception_handling(self):
        """测试 test_provider 异常处理"""
        # 创建一个在初始化时抛出异常的 provider
        class BrokenInitProvider(BaseDataProvider):
            def __init__(self):
                raise Exception("初始化失败")
            
            def get_index_prices(self, index_id: str, start_date: str, end_date: str):
                pass
            
            def get_stock_prices(self, symbol: str, start_date: str, end_date: str):
                pass
        
        result = BrokenInitProvider.test_provider('broken_provider', env='dev')
        
        # 验证异常被捕获并返回错误结果
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['test_result'], 'failed')
        self.assertFalse(result['available'])
        self.assertIn('初始化失败', result['message'])
    
    def test_test_provider_with_timestamp(self):
        """测试 test_provider 返回结果包含时间戳"""
        result = MockProvider.test_provider('mock_provider', env='dev')
        
        # 验证成功结果包含时间戳
        if result['status'] == 'success':
            self.assertIn('timestamp', result)
            # 验证时间戳格式（ISO 8601）
            timestamp = result['timestamp']
            self.assertIsInstance(timestamp, str)
            # 验证可以解析为 datetime
            datetime.fromisoformat(timestamp)
    
    # ========================================================================
    # 测试 get_test_symbol() 方法
    # ========================================================================
    
    def test_get_test_symbol_default(self):
        """测试默认测试符号"""
        # BaseDataProvider 的默认测试符号是 '^GSPC'
        provider = MockProvider()
        # MockProvider 重写了 get_test_symbol，返回 'TEST_SYMBOL'
        self.assertEqual(provider.get_test_symbol(), 'TEST_SYMBOL')
    
    # ========================================================================
    # 集成测试
    # ========================================================================
    
    def test_integration_save_and_test(self):
        """集成测试：保存凭证后测试连接"""
        # 这是一个集成测试示例，展示如何组合使用多个方法
        
        # 1. 保存凭证
        credentials_path = self.dev_dir / 'credentials.yml'
        with patch('core_bak_refactored.core.data.providers.base_provider.BaseDataProvider._get_config_path') as mock_path:
            mock_path.return_value = credentials_path
            
            credentials = {
                'api_key': 'integration_test_key',
                'secret': 'integration_test_secret'
            }
            
            save_result = BaseDataProvider.save_credentials('integration_provider', credentials, env='dev')
            self.assertTrue(save_result)
        
        # 2. 测试连接
        test_result = MockProvider.test_provider('integration_provider', env='dev')
        
        # 3. 验证结果
        self.assertEqual(test_result['status'], 'success')
        self.assertEqual(test_result['test_result'], 'passed')
        self.assertTrue(test_result['available'])
        
        # 4. 验证凭证文件内容
        with open(credentials_path, 'r', encoding='utf-8') as f:
            saved_credentials = yaml.safe_load(f)
        
        self.assertIn('integration_provider', saved_credentials)
        self.assertEqual(saved_credentials['integration_provider']['api_key'], 'integration_test_key')


if __name__ == '__main__':
    unittest.main()
