"""
数据提供者测试套件
测试所有数据提供者的代理配置和基本功能
"""

import sys
import os
import unittest

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)


class TestDataProviders(unittest.TestCase):
    """数据提供者测试类"""
    
    def test_proxy_config_loading(self):
        """测试代理配置加载"""
        try:
            # 测试导入配置管理器
            from core_bak_refactored.infrastructure.config_manager import ConfigManager
            
            # 测试获取系统配置
            config_manager = ConfigManager()
            system_config = config_manager.get_system_config()
            
            # 测试代理工具函数
            from core_bak_refactored.core.data.providers.utils import get_proxies_from_config
            proxies = get_proxies_from_config()
            
            self.assertIsNotNone(system_config)
            self.assertIsNotNone(proxies)
            
        except Exception as e:
            self.fail(f"代理配置测试失败: {e}")
    
    def test_yahoo_finance_provider(self):
        """测试Yahoo Finance Provider"""
        try:
            from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
            
            # 创建Provider实例
            provider = YahooFinanceDataProvider()
            
            self.assertTrue(hasattr(provider, 'available'))
            self.assertTrue(hasattr(provider, 'proxy'))
            
        except Exception as e:
            self.fail(f"Yahoo Finance Provider测试失败: {e}")
    
    def test_akshare_provider(self):
        """测试Akshare Provider"""
        try:
            from core_bak_refactored.core.data.providers.akshare_provider import AKShareDataProvider
            
            # 创建Provider实例
            provider = AKShareDataProvider()
            
            self.assertTrue(hasattr(provider, 'available'))
            self.assertTrue(hasattr(provider, 'proxy'))
            
        except Exception as e:
            self.fail(f"Akshare Provider测试失败: {e}")
    
    def test_tushare_provider(self):
        """测试Tushare Provider"""
        try:
            from core_bak_refactored.core.data.providers.tushare_provider import TushareDataProvider
            
            # 创建Provider实例
            provider = TushareDataProvider()
            
            self.assertTrue(hasattr(provider, 'available'))
            self.assertTrue(hasattr(provider, 'proxy'))
            
        except Exception as e:
            self.fail(f"Tushare Provider测试失败: {e}")


if __name__ == '__main__':
    unittest.main()