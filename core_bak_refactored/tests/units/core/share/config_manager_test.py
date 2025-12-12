""" 
配置管理器测试
"""

import unittest
import tempfile
import json
import os
import yaml
from pathlib import Path
from core_bak_refactored.core.share.config_manager import (
    ConfigManager, MonitoringConfig, AlertingConfig, DataConfig, SystemConfig
)
from core_bak_refactored.core.share.market.market_enums import MarketCode


class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.config_file = self.temp_file.name
        self.temp_file.close()
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        manager = ConfigManager()
        self.assertIsNotNone(manager._config)
    
    def test_get_monitoring_config(self):
        """测试获取监控配置"""
        manager = ConfigManager()
        config = manager.get_monitoring_config()
        self.assertIsInstance(config, MonitoringConfig)
        self.assertEqual(config.check_interval, 300)
    
    def test_get_alerting_config(self):
        """测试获取告警配置"""
        manager = ConfigManager()
        config = manager.get_alerting_config()
        self.assertIsInstance(config, AlertingConfig)
    
    def test_get_data_config(self):
        """测试获取数据配置"""
        # 默认环境是dev，使用dev/data.yml中的配置
        manager = ConfigManager()
        config = manager.get_data_config()
        self.assertIsInstance(config, DataConfig)
        # dev环境使用 market_sources 映射，不再有 primary_source
        self.assertIsNotNone(config.market_sources)
        # 验证 CN 市场使用 akshare
        self.assertEqual(config.market_sources.get('CN'), 'akshare')
    
    def test_get_system_config(self):
        """测试获取系统配置"""
        manager = ConfigManager()
        config = manager.get_system_config()
        self.assertIsInstance(config, SystemConfig)
    
    def test_get_nested_key(self):
        """测试获取嵌套键"""
        manager = ConfigManager()
        value = manager.get('monitoring.check_interval', 100)
        self.assertEqual(value, 300)
    
    def test_set_nested_key(self):
        """测试设置嵌套键"""
        manager = ConfigManager()
        manager.set('monitoring.check_interval', 600)
        value = manager.get('monitoring.check_interval')
        self.assertEqual(value, 600)
    
    def test_save_config(self):
        """测试保存配置"""
        manager = ConfigManager()
        manager.set('test_key', 'test_value')
        manager.save(self.config_file)
        
        with open(self.config_file, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config['test_key'], 'test_value')
    
    def test_load_config(self):
        """测试加载配置"""
        # 在测试环境中，我们不测试JSON文件加载，因为YAML配置优先级更高
        # 我们测试get方法的基本功能
        manager = ConfigManager(environment='test')
        # 测试从YAML配置中获取值（test/data.yml 中 default_index 为 SPX）
        value = manager.get('data.default_index')
        self.assertEqual(value, 'SPX')
    
    def test_yaml_config_loading(self):
        """测试YAML配置加载"""
        # 使用开发环境配置
        manager = ConfigManager(environment='dev')
        # 新配置中使用 market_sources 而不是 regional_data_source
        market_sources = manager.get('data.market_sources')
        self.assertIsNotNone(market_sources)
        # 验证 CN 和 US 市场配置存在
        self.assertIn('CN', market_sources)
        self.assertIn('US', market_sources)
    
    def test_environment_encapsulation(self):
        """测试环境封装（外部不应直接获取环境）"""
        # 外部不应直接获取环境，应通过 ConfigManager 实例获取配置
        manager = ConfigManager()
        
        # 验证可以获取配置（环境已被封装）
        data_config = manager.get('data')
        self.assertIsNotNone(data_config)
        
        # 验证 _get_environment() 是私有方法
        self.assertTrue(hasattr(ConfigManager, '_get_environment'))
        
        # 测试环境切换（通过创建新实例）
        original_env = os.environ.get('DEEPSEEK_ENV')
        try:
            os.environ['DEEPSEEK_ENV'] = 'test'
            test_manager = ConfigManager(environment='test')
            test_value = test_manager.get('data.default_index')
            self.assertEqual(test_value, 'SPX')  # test 环境的配置
        finally:
            # 恢复原始环境
            if original_env:
                os.environ['DEEPSEEK_ENV'] = original_env
            elif 'DEEPSEEK_ENV' in os.environ:
                del os.environ['DEEPSEEK_ENV']


if __name__ == '__main__':
    unittest.main()
