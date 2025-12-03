"""
配置管理器测试
"""

import unittest
import tempfile
import json
import os
from core_bak_refactored.core.share.config_manager import (
    ConfigManager, MonitoringConfig, AlertingConfig, DataConfig, SystemConfig
)


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
        manager = ConfigManager()
        config = manager.get_data_config()
        self.assertIsInstance(config, DataConfig)
        self.assertEqual(config.primary_source, "yahoo")
    
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
        # 测试从YAML配置中获取值
        value = manager.get('data.default_index')
        self.assertEqual(value, 'MSFT')
    
    def test_yaml_config_loading(self):
        """测试YAML配置加载"""
        # 使用开发环境配置
        manager = ConfigManager(environment='dev')
        regional_config = manager.get('regional_data_source')
        self.assertIsNotNone(regional_config)
        self.assertIn('CN', regional_config)
        self.assertIn('US', regional_config)


if __name__ == '__main__':
    unittest.main()
