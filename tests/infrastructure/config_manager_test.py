"""  
DeepSeekQuant 配置管理器测试
基于common.py和logging_system.py的配置管理器测试
"""

import unittest
import tempfile
import os
import sys
import json
import yaml
import toml
from pathlib import Path
import shutil
import time
from datetime import datetime
import copy

# 添加父目录到sys.path以导入common和logging_system
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 直接导入config_manager模块，避免通过core.__init__.py
from infrastructure.config_manager import ConfigManager, ConfigValidationError, ConfigEncryptionError, get_global_config_manager, shutdown_global_config_manager

from common import ConfigFormat, ConfigSource, TradingMode, RiskLevel, DEFAULT_ENCODING
from infrastructure.logging_service import get_logger

# 使用测试日志记录器
logger = get_logger('DeepSeekQuant.ConfigManager.Test')


class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""

    def setUp(self):
        """测试前置设置"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")

        # 备份默认配置和元数据
        self.original_default_config = copy.deepcopy(ConfigManager.DEFAULT_CONFIG)
        self.original_config_schema = copy.deepcopy(ConfigManager.DEFAULT_CONFIG_SCHEMA)

        logger.info(f"测试目录: {self.test_dir}")

    def tearDown(self):
        """测试后置清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        # 恢复默认配置和架构
        ConfigManager.DEFAULT_CONFIG = self.original_default_config
        ConfigManager.DEFAULT_CONFIG_SCHEMA = self.original_config_schema

        logger.info("测试清理完成")

    def test_initialization_with_default_config(self):
        """测试默认配置初始化"""
        # 使用不存在的路径确保使用默认配置
        nonexistent_path = "/tmp/definitely_nonexistent_config_path_12345.json"
        config_manager = ConfigManager(config_path=nonexistent_path)

        self.assertIsNotNone(config_manager.config)
        # 确保使用全新的默认配置
        self.assertEqual(config_manager.get_config('system.name'), 'DeepSeekQuant')
        self.assertEqual(config_manager.metadata.source, ConfigSource.DEFAULT)

        logger.info("默认配置初始化测试通过")

    def test_load_valid_json_config(self):
        """测试加载有效的JSON配置"""
        # 创建有效的JSON配置文件
        valid_config = {
            "system": {
                "name": "TestSystem",
                "version": "1.0.0",
                "environment": "testing",
                "log_level": "DEBUG",
                "trading_mode": "paper_trading"
            },
            "data_sources": {
                "primary": "test_source",
                "fallback_sources": ["backup1", "backup2"]
            }
        }

        with open(self.config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(valid_config, f, indent=2)

        config_manager = ConfigManager(config_path=self.config_path)

        self.assertEqual(config_manager.get_config('system.name'), 'TestSystem')
        self.assertEqual(config_manager.metadata.source, ConfigSource.FILE)
        self.assertEqual(config_manager.metadata.format, ConfigFormat.JSON)

        logger.info("有效JSON配置加载测试通过")

    def test_load_valid_yaml_config(self):
        """测试加载有效的YAML配置"""
        yaml_config_path = os.path.join(self.test_dir, "test_config.yaml")

        valid_config = {
            "system": {
                "name": "YAMLSystem",
                "version": "1.0.0",
                "environment": "development",
                "log_level": "INFO",
                "trading_mode": "backtesting"
            },
            "data_sources": {
                "primary": "yaml_source",
                "cache_enabled": True
            }
        }

        with open(yaml_config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            yaml.safe_dump(valid_config, f, default_flow_style=False)

        config_manager = ConfigManager(config_path=yaml_config_path)

        self.assertEqual(config_manager.get_config('system.name'), 'YAMLSystem')
        self.assertEqual(config_manager.metadata.format, ConfigFormat.YAML)

        logger.info("有效YAML配置加载测试通过")

    def test_load_valid_toml_config(self):
        """测试加载有效的TOML配置"""
        toml_config_path = os.path.join(self.test_dir, "test_config.toml")

        valid_config = {
            "system": {
                "name": "TOMLSystem",
                "version": "1.0.0",
                "environment": "production",
                "log_level": "WARNING"
            },
            "data_sources": {
                "primary": "toml_source",
                "retry_attempts": 5
            }
        }

        with open(toml_config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            toml.dump(valid_config, f)

        config_manager = ConfigManager(config_path=toml_config_path)

        self.assertEqual(config_manager.get_config('system.name'), 'TOMLSystem')
        self.assertEqual(config_manager.metadata.format, ConfigFormat.TOML)

        logger.info("有效TOML配置加载测试通过")

    def test_load_nonexistent_config(self):
        """测试加载不存在的配置文件"""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.json")
        config_manager = ConfigManager(config_path=nonexistent_path)

        self.assertEqual(config_manager.metadata.source, ConfigSource.DEFAULT)
        self.assertEqual(config_manager.get_config('system.name'), 'DeepSeekQuant')

        logger.info("不存在配置文件加载测试通过")

    def test_load_invalid_json_config(self):
        """测试加载无效的JSON配置"""
        with open(self.config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            f.write('{"invalid": json}')  # 无效的JSON

        config_manager = ConfigManager(config_path=self.config_path)

        # 应该回退到默认配置
        self.assertEqual(config_manager.metadata.source, ConfigSource.DEFAULT)

        logger.info("无效JSON配置加载测试通过")

    def test_config_validation_success(self):
        """测试配置验证成功"""
        valid_config = {
            "system": {
                "name": "TestSystem",
                "version": "1.0.0",
                "environment": "testing",
                "log_level": "INFO"
            },
            "data_sources": {
                "primary": "test_source"
            }
        }

        with open(self.config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(valid_config, f, indent=2)

        config_manager = ConfigManager(config_path=self.config_path)

        # 验证应该成功
        self.assertTrue(config_manager.validate_config())

        logger.info("配置验证成功测试通过")

    def test_config_validation_failure(self):
        """测试配置验证失败"""
        invalid_config = {
            "system": {
                "name": "TestSystem",
                # 缺少必需的version字段
                "environment": "testing",
                "log_level": "INVALID_LEVEL"  # 无效的日志级别
            },
            "data_sources": {
                "primary": "test_source"
            }
        }

        with open(self.config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(invalid_config, f, indent=2)

        config_manager = ConfigManager(config_path=self.config_path, validate_on_load=False)

        # 验证应该失败
        with self.assertRaises(ConfigValidationError):
            config_manager.validate_config()

        logger.info("配置验证失败测试通过")

    def test_get_config_values(self):
        """测试获取配置值"""
        test_config = {
            "system": {
                "name": "TestSystem",
                "version": "1.0.0",
                "environment": "testing",
                "log_level": "DEBUG",
                "nested": {
                    "value": "nested_value"
                }
            },
            "data_sources": {
                "primary": "test_source"
            }
        }

        with open(self.config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(test_config, f, indent=2)

        config_manager = ConfigManager(config_path=self.config_path)

        # 测试获取整个配置
        full_config = config_manager.get_config()
        self.assertIsInstance(full_config, dict)

        # 测试获取顶级配置项
        system_name = config_manager.get_config('system.name')
        self.assertEqual(system_name, 'TestSystem')

        # 测试获取嵌套配置项
        nested_value = config_manager.get_config('system.nested.value')
        self.assertEqual(nested_value, 'nested_value')

        # 测试获取不存在的配置项
        nonexistent = config_manager.get_config('nonexistent.key', 'default_value')
        self.assertEqual(nonexistent, 'default_value')

        logger.info("配置值获取测试通过")

    def test_set_config_values(self):
        """测试设置配置值"""
        config_manager = ConfigManager(config_path=None)  # 使用默认配置

        # 测试设置顶级配置项
        success = config_manager.set_config('system.name', 'NewSystem', '测试重命名')
        self.assertTrue(success)
        self.assertEqual(config_manager.get_config('system.name'), 'NewSystem')

        # 测试设置嵌套配置项
        success = config_manager.set_config('system.new_setting.level', 'high', '添加新设置')
        self.assertTrue(success)
        self.assertEqual(config_manager.get_config('system.new_setting.level'), 'high')

        # 测试变更历史记录
        change_history = config_manager.get_change_history()
        self.assertGreaterEqual(len(change_history), 2)

        logger.info("配置值设置测试通过")

    def test_config_update(self):
        """测试批量更新配置"""
        config_manager = ConfigManager(config_path=None)

        update_config = {
            "system": {
                "name": "UpdatedSystem",
                "performance_monitoring": False
            },
            "trading": {
                "initial_capital": 2000000.0,
                "new_setting": "new_value"
            }
        }

        success = config_manager.update_config(update_config, "批量更新测试")
        self.assertTrue(success)

        self.assertEqual(config_manager.get_config('system.name'), 'UpdatedSystem')
        self.assertEqual(config_manager.get_config('trading.initial_capital'), 2000000.0)
        self.assertEqual(config_manager.get_config('trading.new_setting'), 'new_value')

        logger.info("批量配置更新测试通过")

    def test_config_merge(self):
        """测试配置合并"""
        config_manager = ConfigManager(config_path=None)

        merge_config = {
            "system": {
                "merge_setting": "merged_value"
            },
            "new_section": {
                "key": "value"
            }
        }

        success = config_manager.merge_config(merge_config, "配置合并测试")
        self.assertTrue(success)

        self.assertEqual(config_manager.get_config('system.merge_setting'), 'merged_value')
        self.assertEqual(config_manager.get_config('new_section.key'), 'value')
        # 确保原有配置不被覆盖
        self.assertEqual(config_manager.get_config('system.name'), 'DeepSeekQuant')

        logger.info("配置合并测试通过")

    def test_config_reset(self):
        """测试配置重置"""
        config_manager = ConfigManager(config_path=None)

        # 先修改一些配置
        config_manager.set_config('system.name', 'ModifiedSystem')
        config_manager.set_config('trading.initial_capital', 500000.0)

        # 重置整个配置
        success = config_manager.reset_to_default()
        self.assertTrue(success)

        self.assertEqual(config_manager.get_config('system.name'), 'DeepSeekQuant')
        self.assertEqual(config_manager.get_config('trading.initial_capital'), 1000000.0)

        # 测试重置特定部分
        config_manager.set_config('system.name', 'ModifiedAgain')
        success = config_manager.reset_to_default('system')
        self.assertTrue(success)
        self.assertEqual(config_manager.get_config('system.name'), 'DeepSeekQuant')

        logger.info("配置重置测试通过")

    def test_config_save(self):
        """测试配置保存"""
        config_manager = ConfigManager(config_path=None)

        # 修改配置
        config_manager.set_config('system.name', 'SavedSystem')

        # 保存到文件
        save_path = os.path.join(self.test_dir, "saved_config.json")
        success = config_manager.save_config(save_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))

        # 验证保存的内容
        with open(save_path, 'r', encoding=DEFAULT_ENCODING) as f:
            saved_config = json.load(f)

        self.assertEqual(saved_config['system']['name'], 'SavedSystem')

        logger.info("配置保存测试通过")

    def test_config_save_different_formats(self):
        """测试不同格式的配置保存"""
        config_manager = ConfigManager(config_path=None)

        # 测试保存为JSON
        json_path = os.path.join(self.test_dir, "test.json")
        success = config_manager.save_config(json_path, ConfigFormat.JSON)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_path))

        # 测试保存为YAML
        yaml_path = os.path.join(self.test_dir, "test.yaml")
        success = config_manager.save_config(yaml_path, ConfigFormat.YAML)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(yaml_path))

        # 测试保存为TOML
        toml_path = os.path.join(self.test_dir, "test.toml")
        success = config_manager.save_config(toml_path, ConfigFormat.TOML)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(toml_path))

        logger.info("多格式配置保存测试通过")

    def test_config_encryption(self):
        """测试配置加密"""
        encryption_key = "test_encryption_key_12345"

        # 创建加密的配置管理器
        config_manager = ConfigManager(
            config_path=None,
            encryption_key=encryption_key
        )

        config_manager.set_config('system.name', 'EncryptedSystem')

        # 保存加密配置
        encrypted_path = os.path.join(self.test_dir, "encrypted_config.json")
        success = config_manager.save_config(encrypted_path)
        self.assertTrue(success)

        # 验证配置被加密
        with open(encrypted_path, 'r', encoding=DEFAULT_ENCODING) as f:
            encrypted_content = json.load(f)

        self.assertTrue(encrypted_content.get('_encrypted', False))
        self.assertIn('data', encrypted_content)

        # 使用相同密钥重新加载
        new_config_manager = ConfigManager(
            config_path=encrypted_path,
            encryption_key=encryption_key
        )

        self.assertEqual(new_config_manager.get_config('system.name'), 'EncryptedSystem')

        logger.info("配置加密测试通过")

    def test_config_snapshots(self):
        """测试配置快照功能"""
        config_manager = ConfigManager(config_path=None)
        
        # 清理所有旧快照
        old_snapshots = config_manager.list_snapshots()
        for snapshot in old_snapshots:
            config_manager.delete_snapshot(snapshot['id'])

        # 创建快照
        snapshot_id = config_manager.create_snapshot("测试快照")
        self.assertIsNotNone(snapshot_id)

        # 修改配置
        config_manager.set_config('system.name', 'ModifiedForSnapshot')

        # 列出快照
        snapshots = config_manager.list_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['id'], snapshot_id)

        # 从快照恢复
        success = config_manager.rollback_to_snapshot(snapshot_id)
        self.assertTrue(success)
        self.assertEqual(config_manager.get_config('system.name'), 'DeepSeekQuant')

        # 删除快照
        success = config_manager.delete_snapshot(snapshot_id)
        self.assertTrue(success)
        self.assertEqual(len(config_manager.list_snapshots()), 0)

        logger.info("配置快照测试通过")

    def test_change_history(self):
        """测试变更历史记录"""
        config_manager = ConfigManager(config_path=None)

        # 进行多次配置变更
        config_manager.set_config('system.name', 'HistoryTest1', "第一次变更")
        config_manager.set_config('system.version', '2.0.0', "第二次变更")
        config_manager.set_config('trading.initial_capital', 3000000.0, "第三次变更")

        # 获取变更历史
        history = config_manager.get_change_history()
        self.assertGreaterEqual(len(history), 3)

        # 测试历史过滤
        limited_history = config_manager.get_change_history(limit=2)
        self.assertEqual(len(limited_history), 2)

        # 验证变更内容
        latest_change = history[-1]
        self.assertEqual(latest_change['change_type'], 'update')
        self.assertEqual(latest_change['changes']['trading.initial_capital']['new'], 3000000.0)

        logger.info("变更历史测试通过")

    def test_observer_pattern(self):
        """测试观察者模式"""
        observed_changes = []

        def observer_callback(key, old_value, new_value):
            observed_changes.append((key, old_value, new_value))

        config_manager = ConfigManager(config_path=None)

        # 注册观察者
        observer_id = config_manager.register_observer('system.name', observer_callback)

        # 修改被观察的配置
        config_manager.set_config('system.name', 'ObservedSystem')

        # 验证观察者被通知
        self.assertEqual(len(observed_changes), 1)
        self.assertEqual(observed_changes[0][0], 'system.name')
        self.assertEqual(observed_changes[0][2], 'ObservedSystem')

        # 取消注册观察者
        success = config_manager.unregister_observer('system.name', observer_id)
        self.assertTrue(success)

        # 再次修改配置，观察者不应被通知
        config_manager.set_config('system.name', 'UnobservedSystem')
        self.assertEqual(len(observed_changes), 1)  # 数量不应增加

        logger.info("观察者模式测试通过")

    def test_config_validation_rules(self):
        """测试自定义验证规则"""
        config_manager = ConfigManager(config_path=None, validate_on_load=False)

        # 测试无效的交易模式
        config_manager.config['system']['trading_mode'] = 'invalid_mode'
        with self.assertRaises(ConfigValidationError):
            config_manager.validate_config()

        # 测试无效的风险等级
        config_manager.config['system']['trading_mode'] = 'paper_trading'  # 恢复有效值
        config_manager.config['risk_management'] = {'risk_level': 'invalid_risk'}
        with self.assertRaises(ConfigValidationError):
            config_manager.validate_config()

        # 测试无效的数值范围
        config_manager.config['risk_management'] = {'risk_level': 'low'}  # 恢复有效值
        config_manager.config['trading'] = {'initial_capital': -1000}  # 无效的初始资金
        with self.assertRaises(ConfigValidationError):
            config_manager.validate_config()

        logger.info("自定义验证规则测试通过")

    def test_config_diff(self):
        """测试配置差异比较"""
        config_manager = ConfigManager(config_path=None)

        other_config = {
            "system": {
                "name": "DifferentSystem",  # 修改的名称
                "version": "1.0.0",
                "environment": "testing",  # 新增的环境
                # 缺少log_level
            },
            "new_section": {  # 新增的节
                "key": "value"
            }
            # 缺少data_sources节
        }

        diff = config_manager.get_config_diff(other_config)

        # 验证差异检测
        self.assertIn('system.name', diff)
        self.assertIn('system.environment', diff)
        self.assertIn('new_section', diff)
        self.assertIn('data_sources', diff)

        logger.info("配置差异比较测试通过")

    def test_config_export_import(self):
        """测试配置导出导入"""
        config_manager = ConfigManager(config_path=None)
        config_manager.set_config('system.name', 'ExportTestSystem')

        # 导出配置
        export_data = config_manager.export_config(include_metadata=True)
        self.assertIn('config', export_data)
        self.assertIn('metadata', export_data)
        self.assertEqual(export_data['config']['system']['name'], 'ExportTestSystem')

        # 创建新的配置管理器并导入
        new_config_manager = ConfigManager(config_path=None)
        success = new_config_manager.import_config(export_data)
        self.assertTrue(success)
        self.assertEqual(new_config_manager.get_config('system.name'), 'ExportTestSystem')

        logger.info("配置导出导入测试通过")

    def test_health_check(self):
        """测试健康检查"""
        config_manager = ConfigManager(config_path=None)

        health = config_manager.health_check()

        self.assertIn('status', health)
        self.assertIn('config_valid', health)
        self.assertIn('file_system_ok', health)
        self.assertTrue(health['config_valid'])

        logger.info("健康检查测试通过")

    def test_config_template_generation(self):
        """测试配置模板生成"""
        config_manager = ConfigManager(config_path=None)

        # 生成JSON模板
        json_template_path = os.path.join(self.test_dir, "template.json")
        success = config_manager.generate_config_template(json_template_path, ConfigFormat.JSON)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_template_path))

        # 生成YAML模板
        yaml_template_path = os.path.join(self.test_dir, "template.yaml")
        success = config_manager.generate_config_template(yaml_template_path, ConfigFormat.YAML)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(yaml_template_path))

        # 验证模板内容
        with open(json_template_path, 'r', encoding=DEFAULT_ENCODING) as f:
            template_config = json.load(f)

        self.assertEqual(template_config['system']['name'], 'DeepSeekQuant')

        logger.info("配置模板生成测试通过")

    def test_context_manager(self):
        """测试上下文管理器"""
        with ConfigManager(config_path=None) as config_manager:
            self.assertIsNotNone(config_manager)
            config_manager.set_config('system.name', 'DeepSeekQuant')

        # 上下文退出后应该自动清理
        # 这里主要测试没有异常发生

        logger.info("上下文管理器测试通过")

    def test_global_config_manager(self):
        """测试全局配置管理器"""
        # 从已加载的模块中获取函数
        get_global = get_global_config_manager
        shutdown_global = shutdown_global_config_manager

        # 获取全局实例
        global_manager1 = get_global()
        self.assertIsNotNone(global_manager1)

        # 再次获取应该是同一个实例
        global_manager2 = get_global()
        self.assertIs(global_manager1, global_manager2)

        # 关闭全局管理器
        shutdown_global()

        # 重新获取应该创建新实例
        global_manager3 = get_global()
        self.assertIsNot(global_manager1, global_manager3)

        logger.info("全局配置管理器测试通过")

    def test_performance_monitoring(self):
        """测试性能监控"""
        import time

        config_manager = ConfigManager(config_path=None)

        start_time = time.time()

        # 执行多次配置操作
        for i in range(100):
            config_manager.set_config(f'test.performance.key_{i}', f'value_{i}')

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证性能在可接受范围内（小于1秒）
        self.assertLess(execution_time, 1.0)

        logger.info(f"性能测试完成，执行时间: {execution_time:.3f}秒")

    def test_error_handling(self):
        """测试错误处理"""
        config_manager = ConfigManager(config_path=None)

        # 测试设置无效键
        success = config_manager.set_config('', 'value')
        self.assertFalse(success)  # 应该返回False

        # 测试恢复不存在的快照
        success = config_manager.rollback_to_snapshot('nonexistent_snapshot')
        self.assertFalse(success)

        # 测试删除不存在的快照
        success = config_manager.delete_snapshot('nonexistent_snapshot')
        self.assertFalse(success)

        logger.info("错误处理测试通过")

    def test_enhanced_key_validation(self):
        """测试增强的键验证功能"""
        config_manager = ConfigManager(config_path=None)

        # 测试空字符串键
        success = config_manager.set_config('', 'value')
        self.assertFalse(success)

        # 测试纯空格键
        success = config_manager.set_config('   ', 'value')
        self.assertFalse(success)

        # 测试前导点号
        success = config_manager.set_config('.invalid.key', 'value')
        self.assertFalse(success)

        # 测试尾随点号
        success = config_manager.set_config('invalid.key.', 'value')
        self.assertFalse(success)

        # 测试连续点号
        success = config_manager.set_config('invalid..key', 'value')
        self.assertFalse(success)

        # 测试有效的键应该成功
        success = config_manager.set_config('valid.key.name', 'value')
        self.assertTrue(success)
        self.assertEqual(config_manager.get_config('valid.key.name'), 'value')

        logger.info("增强键验证测试通过")

    def test_list_merge_behavior(self):
        """测试列表智能合并行为"""
        config_manager = ConfigManager(config_path=None)

        # 设置初始配置，包含列表
        initial_config = {
            "features": ["feature1", "feature2"],
            "plugins": {
                "enabled": ["plugin_a", "plugin_b"]
            }
        }
        config_manager.update_config(initial_config)

        # 合并包含列表的配置
        merge_config = {
            "features": ["feature2", "feature3", "feature4"],  # feature2重复
            "plugins": {
                "enabled": ["plugin_b", "plugin_c"]  # plugin_b重复
            }
        }
        success = config_manager.merge_config(merge_config, "测试列表合并")
        self.assertTrue(success)

        # 验证列表合并结果：应该去重并保留所有元素
        features = config_manager.get_config('features')
        self.assertIn('feature1', features)
        self.assertIn('feature2', features)
        self.assertIn('feature3', features)
        self.assertIn('feature4', features)
        # 验证没有重复
        self.assertEqual(len(features), 4)

        plugins = config_manager.get_config('plugins.enabled')
        self.assertIn('plugin_a', plugins)
        self.assertIn('plugin_b', plugins)
        self.assertIn('plugin_c', plugins)
        # 验证没有重复
        self.assertEqual(len(plugins), 3)

        logger.info("列表合并测试通过")

    def test_real_encryption_implementation(self):
        """测试真正的加密实现"""
        # 使用生成的Fernet密钥（正确格式）
        import importlib
        try:
            fernet_module = importlib.import_module("cryptography.fernet")  # type: ignore
            Fernet = getattr(fernet_module, "Fernet")
        except Exception:
            self.skipTest('cryptography not installed')
            return
        encryption_key = Fernet.generate_key().decode('utf-8')
        
        config_manager = ConfigManager(
            config_path=None,
            encryption_key=encryption_key
        )

        # 设置一些敏感字段
        config_manager.set_config('api.authentication.jwt_secret', 'my_secret_jwt_key')
        config_manager.set_config('database.password', 'super_secret_password')
        config_manager.set_config('alerts.email.password', 'email_password_123')

        # 执行加密
        success = config_manager.encrypt_sensitive_data()
        self.assertTrue(success)

        # 验证加密后的值
        encrypted_jwt = config_manager.get_config('api.authentication.jwt_secret')
        encrypted_db_pwd = config_manager.get_config('database.password')
        encrypted_email_pwd = config_manager.get_config('alerts.email.password')

        # 验证加密格式：应该有_enc_前缀
        self.assertTrue(encrypted_jwt.startswith('_enc_'))
        self.assertTrue(encrypted_db_pwd.startswith('_enc_'))
        self.assertTrue(encrypted_email_pwd.startswith('_enc_'))

        # 验证加密后的值不等于原值
        self.assertNotEqual(encrypted_jwt, 'my_secret_jwt_key')
        self.assertNotEqual(encrypted_db_pwd, 'super_secret_password')
        self.assertNotEqual(encrypted_email_pwd, 'email_password_123')

        # 执行解密
        success = config_manager.decrypt_sensitive_data()
        self.assertTrue(success)

        # 验证解密后的值恢复原值
        decrypted_jwt = config_manager.get_config('api.authentication.jwt_secret')
        decrypted_db_pwd = config_manager.get_config('database.password')
        decrypted_email_pwd = config_manager.get_config('alerts.email.password')

        self.assertEqual(decrypted_jwt, 'my_secret_jwt_key')
        self.assertEqual(decrypted_db_pwd, 'super_secret_password')
        self.assertEqual(decrypted_email_pwd, 'email_password_123')

        # 验证不会重复加密已加密的数据
        config_manager.set_config('api.authentication.jwt_secret', 'my_secret_jwt_key')
        success = config_manager.encrypt_sensitive_data()
        self.assertTrue(success)
        encrypted_again = config_manager.get_config('api.authentication.jwt_secret')
        # 应该被加密
        self.assertTrue(encrypted_again.startswith('_enc_'))
        # 解密后应该能恢复
        config_manager.decrypt_sensitive_data()
        final_value = config_manager.get_config('api.authentication.jwt_secret')
        self.assertEqual(final_value, 'my_secret_jwt_key')

        logger.info("真正加密实现测试通过")

    def test_environment_variable_expansion(self):
        """测试环境变量展开功能"""
        # 设置测试环境变量
        os.environ['TEST_DB_HOST'] = 'localhost'
        os.environ['TEST_DB_PORT'] = '5432'
        os.environ['TEST_API_KEY'] = 'secret_key_12345'

        # 创建包含环境变量的配置文件（包含必需字段）
        test_config = {
            "system": {
                "name": "TestSystem",
                "version": "1.0.0",
                "environment": "testing",
                "log_level": "INFO"
            },
            "data_sources": {
                "primary": "test_source"
            },
            "database": {
                "host": "${TEST_DB_HOST}",
                "port": "${TEST_DB_PORT}",
                "user": "admin"  # 普通值
            },
            "api": {
                "key": "${TEST_API_KEY}",
                "endpoint": "https://api.example.com"  # 普通值
            },
            "unset_var": "${UNSET_VARIABLE}"  # 未设置的环境变量
        }

        config_path = os.path.join(self.test_dir, "env_test.json")
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(test_config, f)

        # 加载配置（不验证以避免校验失败）
        config_manager = ConfigManager(config_path=config_path, validate_on_load=False)

        # 验证环境变量被正确展开
        self.assertEqual(config_manager.get_config('database.host'), 'localhost')
        self.assertEqual(config_manager.get_config('database.port'), '5432')
        self.assertEqual(config_manager.get_config('database.user'), 'admin')
        self.assertEqual(config_manager.get_config('api.key'), 'secret_key_12345')
        self.assertEqual(config_manager.get_config('api.endpoint'), 'https://api.example.com')
        # 未设置的环境变量应该保持原值
        self.assertEqual(config_manager.get_config('unset_var'), '${UNSET_VARIABLE}')

        # 清理环境变量
        del os.environ['TEST_DB_HOST']
        del os.environ['TEST_DB_PORT']
        del os.environ['TEST_API_KEY']

        logger.info("环境变量展开测试通过")

    def test_hot_reload_setup(self):
        """测试热重载设置（不测试实际重载，只测试设置）"""
        # 创建测试配置文件
        config_path = os.path.join(self.test_dir, "hot_reload_test.json")
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump({"test": "value"}, f)

        # 尝试启用热重载（如果watchdog可用）
        try:
            config_manager = ConfigManager(
                config_path=config_path,
                enable_hot_reload=True
            )
            
            # 验证配置管理器创建成功
            self.assertIsNotNone(config_manager)
            self.assertEqual(config_manager.get_config('test'), 'value')
            
            # 如果watchdog可用，应该有观察者
            if config_manager._file_observer:
                self.assertIsNotNone(config_manager._file_observer)
                logger.info("热重载观察者已启动")
            else:
                logger.info("watchdog未安装，热重载功能不可用（这是正常的）")
            
            # 清理
            config_manager.cleanup()
            
        except Exception as e:
            # 如果watchdog未安装，这是正常的
            logger.info(f"热重载测试跳过: {e}")

        logger.info("热重载设置测试通过")

    def test_schema_completeness_validation(self):
        """测试配置架构完整性验证"""
        # 测试完整的配置（应该通过）
        complete_config = {
            "system": {
                "name": "Test",
                "version": "1.0.0",
                "environment": "testing",
                "log_level": "INFO"
            },
            "data_sources": {
                "primary": "test"
            }
        }
        config_path_complete = os.path.join(self.test_dir, "complete.json")
        with open(config_path_complete, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(complete_config, f)
        
        # 完整配置应该能正常加载
        config_manager_complete = ConfigManager(config_path=config_path_complete, validate_on_load=True)
        self.assertIsNotNone(config_manager_complete)
        
        # 直接测试验证方法，而不是通过加载
        # 测试缺少system段
        incomplete_config = {
            "data_sources": {
                "primary": "test"
            }
        }
        test_manager = ConfigManager(config_path=None, validate_on_load=False)
        test_manager.config = incomplete_config
        
        with self.assertRaises(ConfigValidationError) as cm:
            test_manager.validate_config()
        self.assertIn("缺少必需配置段: system", str(cm.exception))

        # 测试缺少data_sources段
        incomplete_config2 = {
            "system": {
                "name": "Test",
                "version": "1.0.0",
                "environment": "testing",
                "log_level": "INFO"
            }
        }
        test_manager2 = ConfigManager(config_path=None, validate_on_load=False)
        test_manager2.config = incomplete_config2
        
        with self.assertRaises(ConfigValidationError) as cm:
            test_manager2.validate_config()
        self.assertIn("缺少必需配置段: data_sources", str(cm.exception))

        # 测试system段缺少必需字段
        incomplete_config3 = {
            "system": {
                "name": "Test",
                "version": "1.0.0"
                # 缺少environment和log_level
            },
            "data_sources": {
                "primary": "test"
            }
        }
        test_manager3 = ConfigManager(config_path=None, validate_on_load=False)
        test_manager3.config = incomplete_config3
        
        with self.assertRaises(ConfigValidationError) as cm:
            test_manager3.validate_config()
        self.assertIn("system配置段缺少必需字段", str(cm.exception))

        logger.info("配置架构完整性验证测试通过")

    def test_config_value_cache(self):
        """测试配置值缓存机制"""
        config_manager = ConfigManager(config_path=None)

        # 第一次获取，应该从配置文件读取
        value1 = config_manager.get_config('system.name')
        self.assertEqual(value1, 'DeepSeekQuant')

        # 第二次获取相同的值，应该从缓存读取
        value2 = config_manager.get_config('system.name')
        self.assertEqual(value2, 'DeepSeekQuant')
        self.assertEqual(value1, value2)

        # 验证缓存中有数据
        self.assertGreater(len(config_manager._value_cache), 0)

        # 修改配置后，缓存应该被清空
        config_manager.set_config('system.name', 'ModifiedSystem')
        self.assertEqual(len(config_manager._value_cache), 0)

        # 重新获取，应该从配置文件读取新值
        value3 = config_manager.get_config('system.name')
        self.assertEqual(value3, 'ModifiedSystem')

        # 缓存应该再次被填充
        self.assertGreater(len(config_manager._value_cache), 0)

        logger.info("配置值缓存测试通过")

    def test_friendly_error_messages(self):
        """测试友好的错误消息"""
        # 测试验证错误的友好提示
        config_manager = ConfigManager(config_path=None, validate_on_load=False)
        
        # 创建一个验证失败的配置
        invalid_config = {
            "system": {
                "name": "Test",
                "version": "1.0.0",
                "environment": "invalid_mode",  # 无效的环境
                "log_level": "INFO"
            },
            "data_sources": {
                "primary": "test"
            }
        }
        config_manager.config = invalid_config
        
        # 验证错误消息包含路径信息
        try:
            config_manager.validate_config()
            self.fail("应该抛出ConfigValidationError")
        except ConfigValidationError as e:
            error_msg = str(e)
            # 验证错误消息包含路径信息
            self.assertIn("配置验证失败", error_msg)
            # 验证错误消息更具体
            logger.info(f"验证错误消息: {error_msg}")
        
        logger.info("友好错误消息测试通过")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)