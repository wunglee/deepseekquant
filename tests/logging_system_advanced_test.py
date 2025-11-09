"""
日志系统高级功能测试
包含：异步处理器、缓冲处理器、查询引擎、告警管理器测试
"""

import unittest
import tempfile
import os
import time
import json
from datetime import datetime, timedelta
import shutil

from infrastructure.logging_service import (
    LoggingSystem, LogConfig, LogLevel, LogDestination, LogFormat,
    AsyncLogHandler, BufferedLogHandler, LogQueryEngine,
    AlertManager, AlertRule, LogEntry,
    create_docker_config, create_kubernetes_config
)
import logging


class TestAsyncLogHandler(unittest.TestCase):
    """测试异步日志处理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'async_test.log')
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_async_handler_creation(self):
        """测试异步处理器创建"""
        base_handler = logging.FileHandler(self.log_file)
        async_handler = AsyncLogHandler(base_handler, max_queue_size=100)
        
        self.assertIsNotNone(async_handler)
        self.assertEqual(async_handler.max_queue_size, 100)
        self.assertIsNotNone(async_handler.worker_thread)
        
        async_handler.close()
    
    def test_async_handler_logging(self):
        """测试异步处理器记录日志"""
        base_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(message)s')
        base_handler.setFormatter(formatter)
        
        async_handler = AsyncLogHandler(base_handler, max_queue_size=100)
        
        # 创建测试记录
        for i in range(10):
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=f'Test message {i}',
                args=(),
                exc_info=None
            )
            async_handler.emit(record)
        
        # 刷新并关闭
        async_handler.flush()
        async_handler.close()
        
        # 验证日志已写入
        time.sleep(0.5)  # 等待异步写入完成
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)


class TestBufferedLogHandler(unittest.TestCase):
    """测试缓冲日志处理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'buffered_test.log')
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_buffered_handler_creation(self):
        """测试缓冲处理器创建"""
        base_handler = logging.FileHandler(self.log_file)
        buffered_handler = BufferedLogHandler(
            base_handler,
            buffer_size=10,
            flush_interval=1.0
        )
        
        self.assertIsNotNone(buffered_handler)
        self.assertEqual(buffered_handler.buffer_size, 10)
        self.assertEqual(buffered_handler.flush_interval, 1.0)
        
        buffered_handler.close()
    
    def test_buffered_handler_auto_flush(self):
        """测试缓冲处理器自动刷新"""
        base_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(message)s')
        base_handler.setFormatter(formatter)
        
        buffered_handler = BufferedLogHandler(
            base_handler,
            buffer_size=5,  # 小缓冲区
            flush_interval=10.0  # 长间隔
        )
        
        # 写入超过缓冲区大小的日志
        for i in range(10):
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=f'Test message {i}',
                args=(),
                exc_info=None
            )
            buffered_handler.emit(record)
        
        buffered_handler.close()
        
        # 验证日志已写入
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 10)


class TestLogQueryEngine(unittest.TestCase):
    """测试日志查询引擎"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        os.makedirs(self.log_dir)
        
        # 创建测试日志文件
        self._create_test_logs()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_logs(self):
        """创建测试日志文件"""
        log_file = os.path.join(self.log_dir, 'test.log')
        
        with open(log_file, 'w') as f:
            for i in range(10):
                log_entry = {
                    'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                    'level': 'INFO' if i % 2 == 0 else 'ERROR',
                    'logger_name': 'test',  # 修正为logger_name
                    'message': f'Test message {i}',
                    'module': 'test',
                    'function': 'test_func',
                    'line_number': 10,
                    'process_id': 1234,
                    'thread_id': 5678,
                    'thread_name': 'MainThread'
                }
                f.write(json.dumps(log_entry) + '\n')
    
    def test_query_engine_creation(self):
        """测试查询引擎创建"""
        engine = LogQueryEngine(self.log_dir, LogFormat.JSON)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.log_directory, self.log_dir)
    
    def test_query_engine_build_index(self):
        """测试构建索引"""
        engine = LogQueryEngine(self.log_dir, LogFormat.JSON)
        engine.build_index()
        
        # 验证索引已构建
        self.assertGreater(len(engine.index), 0)
    
    def test_query_engine_search(self):
        """测试日志搜索"""
        engine = LogQueryEngine(self.log_dir, LogFormat.JSON)
        results = engine.search('Test message', max_results=100)
        
        # 应该找到日志
        self.assertGreater(len(results), 0)
    
    def test_query_engine_search_by_level(self):
        """测试按级别搜索"""
        engine = LogQueryEngine(self.log_dir, LogFormat.JSON)
        results = engine.search('', level=LogLevel.ERROR, max_results=100)
        
        # 应该只返回ERROR级别的日志
        for entry in results:
            self.assertEqual(entry.level, LogLevel.ERROR)


class TestAlertManager(unittest.TestCase):
    """测试告警管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        config = LogConfig(
            destinations=[LogDestination.CONSOLE],
            file_path=os.path.join(self.temp_dir, 'test.log')
        )
        self.logging_system = LoggingSystem(config)
    
    def tearDown(self):
        """清理测试环境"""
        self.logging_system.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_alert_manager_creation(self):
        """测试告警管理器创建"""
        alert_manager = AlertManager(self.logging_system)
        self.assertIsNotNone(alert_manager)
        self.assertEqual(len(alert_manager.rules), 0)
    
    def test_add_alert_rule(self):
        """测试添加告警规则"""
        alert_manager = AlertManager(self.logging_system)
        
        def condition(stats):
            return stats.get('total_logs', 0) > 5
        
        def action(name, stats):
            pass
        
        rule = AlertRule(
            name='test_rule',
            condition=condition,
            action=action
        )
        
        alert_manager.add_rule(rule)
        self.assertEqual(len(alert_manager.rules), 1)
        self.assertIn('test_rule', alert_manager.rules)
    
    def test_error_rate_alert(self):
        """测试错误率告警"""
        alert_manager = AlertManager(self.logging_system)
        rule = alert_manager.create_error_rate_alert(threshold=0.5)
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, 'high_error_rate')
    
    def test_disk_space_alert(self):
        """测试磁盘空间告警"""
        alert_manager = AlertManager(self.logging_system)
        rule = alert_manager.create_disk_space_alert(threshold_mb=100)
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, 'large_log_files')


class TestDockerKubernetesConfig(unittest.TestCase):
    """测试Docker和Kubernetes配置"""
    
    def test_docker_config(self):
        """测试Docker配置"""
        config = create_docker_config()
        
        self.assertEqual(config.level, LogLevel.INFO)
        self.assertEqual(config.format, LogFormat.JSON)
        self.assertIn(LogDestination.CONSOLE, config.destinations)
        self.assertFalse(config.audit_log_enabled)
    
    def test_kubernetes_config(self):
        """测试Kubernetes配置"""
        config = create_kubernetes_config()
        
        self.assertEqual(config.level, LogLevel.DEBUG)
        self.assertEqual(config.format, LogFormat.JSON)
        self.assertTrue(config.include_function)
        self.assertTrue(config.include_line_number)


class TestAdvancedIntegration(unittest.TestCase):
    """测试高级功能集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logging_system_with_query_engine(self):
        """测试日志系统集成查询引擎"""
        config = LogConfig(
            destinations=[LogDestination.FILE],
            file_path=os.path.join(self.temp_dir, 'test.log'),
            format=LogFormat.JSON
        )
        
        logging_system = LoggingSystem(config)
        
        # 验证查询引擎已初始化
        self.assertIsNotNone(logging_system.query_engine)
        
        # 记录一些日志
        logger = logging_system.get_logger('test')
        logger.info('Test message 1')
        logger.error('Test error message')
        
        logging_system.flush()
        time.sleep(0.1)
        
        # 尝试搜索（可能为空，因为日志刚写入）
        results = logging_system.advanced_search('Test')
        self.assertIsInstance(results, list)
        
        logging_system.shutdown()
    
    def test_logging_system_with_alert_manager(self):
        """测试日志系统集成告警管理器"""
        config = LogConfig(
            destinations=[LogDestination.CONSOLE],
            file_path=os.path.join(self.temp_dir, 'test.log')
        )
        
        logging_system = LoggingSystem(config)
        
        # 验证告警管理器已初始化
        self.assertIsNotNone(logging_system.alert_manager)
        
        # 设置默认告警
        logging_system.setup_default_alerts()
        
        # 验证告警规则已添加
        self.assertGreater(len(logging_system.alert_manager.rules), 0)
        
        # 检查告警
        logging_system.check_alerts()
        
        logging_system.shutdown()


if __name__ == '__main__':
    unittest.main()
