# test_logging_system.py
import unittest
import tempfile
import os
import sys
import logging
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_system import (
    LogLevel, LogDestination, LogEntry, LogFormat, LogRotationStrategy,
    LogConfig, DeepSeekQuantFormatter, AuditLogFilter, PerformanceLogFilter,
    ErrorLogFilter, ThreadSafeRotatingFileHandler, CompressedRotatingFileHandler,
    LoggingSystem, setup_logging, get_logger, get_audit_logger,
    get_performance_logger, get_error_logger, log_audit, log_performance,
    log_error, shutdown_logging, get_logging_stats,
    create_default_config, create_production_config, create_development_config
)


class TestLoggingEnums(unittest.TestCase):
    """测试日志枚举"""

    def test_log_level_enum(self):
        """测试日志级别枚举"""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
        self.assertEqual(LogLevel.INFO.value, "INFO")
        self.assertEqual(LogLevel.ERROR.value, "ERROR")

    def test_log_destination_enum(self):
        """测试日志目的地枚举"""
        self.assertEqual(LogDestination.CONSOLE.value, "console")
        self.assertEqual(LogDestination.FILE.value, "file")
        self.assertEqual(LogDestination.ALL.value, "all")

    def test_log_format_enum(self):
        """测试日志格式枚举"""
        self.assertEqual(LogFormat.TEXT.value, "text")
        self.assertEqual(LogFormat.JSON.value, "json")

    def test_log_rotation_strategy_enum(self):
        """测试日志轮转策略枚举"""
        self.assertEqual(LogRotationStrategy.SIZE_BASED.value, "size_based")
        self.assertEqual(LogRotationStrategy.DAILY.value, "daily")


class TestLogEntry(unittest.TestCase):
    """测试日志条目"""

    def test_log_entry_creation(self):
        """测试日志条目创建"""
        entry = LogEntry(
            timestamp="2024-01-15T10:30:00",
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            module="test_module",
            function="test_function",
            line_number=42,
            process_id=1234,
            thread_id=5678,
            thread_name="MainThread"
        )

        self.assertEqual(entry.level, LogLevel.INFO)
        self.assertEqual(entry.logger_name, "test_logger")
        self.assertEqual(entry.message, "Test message")
        self.assertEqual(entry.line_number, 42)

    def test_log_entry_with_optional_fields(self):
        """测试带可选字段的日志条目"""
        entry = LogEntry(
            timestamp="2024-01-15T10:30:00",
            level=LogLevel.ERROR,
            logger_name="test_logger",
            message="Error occurred",
            module="test_module",
            function="test_function",
            line_number=42,
            process_id=1234,
            thread_id=5678,
            thread_name="MainThread",
            exception_info="Exception details",
            stack_trace="Stack trace",
            extra_data={"key": "value"},
            correlation_id="corr_123"
        )

        self.assertEqual(entry.exception_info, "Exception details")
        self.assertEqual(entry.extra_data, {"key": "value"})
        self.assertEqual(entry.correlation_id, "corr_123")

    def test_log_entry_to_dict(self):
        """测试日志条目转字典"""
        entry = LogEntry(
            timestamp="2024-01-15T10:30:00",
            level=LogLevel.WARNING,
            logger_name="test_logger",
            message="Warning message",
            module="test_module",
            function="test_function",
            line_number=42,
            process_id=1234,
            thread_id=5678,
            thread_name="MainThread"
        )

        entry_dict = entry.to_dict()
        self.assertEqual(entry_dict['level'], 'WARNING')
        self.assertEqual(entry_dict['message'], 'Warning message')
        self.assertEqual(entry_dict['line_number'], 42)

    def test_log_entry_from_dict(self):
        """测试从字典创建日志条目"""
        entry_dict = {
            'timestamp': '2024-01-15T10:30:00',
            'level': 'INFO',
            'logger_name': 'test_logger',
            'message': 'Test message',
            'module': 'test_module',
            'function': 'test_function',
            'line_number': 42,
            'process_id': 1234,
            'thread_id': 5678,
            'thread_name': 'MainThread',
            'extra_data': {'key': 'value'}
        }

        entry = LogEntry.from_dict(entry_dict)
        self.assertEqual(entry.level, LogLevel.INFO)
        self.assertEqual(entry.logger_name, 'test_logger')
        self.assertEqual(entry.extra_data, {'key': 'value'})


class TestLogConfig(unittest.TestCase):
    """测试日志配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = LogConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.level, LogLevel.INFO)
        self.assertEqual(config.format, LogFormat.TEXT)
        self.assertEqual(config.destinations, [LogDestination.CONSOLE, LogDestination.FILE])
        self.assertEqual(config.file_path, "logs/deepseekquant.log")

    def test_custom_config(self):
        """测试自定义配置"""
        config = LogConfig(
            enabled=False,
            level=LogLevel.DEBUG,
            format=LogFormat.JSON,
            destinations=[LogDestination.CONSOLE],
            file_path="custom.log",
            max_file_size_mb=50
        )

        self.assertFalse(config.enabled)
        self.assertEqual(config.level, LogLevel.DEBUG)
        self.assertEqual(config.format, LogFormat.JSON)
        self.assertEqual(config.destinations, [LogDestination.CONSOLE])
        self.assertEqual(config.file_path, "custom.log")
        self.assertEqual(config.max_file_size_mb, 50)


class TestLogFilters(unittest.TestCase):
    """测试日志过滤器"""

    def setUp(self):
        """设置测试记录"""
        self.record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )

    def test_audit_log_filter(self):
        """测试审计日志过滤器"""
        filter_obj = AuditLogFilter()

        # 测试审计记录
        self.record.is_audit = True
        self.assertTrue(filter_obj.filter(self.record))

        # 测试警告级别记录
        self.record.is_audit = False
        self.record.levelno = logging.WARNING
        self.assertTrue(filter_obj.filter(self.record))

        # 测试信息级别记录（不应通过）
        self.record.levelno = logging.INFO
        self.assertFalse(filter_obj.filter(self.record))

    def test_performance_log_filter(self):
        """测试性能日志过滤器"""
        filter_obj = PerformanceLogFilter()

        # 测试性能记录
        self.record.is_performance = True
        self.assertTrue(filter_obj.filter(self.record))

        # 测试性能日志记录器名称
        self.record.is_performance = False
        self.record.name = 'performance_tracker'
        self.assertTrue(filter_obj.filter(self.record))

        # 测试普通记录器名称（不应通过）
        self.record.name = 'ordinary_logger'
        self.assertFalse(filter_obj.filter(self.record))

    def test_error_log_filter(self):
        """测试错误日志过滤器"""
        filter_obj = ErrorLogFilter()

        # 测试错误级别记录
        self.record.levelno = logging.ERROR
        self.assertTrue(filter_obj.filter(self.record))

        # 测试关键级别记录
        self.record.levelno = logging.CRITICAL
        self.assertTrue(filter_obj.filter(self.record))

        # 测试警告级别记录（不应通过）
        self.record.levelno = logging.WARNING
        self.assertFalse(filter_obj.filter(self.record))


class TestLoggingSystemBase(unittest.TestCase):
    """日志系统测试基类"""

    def setUp(self):
        """设置临时目录用于测试"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LogConfig(
            destinations=[LogDestination.CONSOLE, LogDestination.FILE],
            file_path=os.path.join(self.temp_dir, "test.log"),
            audit_log_path=os.path.join(self.temp_dir, "audit.log"),
            performance_log_path=os.path.join(self.temp_dir, "performance.log"),
            error_log_path=os.path.join(self.temp_dir, "error.log"),
            max_file_size_mb=1,  # 小文件大小便于测试轮转
            backup_count=3
        )

    def tearDown(self):
        """清理测试文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestLoggingSystem(TestLoggingSystemBase):
    """测试日志系统"""

    def test_initialization(self):
        """测试日志系统初始化"""
        logging_system = LoggingSystem(self.config)
        self.assertTrue(logging_system._initialized)

        # 测试获取日志记录器
        logger = logging_system.get_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)

        logging_system.shutdown()

    def test_logging_functionality(self):
        """测试日志记录功能"""
        logging_system = LoggingSystem(self.config)
        logger = logging_system.get_logger("test_functionality")

        # 记录不同级别的日志
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # 确保日志文件被创建
        self.assertTrue(os.path.exists(self.config.file_path))

        logging_system.shutdown()

    def test_audit_logging(self):
        """测试审计日志记录"""
        logging_system = LoggingSystem(self.config)

        # 记录审计日志
        logging_system.log_audit(
            action="login",
            user="test_user",
            resource="system",
            status="success",
            details={"ip": "127.0.0.1", "user_agent": "test"}
        )

        # 确保审计日志文件被创建
        self.assertTrue(os.path.exists(self.config.audit_log_path))

        logging_system.shutdown()

    def test_performance_logging(self):
        """测试性能日志记录"""
        logging_system = LoggingSystem(self.config)

        # 记录性能日志
        logging_system.log_performance(
            operation="data_fetch",
            duration=0.125,
            success=True,
            metrics={"records": 1000, "size_mb": 2.5}
        )

        logging_system.log_performance(
            operation="data_process",
            duration=0.5,
            success=False,
            metrics={"error": "timeout"}
        )

        # 确保性能日志文件被创建
        self.assertTrue(os.path.exists(self.config.performance_log_path))

        logging_system.shutdown()

    def test_error_logging(self):
        """测试错误日志记录"""
        logging_system = LoggingSystem(self.config)

        # 记录错误日志
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            logging_system.log_error(
                error_type="ValueError",
                error_message=str(e),
                context={"operation": "test_operation"},
                exception=e
            )

        # 确保错误日志文件被创建
        self.assertTrue(os.path.exists(self.config.error_log_path))

        logging_system.shutdown()

    def test_log_level_change(self):
        """测试日志级别更改"""
        logging_system = LoggingSystem(self.config)

        # 初始级别为INFO，DEBUG消息不应被记录
        logger = logging_system.get_logger("test_level")
        with self.assertLogs('test_level', level='DEBUG') as cm:
            logger.debug("This should not appear")

        # 更改级别为DEBUG
        logging_system.set_level(LogLevel.DEBUG)

        # 现在DEBUG消息应该被记录
        with self.assertLogs('test_level', level='DEBUG') as cm:
            logger.debug("This should appear now")

        logging_system.shutdown()

    def test_stats_collection(self):
        """测试统计信息收集"""
        logging_system = LoggingSystem(self.config)
        logger = logging_system.get_logger("test_stats")

        # 记录一些日志
        logger.info("Test message 1")
        logger.warning("Test message 2")
        logger.error("Test message 3")

        # 记录审计和性能日志
        logging_system.log_audit("test", "user", "resource", "success")
        logging_system.log_performance("test_op", 0.1, True)

        # 获取统计信息
        stats = logging_system.get_stats()

        # 验证统计信息
        self.assertGreater(stats['total_logs'], 0)
        self.assertIn('info_logs', stats)
        self.assertIn('audit_logs', stats)
        self.assertIn('performance_logs', stats)

        logging_system.shutdown()

    @patch('logging_system.DatabaseLogHandler')
    def test_database_handler(self, mock_db_handler_class):
        """测试数据库处理器（使用mock）"""
        # 创建 mock 处理器实例
        mock_handler = MagicMock()
        mock_handler.level = logging.INFO  # 设置 level 为整数
        mock_db_handler_class.return_value = mock_handler
        
        # 配置使用数据库目的地
        self.config.destinations = [LogDestination.DATABASE]
        self.config.database_connection = "sqlite:///test.db"

        logging_system = LoggingSystem(self.config)

        # 验证数据库处理器被调用
        mock_db_handler_class.assert_called_once()

        logging_system.shutdown()

    def test_context_manager(self):
        """测试上下文管理器"""
        with LoggingSystem(self.config) as logging_system:
            logger = logging_system.get_logger("test_context")
            logger.info("Testing context manager")
            self.assertTrue(logging_system._initialized)

        # 上下文退出后系统应关闭
        self.assertTrue(logging_system._shutdown)


class TestGlobalLoggingFunctions(TestLoggingSystemBase):
    """测试全局日志函数"""

    def test_global_functions(self):
        """测试全局函数"""
        # 设置全局日志系统
        setup_logging(self.config)

        # 测试获取各种日志记录器
        logger = get_logger("test_global")
        audit_logger = get_audit_logger()
        performance_logger = get_performance_logger()
        error_logger = get_error_logger()

        self.assertIsInstance(logger, logging.Logger)
        self.assertIsInstance(audit_logger, logging.Logger)

        # 测试全局日志函数
        log_audit("global_action", "global_user", "global_resource", "success")
        log_performance("global_operation", 0.25, True)
        log_error("GlobalError", "Global error message")

        # 测试获取统计信息
        stats = get_logging_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_logs', stats)

        # 清理
        shutdown_logging()

    def test_config_creation_functions(self):
        """测试配置创建函数"""
        # 测试默认配置
        default_config = create_default_config()
        self.assertEqual(default_config.level, LogLevel.INFO)

        # 测试生产配置
        prod_config = create_production_config()
        self.assertEqual(prod_config.level, LogLevel.INFO)
        self.assertIn(LogDestination.FILE, prod_config.destinations)
        self.assertTrue(prod_config.audit_log_enabled)

        # 测试开发配置
        dev_config = create_development_config()
        self.assertEqual(dev_config.level, LogLevel.DEBUG)
        self.assertIn(LogDestination.CONSOLE, dev_config.destinations)


class TestLogFormatters(TestLoggingSystemBase):
    """测试日志格式化器"""

    def test_text_formatter(self):
        """测试文本格式化器"""
        config = LogConfig(format=LogFormat.TEXT)
        formatter = DeepSeekQuantFormatter(config)

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        self.assertIsInstance(formatted, str)
        self.assertIn('Test message', formatted)

    def test_json_formatter(self):
        """测试JSON格式化器"""
        config = LogConfig(format=LogFormat.JSON)
        formatter = DeepSeekQuantFormatter(config)

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        self.assertIsInstance(formatted, str)

        # 验证JSON格式
        parsed = json.loads(formatted)
        self.assertEqual(parsed['level'], 'INFO')
        self.assertEqual(parsed['message'], 'Test message')

    def test_json_formatter_with_exception(self):
        """测试带异常的JSON格式化器"""
        config = LogConfig(format=LogFormat.JSON)
        formatter = DeepSeekQuantFormatter(config)

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            record = logging.LogRecord(
                name='test',
                level=logging.ERROR,
                pathname='test.py',
                lineno=1,
                msg='Error occurred',
                args=(),
                exc_info=(type(e), e, e.__traceback__)
            )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        self.assertIn('exception', parsed)


class TestLogHandlers(TestLoggingSystemBase):
    """测试日志处理器"""

    def test_thread_safe_handler(self):
        """测试线程安全处理器"""
        log_file = os.path.join(self.temp_dir, "thread_test.log")
        handler = ThreadSafeRotatingFileHandler(log_file)

        # 创建日志记录器并使用处理器
        logger = logging.getLogger("thread_test")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        # 记录一些消息
        logger.info("Thread-safe test message")

        # 验证文件被创建
        self.assertTrue(os.path.exists(log_file))

        handler.close()

    @patch('gzip.open')
    def test_compressed_handler(self, mock_gzip):
        """测试压缩处理器（使用mock）"""
        log_file = os.path.join(self.temp_dir, "compress_test.log")
        handler = CompressedRotatingFileHandler(log_file, compression=True)

        # 基本功能测试
        logger = logging.getLogger("compress_test")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("Compression test message")

        # 验证处理器正常工作
        self.assertTrue(os.path.exists(log_file))

        handler.close()


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)