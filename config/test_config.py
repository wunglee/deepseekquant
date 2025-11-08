# test_config.py
"""
测试配置文件
"""

# 测试用日志配置
TEST_LOG_CONFIG = {
    'enabled': True,
    'level': 'DEBUG',
    'format': 'text',
    'destinations': ['console', 'file'],
    'file_path': 'test.log',
    'max_file_size_mb': 1,
    'backup_count': 3,
    'audit_log_enabled': True,
    'performance_log_enabled': True,
    'error_log_enabled': True
}

# 测试用交易信号数据
TEST_SIGNAL_DATA = {
    'id': 'test_signal_001',
    'symbol': 'AAPL',
    'signal_type': 'buy',
    'price': 150.0,
    'timestamp': '2024-01-15T10:30:00',
    'quantity': 100,
    'stop_loss': 145.0,
    'take_profit': 160.0,
    'metadata': {
        'source': 'technical',
        'confidence': 0.9,
        'strength': 'strong'
    }
}