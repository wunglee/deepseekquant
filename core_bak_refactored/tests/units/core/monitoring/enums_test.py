"""
监控模块枚举测试
"""

import unittest
from core_bak_refactored.core.monitoring.enums import AlertSeverity, AlertChannel


class TestAlertSeverity(unittest.TestCase):
    """测试告警严重程度枚举"""
    
    def test_severity_levels_exist(self):
        """测试所有严重等级都存在"""
        self.assertEqual(AlertSeverity.INFO.value, 'info')
        self.assertEqual(AlertSeverity.WARNING.value, 'warning')
        self.assertEqual(AlertSeverity.ERROR.value, 'error')
        self.assertEqual(AlertSeverity.CRITICAL.value, 'critical')
    
    def test_str_conversion(self):
        """测试字符串转换"""
        self.assertEqual(str(AlertSeverity.ERROR), 'error')
    
    def test_is_str_enum(self):
        """测试继承自str"""
        self.assertIsInstance(AlertSeverity.WARNING, str)


class TestAlertChannel(unittest.TestCase):
    """测试告警通道枚举"""
    
    def test_channels_exist(self):
        """测试所有通道都存在"""
        self.assertEqual(AlertChannel.EMAIL.value, 'email')
        self.assertEqual(AlertChannel.LOG.value, 'log')
        self.assertEqual(AlertChannel.WECHAT.value, 'wechat')
    
    def test_str_conversion(self):
        """测试字符串转换"""
        self.assertEqual(str(AlertChannel.EMAIL), 'email')


if __name__ == '__main__':
    unittest.main()
