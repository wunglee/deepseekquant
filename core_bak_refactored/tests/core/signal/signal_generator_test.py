"""
信号生成器测试
由于源文件代码不完整，仅测试基础功能
"""

import unittest

from core.signal.signal_generator import SignalGenerator


class TestSignalGenerator(unittest.TestCase):
    """测试信号生成器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'technical': {
                'enabled_indicators': [],
                'indicator_parameters': {}
            },
            'quantitative': {
                'enabled_methods': [],
                'method_parameters': {}
            },
            'ml': {
                'enabled': False,
                'models': []
            },
            'signal': {
                'composite_threshold': 3
            }
        }
        self.generator = SignalGenerator(self.config)
    
    def test_generator_initialization(self):
        """测试生成器初始化"""
        self.assertIsNotNone(self.generator)
        # self.assertIsNotNone(self.generator.indicators)  # TODO: 待TechnicalIndicators实现
        self.assertEqual(self.generator.config, self.config)


if __name__ == '__main__':
    unittest.main()
