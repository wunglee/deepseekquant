"""
测试公共基类与工具
"""

import unittest
from unittest.mock import MagicMock


class DeepSeekQuantTestBase(unittest.TestCase):
    """统一的测试基类"""

    @classmethod
    def setUpClass(cls):
        # 可扩展的全局测试设置（保留为空以避免影响现有逻辑）
        pass

    def create_mock_component(self, component_class, **kwargs):
        """创建统一的Mock组件"""
        mock_component = MagicMock(spec=component_class)
        mock_component.get_status.return_value = {"status": "ready"}
        mock_component.initialize.return_value = True
        for key, value in kwargs.items():
            setattr(mock_component, key, value)
        return mock_component
