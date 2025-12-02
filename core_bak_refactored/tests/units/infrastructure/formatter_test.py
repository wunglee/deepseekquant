import pytest
from datetime import datetime, date
from core_bak_refactored.infrastructure.formatter import DataFormatter


class TestDataFormatter:
    """测试数据格式化器。"""

    def test_init(self):
        """测试初始化。"""
        formatter = DataFormatter(decimal_places=3)
        assert formatter.decimal_places == 3

    def test_format_number(self):
        """测试数值格式化。"""
        formatter = DataFormatter(decimal_places=2)
        
        assert formatter.format_number(123.456) == '123.46'
        assert formatter.format_number(100) == '100.00'
        assert formatter.format_number(None) == 'N/A'

    def test_format_number_custom_places(self):
        """测试自定义小数位数。"""
        formatter = DataFormatter(decimal_places=2)
        
        assert formatter.format_number(123.456, decimal_places=1) == '123.5'

    def test_format_percentage(self):
        """测试百分比格式化。"""
        formatter = DataFormatter(decimal_places=2)
        
        assert formatter.format_percentage(0.1234) == '12.34%'
        assert formatter.format_percentage(1.5) == '150.00%'
        assert formatter.format_percentage(None) == 'N/A'

    def test_format_currency(self):
        """测试货币格式化。"""
        formatter = DataFormatter(decimal_places=2)
        
        assert formatter.format_currency(1234.56) == '$1,234.56'
        assert formatter.format_currency(1000000) == '$1,000,000.00'
        assert formatter.format_currency(None) == 'N/A'

    def test_format_currency_custom_symbol(self):
        """测试自定义货币符号。"""
        formatter = DataFormatter(decimal_places=2)
        
        assert formatter.format_currency(100, currency_symbol='¥') == '¥100.00'

    def test_format_date(self):
        """测试日期格式化。"""
        formatter = DataFormatter()
        
        dt = datetime(2024, 1, 1, 12, 30, 45)
        result = formatter.format_date(dt)
        assert '2024-01-01' in result
        assert '12:30:45' in result

    def test_format_date_custom_format(self):
        """测试自定义日期格式。"""
        formatter = DataFormatter()
        
        dt = datetime(2024, 1, 1)
        result = formatter.format_date(dt, format_str='%Y/%m/%d')
        assert result == '2024/01/01'

    def test_format_large_number_thousand(self):
        """测试千位格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_large_number(1500) == '1.5K'
        assert formatter.format_large_number(5000) == '5.0K'

    def test_format_large_number_million(self):
        """测试百万位格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_large_number(2_500_000) == '2.5M'

    def test_format_large_number_billion(self):
        """测试十亿位格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_large_number(3_000_000_000) == '3.0B'

    def test_format_large_number_trillion(self):
        """测试万亿位格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_large_number(1_500_000_000_000) == '1.5T'

    def test_format_duration_seconds(self):
        """测试秒级时长格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_duration(45) == '45s'

    def test_format_duration_minutes(self):
        """测试分钟级时长格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_duration(90) == '1m 30s'

    def test_format_duration_hours(self):
        """测试小时级时长格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_duration(3665) == '1h 1m 5s'

    def test_format_duration_days(self):
        """测试天级时长格式化。"""
        formatter = DataFormatter()
        
        result = formatter.format_duration(90000)
        assert 'd' in result

    def test_format_boolean(self):
        """测试布尔值格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_boolean(True) == 'Yes'
        assert formatter.format_boolean(False) == 'No'
        assert formatter.format_boolean(None) == 'N/A'

    def test_format_boolean_custom_text(self):
        """测试自定义布尔值文本。"""
        formatter = DataFormatter()
        
        assert formatter.format_boolean(True, '是', '否') == '是'
        assert formatter.format_boolean(False, '是', '否') == '否'

    def test_truncate_string(self):
        """测试字符串截断。"""
        formatter = DataFormatter()
        
        long_text = 'This is a very long string that needs to be truncated'
        result = formatter.truncate_string(long_text, 20)
        
        assert len(result) == 20
        assert result.endswith('...')

    def test_truncate_string_short(self):
        """测试短字符串不截断。"""
        formatter = DataFormatter()
        
        short_text = 'Short'
        result = formatter.truncate_string(short_text, 20)
        
        assert result == short_text

    def test_format_list(self):
        """测试列表格式化。"""
        formatter = DataFormatter()
        
        items = ['AAPL', 'GOOGL', 'MSFT']
        result = formatter.format_list(items)
        
        assert result == 'AAPL, GOOGL, MSFT'

    def test_format_list_max_items(self):
        """测试限制列表项数。"""
        formatter = DataFormatter()
        
        items = ['A', 'B', 'C', 'D', 'E']
        result = formatter.format_list(items, max_items=3)
        
        assert 'A, B, C' in result
        assert '+2 more' in result

    def test_format_list_empty(self):
        """测试空列表格式化。"""
        formatter = DataFormatter()
        
        assert formatter.format_list([]) == 'N/A'
