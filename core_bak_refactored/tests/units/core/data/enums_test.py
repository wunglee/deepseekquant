"""
数据模块枚举测试
"""

import unittest
from core_bak_refactored.core.data.enums import (
    DataSourceType, DataFrequency,
    DataType, DataInterval, DataPeriod, DataFormat,
    DataQualityIssueType
)


class TestDataSourceType(unittest.TestCase):
    """测试数据源类型枚举"""
    
    def test_data_source_types_exist(self):
        """测试所有数据源类型都存在"""
        self.assertEqual(DataSourceType.YAHOO_FINANCE.value, "yahoo")
        self.assertEqual(DataSourceType.ALPHA_VANTAGE.value, "alpha_vantage")
        self.assertEqual(DataSourceType.DATABASE.value, "database")
    
    def test_enum_values_unique(self):
        """测试枚举值唯一性"""
        values = [e.value for e in DataSourceType]
        self.assertEqual(len(values), len(set(values)))


class TestDataFrequency(unittest.TestCase):
    """测试数据频率枚举"""
    
    def test_frequency_types_exist(self):
        """测试所有频率类型都存在"""
        self.assertEqual(DataFrequency.DAILY.value, "daily")
        self.assertEqual(DataFrequency.MINUTE.value, "minute")
        self.assertEqual(DataFrequency.HOUR.value, "hour")


class TestDataType(unittest.TestCase):
    """测试数据类型枚举"""
    
    def test_data_types_exist(self):
        """测试数据类型存在"""
        self.assertEqual(DataType.OHLCV.value, 'ohlcv')
        self.assertEqual(DataType.DIVIDENDS.value, 'dividends')
    
    def test_str_conversion(self):
        """测试字符串转换"""
        self.assertEqual(str(DataType.OHLCV), 'ohlcv')


class TestDataInterval(unittest.TestCase):
    """测试数据间隔枚举"""
    
    def test_intervals_exist(self):
        """测试间隔值存在"""
        self.assertEqual(DataInterval.MINUTE_1.value, '1m')
        self.assertEqual(DataInterval.DAY_1.value, '1d')


class TestDataPeriod(unittest.TestCase):
    """测试数据期间枚举"""
    
    def test_periods_exist(self):
        """测试期间值存在"""
        self.assertEqual(DataPeriod.DAY_1.value, '1d')
        self.assertEqual(DataPeriod.YEAR_1.value, '1y')


class TestDataFormat(unittest.TestCase):
    """测试数据格式枚举"""
    
    def test_formats_exist(self):
        """测试格式值存在"""
        self.assertEqual(DataFormat.JSON.value, 'json')
        self.assertEqual(DataFormat.CSV.value, 'csv')


class TestDataQualityIssueType(unittest.TestCase):
    """测试数据质量问题类型枚举"""
    
    def test_issue_types_exist(self):
        """测试所有问题类型都存在"""
        self.assertEqual(DataQualityIssueType.MISSING_DATA.value, 'missing_data')
        self.assertEqual(DataQualityIssueType.OUTLIER.value, 'outlier')


if __name__ == '__main__':
    unittest.main()
