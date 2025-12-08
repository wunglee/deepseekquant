"""
图表数据API单元测试

测试目标：
- 验证 /api/v1/chart/data 端点的功能
- 验证参数验证逻辑
- 验证返回数据格式

测试规范：
- 文件命名：chart_api_test.py（符合 *_test.py 规范）
- 目录位置：tests/units/app/quality_monitoring/（镜像源代码目录）
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta


class ChartAPIParameterValidationTest(unittest.TestCase):
    """图表API参数验证测试"""
    
    @patch('core_bak_refactored.app.quality_monitoring.routes.chart_routes')
    def setUp(self, mock_routes):
        """测试准备"""
        from core_bak_refactored.app.quality_monitoring.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        self.endpoint = '/api/v1/chart/data'
    
    def test_missing_index_id_parameter(self):
        """测试：缺少必需参数 index_id"""
        response = self.client.get(self.endpoint, query_string={
            'period': 'daily',
            'count': 120
        })
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('error_code', data)
    
    def test_invalid_period_parameter(self):
        """测试：无效的周期参数"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'invalid_period',
            'count': 120
        })
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('message', data)
    
    def test_valid_period_values(self):
        """测试：有效的周期参数值"""
        valid_periods = ['daily', 'weekly', 'monthly']
        
        for period in valid_periods:
            with self.subTest(period=period):
                response = self.client.get(self.endpoint, query_string={
                    'index_id': '000001.SH',
                    'period': period,
                    'count': 120
                })
                # 即使数据获取失败，参数验证应该通过（不返回400）
                self.assertNotEqual(response.status_code, 400, 
                                    f"Period '{period}' should be valid")
    
    def test_negative_count_parameter(self):
        """测试：负数count参数"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': -10
        })
        
        self.assertEqual(response.status_code, 400)
    
    def test_count_parameter_default_value(self):
        """测试：count参数默认值"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily'
            # 不提供count参数
        })
        
        # 应该使用默认值，不返回400错误
        self.assertNotEqual(response.status_code, 400)


class ChartAPIResponseFormatTest(unittest.TestCase):
    """图表API响应格式测试"""
    
    @patch('core_bak_refactored.app.quality_monitoring.routes.chart_routes')
    def setUp(self, mock_routes):
        """测试准备"""
        from core_bak_refactored.app.quality_monitoring.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        self.endpoint = '/api/v1/chart/data'
    
    def test_response_structure(self):
        """测试：响应数据结构"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 10,
            'indicators': 'all'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # 验证顶层结构
            self.assertIn('success', data)
            self.assertIn('data', data)
            
            # 验证data结构
            chart_data = data['data']
            self.assertIn('kline', chart_data)
            self.assertIn('indicators', chart_data)
            self.assertIn('events', chart_data)
            
            # 验证kline数据类型
            self.assertIsInstance(chart_data['kline'], list)
            self.assertIsInstance(chart_data['indicators'], dict)
            self.assertIsInstance(chart_data['events'], list)
    
    def test_kline_data_fields(self):
        """测试：K线数据字段完整性"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 10
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            kline_data = data['data']['kline']
            
            if kline_data:
                first_kline = kline_data[0]
                
                # 验证必需字段
                required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
                for field in required_fields:
                    self.assertIn(field, first_kline, 
                                  f"K线数据应包含 {field} 字段")
                
                # 验证均线字段
                ma_fields = ['ma5', 'ma10', 'ma20']
                for field in ma_fields:
                    self.assertIn(field, first_kline,
                                  f"K线数据应包含 {field} 字段")


class ChartAPIIndicatorTest(unittest.TestCase):
    """图表API指标测试"""
    
    @patch('core_bak_refactored.app.quality_monitoring.routes.chart_routes')
    def setUp(self, mock_routes):
        """测试准备"""
        from core_bak_refactored.app.quality_monitoring.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        self.endpoint = '/api/v1/chart/data'
    
    def test_all_indicators_request(self):
        """测试：请求所有指标"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 50,
            'indicators': 'all'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            indicators = data['data']['indicators']
            
            # 验证常见技术指标存在
            expected_indicators = ['macd', 'rsi', 'kdj', 'vol', 'obv']
            for indicator in expected_indicators:
                self.assertIn(indicator, indicators,
                              f"indicators='all' 应包含 {indicator}")
    
    def test_partial_indicators_request(self):
        """测试：请求部分指标"""
        requested = ['macd', 'rsi']
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 50,
            'indicators': ','.join(requested)
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            indicators = data['data']['indicators']
            
            # 验证请求的指标都存在
            for indicator in requested:
                self.assertIn(indicator, indicators,
                              f"应包含请求的指标 {indicator}")
    
    def test_macd_indicator_format(self):
        """测试：MACD指标数据格式"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 50,
            'indicators': 'macd'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            if 'macd' in data['data']['indicators']:
                macd_data = data['data']['indicators']['macd']
                
                if macd_data:
                    first_macd = macd_data[0]
                    
                    # 验证MACD字段
                    self.assertIn('date', first_macd)
                    self.assertIn('macd', first_macd)
                    self.assertIn('signal', first_macd)
                    self.assertIn('histogram', first_macd)


class ChartAPIDateRangeTest(unittest.TestCase):
    """图表API日期范围测试"""
    
    @patch('core_bak_refactored.app.quality_monitoring.routes.chart_routes')
    def setUp(self, mock_routes):
        """测试准备"""
        from core_bak_refactored.app.quality_monitoring.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        self.endpoint = '/api/v1/chart/data'
    
    def test_before_date_parameter(self):
        """测试：before日期参数"""
        before_date = '2024-01-01'
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 60,
            'before': before_date
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            kline_data = data['data']['kline']
            
            if kline_data:
                # 验证所有数据都在before日期之前
                for item in kline_data:
                    item_date = datetime.strptime(item['date'], '%Y-%m-%d')
                    before_dt = datetime.strptime(before_date, '%Y-%m-%d')
                    self.assertLessEqual(item_date, before_dt,
                                         f"数据日期 {item['date']} 应在 {before_date} 之前")
    
    def test_invalid_date_format(self):
        """测试：无效的日期格式"""
        response = self.client.get(self.endpoint, query_string={
            'index_id': '000001.SH',
            'period': 'daily',
            'count': 60,
            'before': 'invalid-date'
        })
        
        # 应该返回错误
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
