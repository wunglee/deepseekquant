"""
分时图批次增量流程集成测试

测试范围：
- 前端请求 → API路由 → ChartDataAssembler → DataProvider → 数据生成
- 验证批次序号机制是否正确工作
- 验证增量数据是否真正不同

测试策略：
- 模拟前端发送不同批次序号
- 验证后端返回的数据确实不同
- 验证批次序号递增时数据也在变化
"""

import unittest
import json
from datetime import datetime
from core_bak_refactored.app.quality_monitoring.api_service import DataQualityAPIService
from core_bak_refactored.core.share.config_manager import ConfigManager


class IntradayBatchFlowIntegrationTest(unittest.TestCase):
    """分时图批次增量流程集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 创建配置管理器
        cls.config_manager = ConfigManager()
        
        # 创建API服务
        cls.api_service = DataQualityAPIService(cls.config_manager)
        cls.app = cls.api_service.app
        cls.client = cls.app.test_client()
    
    def test_batch_indices_different_data(self):
        """测试：不同批次序号返回不同的数据"""
        symbol = '600036.SH'
        
        # 第1次请求：批次序号 [1, 2, 3]
        response1 = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([1, 2, 3])}&'
            f'timestamps={json.dumps([1000, 1001, 1002])}'
        )
        self.assertEqual(response1.status_code, 200)
        data1 = response1.get_json()
        
        # 第2次请求：批次序号 [4, 5, 6]
        response2 = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([4, 5, 6])}&'
            f'timestamps={json.dumps([1003, 1004, 1005])}'
        )
        self.assertEqual(response2.status_code, 200)
        data2 = response2.get_json()
        
        # 验证返回成功
        self.assertEqual(data1['status'], 'success')
        self.assertEqual(data2['status'], 'success')
        
        # 验证返回的tick数量：3个批次 × 12个tick = 36个tick
        times1 = data1['data']['times']
        times2 = data2['data']['times']
        self.assertEqual(len(times1), 36, "批次[1,2,3]应返回36个tick")
        self.assertEqual(len(times2), 36, "批次[4,5,6]应返回36个tick")
        
        # 验证时间范围不同
        self.assertNotEqual(times1[0], times2[0], "不同批次的起始时间应不同")
        self.assertNotEqual(times1[-1], times2[-1], "不同批次的结束时间应不同")
        
        # 验证价格数据不同（由于使用批次序号作为随机种子）
        prices1 = data1['data']['prices']
        prices2 = data2['data']['prices']
        self.assertNotEqual(prices1, prices2, "不同批次的价格数据应不同")
        
        print(f"✅ 批次[1,2,3]时间范围: {times1[0]} - {times1[-1]}")
        print(f"✅ 批次[4,5,6]时间范围: {times2[0]} - {times2[-1]}")
        print(f"✅ 价格差异验证通过")
    
    def test_incremental_batch_progression(self):
        """测试：模拟前端增量请求，验证批次递增"""
        symbol = '600036.SH'
        
        # 模拟前端逻辑：首次加载空批次，后端返回最近30分钟
        response_initial = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([])}&'
            f'timestamps={json.dumps([])}'
        )
        self.assertEqual(response_initial.status_code, 200)
        data_initial = response_initial.get_json()
        self.assertEqual(data_initial['status'], 'success')
        
        # 验证首次加载返回30批次的数据：30 × 12 = 360个tick
        times_initial = data_initial['data']['times']
        self.assertEqual(len(times_initial), 360, "首次加载应返回360个tick（30个批次）")
        
        print(f"✅ 首次加载: {len(times_initial)}个tick, 时间范围: {times_initial[0]} - {times_initial[-1]}")
        
        # 模拟增量更新：请求下一个批次（假设当前批次序号为100）
        current_batch = 100
        response_inc1 = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([current_batch])}&'
            f'timestamps={json.dumps([2000])}'
        )
        self.assertEqual(response_inc1.status_code, 200)
        data_inc1 = response_inc1.get_json()
        
        # 验证增量返回1批次的数据：1 × 12 = 12个tick
        times_inc1 = data_inc1['data']['times']
        self.assertEqual(len(times_inc1), 12, "增量更新应返回12个tick（1个批次）")
        
        print(f"✅ 增量批次{current_batch}: {len(times_inc1)}个tick, 时间范围: {times_inc1[0]} - {times_inc1[-1]}")
        
        # 再次增量：请求下下个批次
        current_batch = 101
        response_inc2 = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([current_batch])}&'
            f'timestamps={json.dumps([2001])}'
        )
        self.assertEqual(response_inc2.status_code, 200)
        data_inc2 = response_inc2.get_json()
        
        times_inc2 = data_inc2['data']['times']
        self.assertEqual(len(times_inc2), 12, "第二次增量更新应返回12个tick")
        
        # 验证两次增量的时间范围不同
        self.assertNotEqual(times_inc1[0], times_inc2[0], "不同批次的时间应不同")
        
        print(f"✅ 增量批次{current_batch}: {len(times_inc2)}个tick, 时间范围: {times_inc2[0]} - {times_inc2[-1]}")
        print(f"✅ 增量递进验证通过")
    
    def test_multiple_batches_sequential(self):
        """测试：请求多个连续批次，验证数据连续性"""
        symbol = '600036.SH'
        
        # 请求批次 [10, 11, 12]
        response = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([10, 11, 12])}&'
            f'timestamps={json.dumps([3000, 3001, 3002])}'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        
        times = data['data']['times']
        prices = data['data']['prices']
        
        # 验证返回36个tick（3批次 × 12）
        self.assertEqual(len(times), 36)
        self.assertEqual(len(prices), 36)
        
        # 验证时间是递增的（连续批次应该时间连续）
        for i in range(1, len(times)):
            self.assertGreater(times[i], times[i-1], f"时间应递增: {times[i]} > {times[i-1]}")
        
        # 验证每12个tick对应一个批次（批次内时间间隔5秒）
        # 批次10: times[0:12]
        # 批次11: times[12:24]
        # 批次12: times[24:36]
        
        print(f"✅ 批次[10,11,12]返回{len(times)}个tick")
        print(f"✅ 时间范围: {times[0]} - {times[-1]}")
        print(f"✅ 时间连续性验证通过")
    
    def test_api_parameter_validation(self):
        """测试：API参数验证"""
        symbol = '600036.SH'
        
        # 测试1：batch_indices和timestamps长度不匹配
        response1 = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([1, 2])}&'
            f'timestamps={json.dumps([1000])}'  # 长度不匹配
        )
        self.assertEqual(response1.status_code, 400)
        data1 = response1.get_json()
        self.assertEqual(data1['status'], 'error')
        self.assertIn('长度必须相同', data1['message'])
        
        print(f"✅ 参数长度不匹配验证通过: {data1['message']}")
        
        # 测试2：无效的JSON格式
        response2 = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices=invalid_json&'
            f'timestamps={json.dumps([1000])}'
        )
        self.assertEqual(response2.status_code, 400)
        data2 = response2.get_json()
        self.assertEqual(data2['status'], 'error')
        self.assertIn('无效的JSON', data2['message'])
        
        print(f"✅ 无效JSON验证通过: {data2['message']}")
    
    def test_empty_batch_indices_returns_initial_data(self):
        """测试：空批次序号应返回初始数据（最近30分钟）"""
        symbol = '600036.SH'
        
        response = self.client.get(
            f'/api/v1/intraday/data?symbol={symbol}&'
            f'batch_indices={json.dumps([])}&'
            f'timestamps={json.dumps([])}'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        
        times = data['data']['times']
        # 应返回30批次 × 12 = 360个tick
        self.assertEqual(len(times), 360, "空批次应返回初始30分钟数据（360个tick）")
        
        print(f"✅ 空批次序号返回初始数据: {len(times)}个tick")
        print(f"✅ 时间范围: {times[0]} - {times[-1]}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
