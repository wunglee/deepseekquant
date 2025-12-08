"""
数据库层单元测试

测试范围：
- SQLiteDatabase 基本操作
- MarketDataRepository 数据仓库功能
- 增量更新逻辑
- 数据去重机制
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
import pandas as pd

from core_bak_refactored.infrastructure.database import (
    SQLiteDatabase,
    MarketDataRepository,
    get_database
)


class TestSQLiteDatabase(unittest.TestCase):
    """SQLiteDatabase 单元测试"""
    
    def setUp(self):
        """测试前准备：创建内存数据库"""
        self.db = SQLiteDatabase(':memory:')
        self.db.connect()
    
    def tearDown(self):
        """测试后清理：关闭连接"""
        if self.db:
            self.db.close()
    
    def test_connect(self):
        """测试数据库连接"""
        self.assertTrue(self.db.is_connected)
        self.assertIsNotNone(self.db.connection)
    
    def test_execute(self):
        """测试SQL执行"""
        # 创建测试表
        cursor = self.db.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
        )
        self.assertIsNotNone(cursor)
        
        # 插入数据
        self.db.execute("INSERT INTO test_table (name) VALUES (?)", ('test',))
        self.db.commit()
        
        # 查询数据
        result = self.db.fetch_one("SELECT name FROM test_table WHERE id=1")
        self.assertEqual(result['name'], 'test')
    
    def test_fetch_all(self):
        """测试批量查询"""
        self.db.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value INTEGER)"
        )
        
        # 插入多条数据
        for i in range(5):
            self.db.execute("INSERT INTO test_table (value) VALUES (?)", (i,))
        self.db.commit()
        
        # 批量查询
        results = self.db.fetch_all("SELECT value FROM test_table ORDER BY value")
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]['value'], 0)
        self.assertEqual(results[4]['value'], 4)
    
    def test_insert_many(self):
        """测试批量插入"""
        self.db.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, value REAL)"
        )
        
        # 批量插入
        data = [
            {'name': 'a', 'value': 1.0},
            {'name': 'b', 'value': 2.0},
            {'name': 'c', 'value': 3.0}
        ]
        row_count = self.db.insert_many('test_table', data)
        
        self.assertEqual(row_count, 3)
        
        # 验证插入结果
        results = self.db.fetch_all("SELECT name, value FROM test_table ORDER BY name")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['name'], 'a')
    
    def test_transaction_commit(self):
        """测试事务提交"""
        self.db.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)"
        )
        
        with self.db.transaction():
            self.db.execute("INSERT INTO test_table (value) VALUES (?)", ('committed',))
        
        result = self.db.fetch_one("SELECT value FROM test_table WHERE id=1")
        self.assertEqual(result['value'], 'committed')
    
    def test_transaction_rollback(self):
        """测试事务回滚"""
        self.db.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)"
        )
        
        try:
            with self.db.transaction():
                self.db.execute("INSERT INTO test_table (value) VALUES (?)", ('test',))
                raise Exception("模拟错误")
        except:
            pass
        
        # 验证回滚
        results = self.db.fetch_all("SELECT * FROM test_table")
        self.assertEqual(len(results), 0)
    
    def test_table_exists(self):
        """测试表存在性检查"""
        self.assertFalse(self.db.table_exists('non_existent'))
        
        self.db.execute("CREATE TABLE test_table (id INTEGER)")
        self.assertTrue(self.db.table_exists('test_table'))


class TestMarketDataRepository(unittest.TestCase):
    """MarketDataRepository 单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.db = SQLiteDatabase(':memory:')
        self.db.connect()
        self.repository = MarketDataRepository(self.db)
    
    def tearDown(self):
        """测试后清理"""
        if self.db:
            self.db.close()
    
    def test_get_latest_date_empty(self):
        """测试获取最新日期（空数据库）"""
        latest = self.repository.get_latest_date('000300.SH')
        self.assertIsNone(latest)
    
    def test_get_date_range_empty(self):
        """测试获取日期范围（空数据库）"""
        date_range = self.repository.get_date_range('000300.SH')
        self.assertIsNone(date_range)
    
    def test_insert_and_query_prices(self):
        """测试插入和查询价格数据"""
        # 准备测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5, freq='D'),
            'close': [3000.0, 3010.0, 3020.0, 3030.0, 3040.0],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        # 插入数据
        row_count = self.repository.insert_prices('000300.SH', test_data, 'akshare')
        self.assertEqual(row_count, 5)
        
        # 查询数据
        result = self.repository.query_prices('000300.SH', '2025-01-01', '2025-01-05')
        self.assertEqual(len(result), 5)
        self.assertEqual(result['close'].iloc[0], 3000.0)
        self.assertEqual(result['volume'].iloc[-1], 1400000)
    
    def test_get_latest_date(self):
        """测试获取最新日期"""
        test_data = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=3, freq='D'),
            'close': [3000.0, 3010.0, 3020.0],
            'volume': [1000000, 1100000, 1200000]
        })
        
        self.repository.insert_prices('000300.SH', test_data, 'akshare')
        
        latest = self.repository.get_latest_date('000300.SH')
        self.assertEqual(latest, '2025-01-03')
    
    def test_get_date_range(self):
        """测试获取日期范围"""
        test_data = pd.DataFrame({
            'date': pd.date_range('2025-01-05', periods=10, freq='D'),
            'close': [3000.0 + i * 10 for i in range(10)],
            'volume': [1000000] * 10
        })
        
        self.repository.insert_prices('000300.SH', test_data, 'akshare')
        
        date_range = self.repository.get_date_range('000300.SH')
        self.assertEqual(date_range['start'], '2025-01-05')
        self.assertEqual(date_range['end'], '2025-01-14')
    
    def test_insert_duplicate_data(self):
        """测试插入重复数据（自动去重）"""
        test_data = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],
            'close': [3000.0, 3010.0],
            'volume': [1000000, 1100000]
        })
        
        # 第一次插入
        self.repository.insert_prices('000300.SH', test_data, 'akshare')
        
        # 第二次插入相同数据
        test_data['close'] = [3001.0, 3011.0]  # 修改价格
        self.repository.insert_prices('000300.SH', test_data, 'akshare')
        
        # 验证数据被更新而非重复
        result = self.repository.query_prices('000300.SH', '2025-01-01', '2025-01-02')
        self.assertEqual(len(result), 2)
        self.assertEqual(result['close'].iloc[0], 3001.0)  # 已更新
    
    def test_query_partial_range(self):
        """测试部分日期范围查询"""
        test_data = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=10, freq='D'),
            'close': [3000.0 + i * 10 for i in range(10)],
            'volume': [1000000] * 10
        })
        
        self.repository.insert_prices('000300.SH', test_data, 'akshare')
        
        # 查询部分范围
        result = self.repository.query_prices('000300.SH', '2025-01-03', '2025-01-07')
        self.assertEqual(len(result), 5)
        self.assertEqual(result['date'].min().strftime('%Y-%m-%d'), '2025-01-03')
        self.assertEqual(result['date'].max().strftime('%Y-%m-%d'), '2025-01-07')
    
    def test_multiple_indexes(self):
        """测试多个指数独立存储"""
        # 插入沪深300数据
        data_300 = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=3, freq='D'),
            'close': [3000.0, 3010.0, 3020.0],
            'volume': [1000000, 1100000, 1200000]
        })
        self.repository.insert_prices('000300.SH', data_300, 'akshare')
        
        # 插入上证50数据
        data_50 = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=3, freq='D'),
            'close': [2500.0, 2510.0, 2520.0],
            'volume': [500000, 510000, 520000]
        })
        self.repository.insert_prices('000016.SH', data_50, 'akshare')
        
        # 验证独立性
        result_300 = self.repository.query_prices('000300.SH', '2025-01-01', '2025-01-03')
        result_50 = self.repository.query_prices('000016.SH', '2025-01-01', '2025-01-03')
        
        self.assertEqual(len(result_300), 3)
        self.assertEqual(len(result_50), 3)
        self.assertEqual(result_300['close'].iloc[0], 3000.0)
        self.assertEqual(result_50['close'].iloc[0], 2500.0)


class TestDatabaseFactory(unittest.TestCase):
    """数据库工厂函数测试"""
    
    def test_get_database_memory(self):
        """测试创建内存数据库"""
        db = get_database('sqlite')
        self.assertIsInstance(db, SQLiteDatabase)
        self.assertEqual(db.database_path, ':memory:')
    
    def test_get_database_file(self):
        """测试创建文件数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            db = get_database('sqlite', database_path=temp_path)
            self.assertIsInstance(db, SQLiteDatabase)
            self.assertEqual(db.database_path, temp_path)
            
            db.connect()
            self.assertTrue(os.path.exists(temp_path))
            db.close()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_get_database_invalid_type(self):
        """测试无效数据库类型"""
        with self.assertRaises(ValueError):
            get_database('postgresql')


if __name__ == '__main__':
    unittest.main()
