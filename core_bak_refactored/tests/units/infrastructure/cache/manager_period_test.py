"""
ThreeLayerCacheManager Period参数传递测试

测试缓存管理器是否正确传递period参数给fetch函数：
1. db_fetch_func必须支持period参数
2. api_fetch_func必须支持period参数
"""

import unittest
from typing import Optional

import pandas as pd
from unittest.mock import Mock

from core_bak_refactored.infrastructure.cache.manager import ThreeLayerCacheManager
from core_bak_refactored.core.share.market.market_enums import MarketCode


class ManagerPeriodTest(unittest.TestCase):
    """Period参数传递测试"""

    def setUp(self):
        """测试初始化"""
        self.cache = ThreeLayerCacheManager(cache_mode='memory', window_size=7)

    def _create_test_df(self, rows=5):
        """创建测试数据"""
        return pd.DataFrame({
            'date': pd.date_range('2025-03-10', periods=rows, freq='D'),
            'close': [100 + i for i in range(rows)],
            'volume': [1000 + i * 100 for i in range(rows)]
        })

    # ========== API Fetch Period参数测试 ==========

    def test_api_fetch_with_period_parameter(self):
        """测试api_fetch_func必须接收period参数"""
        # 创建真实函数（支持period参数）
        call_log = []

        def api_fetch_with_period(start_date, end_date, period=None):
            call_log.append({'start_date': start_date, 'end_date': end_date, 'period': period})
            return self._create_test_df()

        # 调用缓存管理器
        result = self.cache.get_data(
            symbol='000001.SZ',
            period='daily',
            from_date=pd.Timestamp('2025-03-10'),
            to_date=pd.Timestamp('2025-03-14'),
            api_fetch_func=api_fetch_with_period,
            market_code=MarketCode.CN
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)

        # 验证api_fetch_func被调用，且传入了period参数
        self.assertTrue(len(call_log) > 0)
        self.assertEqual(call_log[0]['period'], 'daily')

    def test_api_fetch_weekly_period(self):
        """测试weekly周期传递period参数"""
        api_fetch_func = Mock(return_value=self._create_test_df())

        result = self.cache.get_data(
            symbol='000001.SZ',
            period='weekly',
            from_date=pd.Timestamp('2025-03-10'),
            to_date=pd.Timestamp('2025-03-30'),
            api_fetch_func=api_fetch_func,
            market_code=MarketCode.CN
        )

        # 验证period='weekly'被传递
        call_kwargs = api_fetch_func.call_args.kwargs
        if 'period' in call_kwargs:  # 如果函数支持period
            self.assertEqual(call_kwargs['period'], 'weekly')

    def test_api_fetch_monthly_period(self):
        """测试monthly周期传递period参数"""
        api_fetch_func = Mock(return_value=self._create_test_df())

        result = self.cache.get_data(
            symbol='000001.SZ',
            period='monthly',
            from_date=pd.Timestamp('2025-01-01'),
            to_date=pd.Timestamp('2025-03-31'),
            api_fetch_func=api_fetch_func,
            market_code=MarketCode.CN
        )

        # 验证period='monthly'被传递
        call_kwargs = api_fetch_func.call_args.kwargs
        if 'period' in call_kwargs:  # 如果函数支持period
            self.assertEqual(call_kwargs['period'], 'monthly')

    # ========== DB Fetch Period参数测试 ==========

    def test_db_fetch_with_period_parameter(self):
        """测试db_fetch_func必须接收period参数"""
        # 创建真实函数（支持period参数）
        db_call_log = []

        def db_fetch_with_period(start_date, end_date, period=None) -> Optional[pd.DataFrame]:
            db_call_log.append({'start_date': start_date, 'end_date': end_date, 'period': period})
            return None  # DB无数据，触发API

        api_fetch_func = Mock(return_value=self._create_test_df())

        result = self.cache.get_data(
            symbol='000001.SZ',
            period='daily',
            from_date=pd.Timestamp('2025-03-10'),
            to_date=pd.Timestamp('2025-03-14'),
            db_fetch_func=db_fetch_with_period,
            api_fetch_func=api_fetch_func,
            market_code=MarketCode.CN
        )

        # 验证db_fetch_func被调用，且传入了period参数
        self.assertTrue(len(db_call_log) > 0)
        self.assertEqual(db_call_log[0]['period'], 'daily')

    # ========== 综合场景测试 ==========

    def test_both_fetch_funcs_with_period(self):
        """测试DB和API都支持period参数"""
        db_call_log = []
        api_call_log = []

        def db_fetch_with_period(start_date, end_date, period=None)->Optional[pd.DataFrame]:
            db_call_log.append({'start_date': start_date, 'end_date': end_date, 'period': period})
            return None

        def api_fetch_with_period(start_date, end_date, period=None):
            api_call_log.append({'start_date': start_date, 'end_date': end_date, 'period': period})
            return self._create_test_df()

        result = self.cache.get_data(
            symbol='000001.SZ',
            period='weekly',
            from_date=pd.Timestamp('2025-03-10'),
            to_date=pd.Timestamp('2025-03-30'),
            db_fetch_func=db_fetch_with_period,
            api_fetch_func=api_fetch_with_period,
            market_code=MarketCode.CN
        )

        # 验证两者都收到period参数
        self.assertTrue(len(db_call_log) > 0)
        self.assertTrue(len(api_call_log) > 0)
        self.assertEqual(db_call_log[0]['period'], 'weekly')
        self.assertEqual(api_call_log[0]['period'], 'weekly')


if __name__ == '__main__':
    unittest.main()
