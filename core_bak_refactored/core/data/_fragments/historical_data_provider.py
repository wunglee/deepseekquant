"""
功能碎片：历史数据提供者
从 core/risk/backtest_framework.py 提取
状态：待整合到 core/data 模块

职责：
- 历史价格数据获取
- 多数据源适配（Yahoo Finance / JoinQuant / Wind）
- 数据预处理和格式化

迁移计划：
当 core_bak_refactored/core/data 模块开发完成后，整合此文件到该模块

相关文件：
- 源文件：core/risk/backtest_framework.py (HistoricalDataProvider, MockHistoricalDataProvider)
- 调用者：core/risk/stress_test_validator.py (StressTestValidator)
"""

import numpy as np
import pandas as pd
from typing import Protocol
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.DataFragments')


# =============================================================================
# 协议接口（数据模块标准）
# =============================================================================

class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（数据模块标准接口）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    """
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
        """
        ...
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series with date index and return values
        """
        ...


# =============================================================================
# 模拟实现（临时，等待真实数据源替换）
# =============================================================================

class MockHistoricalDataProvider:
    """
    模拟历史数据提供者（功能碎片临时实现）
    
    用途：
    - 在core/data模块未完成前，提供测试数据
    - 基于真实历史事件参数生成合理的模拟数据
    - 支持框架功能验证和测试
    
    警告：
    - 数据为模拟生成，仅用于框架验证
    - 真实回测需要替换为RealHistoricalDataProvider
    
    迁移计划：
    - 当core/data完成后，此类迁移到data模块作为Mock工具
    """
    
    def __init__(self):
        self.event_params = {
            '2015_china_market_crash': {
                'period': ('2015-06-15', '2015-08-26'),
                'expected_decline': -0.43,
                'volatility_multiplier': 2.5
            },
            'covid_19_pandemic': {
                'period': ('2020-02-20', '2020-03-23'),
                'expected_decline': -0.20,
                'volatility_multiplier': 3.0
            },
            '2008_financial_crisis': {
                'period': ('2008-09-15', '2008-11-20'),
                'expected_decline': -0.40,
                'volatility_multiplier': 3.5
            },
            '2011_eurozone_debt_crisis': {
                'period': ('2011-09-01', '2011-11-30'),
                'expected_decline': -0.25,
                'volatility_multiplier': 2.5
            },
            '2011_us_debt_ceiling_crisis': {
                'period': ('2011-07-22', '2011-08-10'),
                'expected_decline': -0.12,
                'volatility_multiplier': 2.0
            }
        }
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟的指数价格数据"""
        # 解析日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # 交易日
        
        # 检查是否在已知事件窗口内
        event_decline = 0.0
        event_vol = 1.0
        for event_id, params in self.event_params.items():
            event_start = pd.to_datetime(params['period'][0])
            event_end = pd.to_datetime(params['period'][1])
            if start >= event_start and end <= event_end:
                event_decline = params['expected_decline']
                event_vol = params['volatility_multiplier']
                logger.info(f"检测到事件窗口: {event_id}, decline={event_decline}, vol={event_vol}")
                break
        
        # 生成模拟价格序列（确定性趋势 + 随机波动）
        n_days = len(dates)
        initial_price = 3000.0  # 沪深300典型水平
        
        # 基于事件参数生成收益率
        base_volatility = 0.015  # 1.5%日波动率
        daily_volatility = base_volatility * event_vol
        
        # 生成确定性下跌趋势（事件期间）+ 随机波动
        if event_decline != 0.0 and n_days > 0:
            # 精确匹配总收益率到expected_decline
            daily_drift = max(-0.95, (1.0 + event_decline) ** (1.0 / n_days) - 1.0)
            random_component = np.zeros(n_days)
            daily_returns = daily_drift + random_component
        else:
            # 非事件期间：纯随机游走
            daily_returns = np.random.normal(0, daily_volatility, n_days)
        
        # 计算价格序列
        prices = initial_price * np.cumprod(1 + daily_returns)
        
        # 生成成交量（简化模拟）
        base_volume = 100000000  # 1亿手
        volumes = base_volume * (1 + np.random.uniform(-0.3, 0.5, n_days))
        volumes = np.clip(volumes, 0, None)
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        logger.debug(f"生成模拟数据: {index_id}, {len(df)}天, 总收益率={prices[-1]/prices[0]-1:.2%}")
        return df
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """获取指数收益率序列"""
        df = self.get_index_prices(index_id, start_date, end_date)
        returns = df['close'].pct_change().fillna(0)
        returns.index = df['date']
        return returns


# =============================================================================
# 真实数据提供者（待实现）
# =============================================================================

class RealHistoricalDataProvider:
    """
    真实历史数据提供者（最小可用实现）
    
    当前迭代目标：在不越界的前提下，提供可运行的适配层。
    - 默认使用 mock 数据源，保证端到端可运行
    - 当 data_source='yahoo' 时，尝试使用 yfinance（若不可用或失败则回退到 mock）
    
    注意：网络依赖与第三方库非本轮迭代重点，故仅作可选尝试。
    """
    
    def __init__(self, data_source: str = 'mock'):
        self.data_source = data_source
        self._mock = MockHistoricalDataProvider()
        self._cache = {}
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        # 优先使用缓存
        cache_key = f"prices:{index_id}:{start_date}:{end_date}:{self.data_source}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.data_source == 'mock':
            data = self._mock.get_index_prices(index_id, start_date, end_date)
            self._cache[cache_key] = data
            return data
        
        if self.data_source == 'yahoo':
            try:
                import pandas as _pd
                # 延迟导入，避免环境无依赖时报错
                import yfinance as yf
                df = yf.download(index_id, start=start_date, end=end_date, progress=False)
                # 统一格式
                if not df.empty:
                    out = _pd.DataFrame({
                        'date': _pd.to_datetime(df.index),
                        'close': df['Close'].astype(float),
                        'volume': df.get('Volume', _pd.Series([0]*len(df))).astype(float)
                    })
                    self._cache[cache_key] = out
                    return out
                else:
                    logger.warning("yahoo数据为空，回退mock")
                    data = self._mock.get_index_prices(index_id, start_date, end_date)
                    self._cache[cache_key] = data
                    return data
            except Exception as e:
                logger.error(f"yahoo获取失败: {e}，回退mock")
                data = self._mock.get_index_prices(index_id, start_date, end_date)
                self._cache[cache_key] = data
                return data
        
        # 未知数据源，回退mock
        logger.warning(f"未知数据源 {self.data_source}，回退mock")
        data = self._mock.get_index_prices(index_id, start_date, end_date)
        self._cache[cache_key] = data
        return data
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        df = self.get_index_prices(index_id, start_date, end_date)
        returns = df['close'].pct_change().fillna(0)
        returns.index = df['date']
        return returns


# =============================================================================
# 迁移检查清单
# =============================================================================

"""
功能碎片迁移检查清单（core/data模块开发时使用）

□ 1. 接口标准化
    □ 确认 HistoricalDataProvider 协议符合 core/data 模块设计
    □ 添加更多方法（如 get_stock_prices, get_option_data 等）
    
□ 2. 真实数据源集成
    □ 实现 YahooFinanceAdapter
    □ 实现 JoinQuantAdapter
    □ 实现 WindAdapter（可选）
    
□ 3. Mock 数据优化
    □ 将 MockHistoricalDataProvider 迁移到 core/data/mocks/
    □ 支持更多事件场景
    □ 改进模拟数据质量（基于真实统计特征）
    
□ 4. 数据缓存机制
    □ 实现本地缓存（文件系统）
    □ 实现内存缓存（Redis可选）
    □ 缓存过期策略
    
□ 5. 异常处理
    □ 网络异常降级
    □ 数据不完整告警
    □ 数据质量验证
    
□ 6. 调用者更新
    □ 更新 core/risk/stress_test_validator.py 的导入路径
    □ 更新测试用例
    □ 更新文档示例
    
□ 7. 测试覆盖
    □ 单元测试（各数据源）
    □ 集成测试（端到端数据获取）
    □ Mock 与真实数据对比测试
"""
