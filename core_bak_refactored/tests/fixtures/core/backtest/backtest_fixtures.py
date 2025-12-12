"""
测试fixture和辅助工具 - 5B-5阶段A+B

职责：
1. 提供标准化的测试事件配置
2. 封装对业务模块的调用（委托模式）
3. 向后兼容的便利性包装

设计原则：
- 单一职责：仅负责测试编排，不实现业务逻辑
- 委托模式：所有业务逻辑调用业务模块接口
- 向后兼容：保持原有API不变，内部委托给业务模块

架构说明（重要）：
本文件为测试辅助工具，所有功能均委托给业务模块：
- IndustrySampleGenerator: 委托 core.risk.IndustryParameterAnalyzer
- DataProcessingHelper: 委托 core.data.DataUtils
- TestAssertionHelper: 委托 tests.common.TestAssertions

职责归位完成：
✅ 业务逻辑已迁移到业务模块
✅ 测试模块仅保留便利性包装
✅ 符合ARCHITECTURE.md的模块边界
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 从业务模块导入EventConfig（职责归位）
from core_bak_refactored.core.backtest.event_analysis import EventConfig


class TestEventProvider:
    """
    测试事件数据提供者
    
    职责：提供5个标准历史事件配置（专家第2轮5.4节）
    """
    
    # 标准事件窗口配置（避免硬编码重复）
    DEFAULT_WINDOW_DAYS = 30
    DEFAULT_BASELINE_DAYS = 252
    
    @staticmethod
    def get_standard_events() -> List[EventConfig]:
        """
        获取5个典型历史事件配置
        
        Returns:
            标准事件配置列表
        """
        return [
            EventConfig(
                event_id='2008_financial_crisis',
                index_id='000300.SH',
                event_date='2008-09-15',
                event_type='market_crash',
                expected_decline=-0.40,
                market_id='CN'
            ),
            EventConfig(
                event_id='2015_china_market_crash',
                index_id='000300.SH',
                event_date='2015-06-15',
                event_type='market_crash',
                expected_decline=-0.43,
                market_id='CN'
            ),
            EventConfig(
                event_id='covid_19_pandemic',
                index_id='000300.SH',
                event_date='2020-02-20',
                event_type='market_crash',
                expected_decline=-0.20,
                market_id='CN'
            ),
            EventConfig(
                event_id='2022_russia_ukraine_conflict',
                index_id='000300.SH',
                event_date='2022-02-24',
                event_type='geopolitical_risk',
                expected_decline=-0.12,
                market_id='CN'
            ),
            EventConfig(
                event_id='2011_eurozone_debt_crisis',
                index_id='000300.SH',
                event_date='2011-09-01',
                event_type='sovereign_debt_crisis',
                expected_decline=-0.25,
                market_id='CN'
            )
        ]


class IndustrySampleGenerator:
    """
    行业样本数据生成器
    
    职责：封装 core.risk.IndustryParameterAnalyzer.generate_test_samples() 方法
    
    架构说明：
    - 业务逻辑已归位到 core.risk.IndustryParameterAnalyzer
    - 本类为测试便利性包装，直接调用业务模块方法
    - 避免测试模块承载业务逻辑
    """
    
    @classmethod
    def generate_all_industries(cls, n_samples: int = 1200, seed: int = 42) -> Dict[str, List[float]]:
        """
        生成所有行业的样本数据
        
        Args:
            n_samples: 样本量（默认1200，满足≥1000要求）
            seed: 随机种子
        
        Returns:
            行业样本字典 {industry_name: samples}
        """
        # 调用业务模块方法（职责归位）
        from core_bak_refactored.core.risk.stress_testing import IndustryParameterAnalyzer
        return IndustryParameterAnalyzer.generate_test_samples(n_samples=n_samples, seed=seed)


class DataProcessingHelper:
    """
    数据处理辅助工具（委托给业务模块）
    
    职责：提供向后兼容的API，委托给 core.data.DataUtils
    
    架构说明：
    - 业务逻辑已迁移到 core.data._fragments.data_utils.DataUtils
    - 本类仅作为便利性包装层，保持向后兼容
    - 所有方法直接委托给业务模块
    """
    
    @staticmethod
    def calculate_actual_return(event_window_df: pd.DataFrame) -> float:
        """
        计算事件窗口实际收益率（委托给EventAnalyzer）
        
        Args:
            event_window_df: 事件窗口数据
        
        Returns:
            实际收益率（如果数据不足返回0.0）
        """
        from core_bak_refactored.core.backtest.event_analysis import EventAnalyzer
        return EventAnalyzer.calculate_actual_return(event_window_df)
    
    @staticmethod
    def safe_get_event_data(data_provider, event: EventConfig, 
                          window_days: int = 30, 
                          baseline_days: int = 252) -> Tuple[pd.DataFrame, bool]:
        """
        安全获取事件数据（委托给EventAnalyzer）
        
        Args:
            data_provider: 数据提供者
            event: 事件配置
            window_days: 事件窗口天数
            baseline_days: 基准窗口天数
        
        Returns:
            (事件窗口数据, 是否成功)
        """
        from core_bak_refactored.core.backtest.event_analysis import EventAnalyzer
        return EventAnalyzer.safe_get_event_data(data_provider, event, window_days, baseline_days)
    
    @staticmethod
    def calculate_prediction_error(actual_return: float, expected_decline: float) -> float:
        """
        计算预测误差（委托给EventAnalyzer）
        
        Args:
            actual_return: 实际收益率
            expected_decline: 预期下跌幅度
        
        Returns:
            预测误差（绝对值）
        """
        from core_bak_refactored.core.backtest.event_analysis import EventAnalyzer
        return EventAnalyzer.calculate_prediction_error(actual_return, expected_decline)


class TestAssertionHelper:
    """
    测试断言辅助工具（委托给通用测试工具）
    
    职责：提供向后兼容的API，委托给 tests.common.TestAssertions
    
    架构说明：
    - 断言逻辑已迁移到 tests.common.test_assertions.TestAssertions
    - 本类仅作为便利性包装层，保持向后兼容
    - 所有方法直接委托给通用测试工具
    """
    
    @staticmethod
    def assert_quality_score(test_case, quality_report, threshold: float = 0.60, use_real_data: bool = False):
        """
        断言数据质量评分（委托给TestAssertions）
        
        Args:
            test_case: unittest.TestCase实例
            quality_report: 数据质量报告
            threshold: 阈值（默认60%）
            use_real_data: 是否使用真实数据（影响阈值）
        """
        from core_bak_refactored.tests.common.assertions import TestAssertions
        TestAssertions.assert_quality_score(test_case, quality_report, threshold, use_real_data)
    
    @staticmethod
    def assert_error_within_threshold(test_case, error: float, threshold: float, context: str = ""):
        """
        断言误差在阈值内（委托给TestAssertions）
        
        Args:
            test_case: unittest.TestCase实例
            error: 误差值
            threshold: 阈值
            context: 上下文信息
        """
        from core_bak_refactored.tests.common.assertions import TestAssertions
        TestAssertions.assert_error_within_threshold(test_case, error, threshold, context)
