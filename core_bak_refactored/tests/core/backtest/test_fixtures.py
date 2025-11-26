"""
测试fixture和辅助工具 - 5B-5阶段A+B

职责：
1. 提供标准化的测试事件配置
2. 封装通用的数据生成逻辑
3. 提供可复用的断言辅助方法

设计原则：
- 单一职责：每个类只负责一类测试数据或工具
- 可复用性：所有测试用例共享同一套配置
- 可维护性：配置集中管理，易于调整

架构说明（重要）：
本文件为测试辅助工具，部分功能是对业务模块能力的封装：
- IndustrySampleGenerator: 调用 core.risk.IndustryParameterAnalyzer.generate_test_samples()
- DataProcessingHelper: 封装 core.data 的数据处理能力（未来可迁移）
- TestAssertionHelper: 纯测试断言逻辑，职责合理

未来优化方向：
当业务模块提供完整的辅助方法后，本文件应仅保留测试编排逻辑，
删除业务逻辑的重复实现，改为直接调用业务模块接口。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class EventConfig:
    """历史事件配置数据类"""
    event_id: str
    index_id: str
    event_date: str
    event_type: str
    expected_decline: float
    market_id: str


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
    数据处理辅助工具
    
    职责：提供通用的数据处理方法，消除重复代码
    
    架构说明：
    - 本类封装了data模块的数据处理能力
    - 理想情况应直接调用 core.data 提供的工具方法
    - 当前作为过渡方案，避免测试代码重复
    
    未来优化：
    - 将这些方法迁移到 core.data._fragments.data_utils 模块
    - 测试直接调用业务模块接口
    """
    
    @staticmethod
    def calculate_actual_return(event_window_df: pd.DataFrame) -> float:
        """
        计算事件窗口实际收益率
        
        Args:
            event_window_df: 事件窗口数据
        
        Returns:
            实际收益率（如果数据不足返回0.0）
        """
        if len(event_window_df) >= 2:
            return (event_window_df['close'].iloc[-1] / 
                   event_window_df['close'].iloc[0] - 1)
        return 0.0
    
    @staticmethod
    def safe_get_event_data(data_provider, event: EventConfig, 
                          window_days: int = 30, 
                          baseline_days: int = 252) -> Tuple[pd.DataFrame, bool]:
        """
        安全获取事件数据（带异常处理）
        
        Args:
            data_provider: 数据提供者
            event: 事件配置
            window_days: 事件窗口天数
            baseline_days: 基准窗口天数
        
        Returns:
            (事件窗口数据, 是否成功)
        """
        try:
            window_data = data_provider.get_event_window_data(
                index_id=event.index_id,
                event_date=event.event_date,
                window_days=window_days,
                baseline_days=baseline_days
            )
            return window_data['event_window'], True
        except Exception:
            return pd.DataFrame(), False
    
    @staticmethod
    def calculate_prediction_error(actual_return: float, expected_decline: float) -> float:
        """
        计算预测误差
        
        Args:
            actual_return: 实际收益率
            expected_decline: 预期下跌幅度
        
        Returns:
            预测误差（绝对值）
        """
        return abs(actual_return - expected_decline)


class TestAssertionHelper:
    """
    测试断言辅助工具
    
    职责：提供标准化的断言方法，增强错误信息可读性
    """
    
    @staticmethod
    def assert_quality_score(test_case, quality_report, threshold: float = 0.60, use_real_data: bool = False):
        """
        断言数据质量评分
        
        Args:
            test_case: unittest.TestCase实例
            quality_report: 数据质量报告
            threshold: 阈值（默认60%）
            use_real_data: 是否使用真实数据（影响阈值）
        """
        actual_threshold = 0.90 if use_real_data else threshold
        test_case.assertGreaterEqual(
            quality_report.overall_score,
            actual_threshold,
            f"数据质量评分过低: {quality_report.overall_score:.2%} < {actual_threshold:.0%}"
        )
    
    @staticmethod
    def assert_error_within_threshold(test_case, error: float, threshold: float, context: str = ""):
        """
        断言误差在阈值内
        
        Args:
            test_case: unittest.TestCase实例
            error: 误差值
            threshold: 阈值
            context: 上下文信息
        """
        test_case.assertLessEqual(
            error,
            threshold,
            f"{context}误差超限: {error:.2%} > {threshold:.0%}"
        )
