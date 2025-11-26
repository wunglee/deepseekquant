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
    
    职责：生成符合专家第2轮答复的行业参数样本
    """
    
    # 专家第2轮参数范围（基准=0.10）
    INDUSTRY_CONFIGS = {
        'financial': {
            'mean': -0.150,
            'std': 0.020,
            'rationale': '系统性风险敏感度高'
        },
        'technology': {
            'mean': -0.120,
            'std': 0.022,
            'rationale': '成长性高但波动性大'
        },
        'cyclical': {
            'mean': -0.135,
            'std': 0.025,
            'rationale': '经济周期敏感性强'
        },
        'defensive': {
            'mean': -0.085,
            'std': 0.015,
            'rationale': '风险抵御能力强'
        }
    }
    
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
        np.random.seed(seed)
        
        industry_samples = {}
        for industry, config in cls.INDUSTRY_CONFIGS.items():
            industry_samples[industry] = cls._generate_samples(
                mean=config['mean'],
                std=config['std'],
                n=n_samples
            )
        
        return industry_samples
    
    @staticmethod
    def _generate_samples(mean: float, std: float, n: int) -> List[float]:
        """
        生成单个行业的冲击样本
        
        Args:
            mean: 平均冲击
            std: 标准差
            n: 样本量
        
        Returns:
            冲击系数样本列表
        """
        samples = np.random.normal(loc=mean, scale=std, size=n)
        # 限制在业务合理范围（-50%至+20%）
        samples = np.clip(samples, -0.5, 0.2)
        return samples.tolist()


class DataProcessingHelper:
    """
    数据处理辅助工具
    
    职责：提供通用的数据处理方法，消除重复代码
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
