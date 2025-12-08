"""[专家碎片] 数据质量验证器 - 第6轮专家指导(迁移至 quality 子域)

架构变更说明（2025-12-06）：
- 原名：DataQualityEnhancer（数据质量增强器）
- 原职责：多源数据智能切换、质量驱动切换、主源优先备源降级
- 新职责：单纯的数据质量验证与评分，不再负责数据源切换
- 设计原则：单一职责原则（SRP），仅评估数据质量，不参与数据获取决策

新架构规则（2025-12-06）：
- 用户指定单一数据源（primary_source），不再自动切换
- 本类仅负责数据质量评分，返回质量报告供调用方决策
- 移除了backup_sources参数和多源切换逻辑

职责:
- 数据质量验证与评分
- 完整的质量评分体系(完整性+一致性+准确性+异常检测)
- 提供标准化的质量报告

设计模式:
- 策略模式: 支持多种质量评估策略
- 报告模式: 标准化的质量报告输出
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger('DeepSeekQuant.DataQualityEnhancer')

# 数据质量阈值常量
DEFAULT_QUALITY_THRESHOLD = 0.8  # 默认质量阈值
MIN_DATA_ROWS = 1  # 最少数据行数


@dataclass
class DataQualityReport:
    """数据质量报告
    
    Attributes:
        completeness_score: 完整性评分(0-1)
        consistency_score: 一致性评分(0-1)
        accuracy_score: 准确性评分(0-1)
        outliers_detected: 检测到的异常值数量
        total_rows: 总行数
        missing_values: 缺失值总数
        overall_score: 综合评分(0-1)
    """
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    outliers_detected: int
    total_rows: int
    missing_values: int
    overall_score: float

class DataQualityEnhancer:
    """数据质量验证器
    
    新架构设计（2025-12-06）：
    - 仅负责对数据进行质量评估
    - 返回DataQualityReport供调用方决策
    
    职责:
    - 数据质量验证与评分
    - 提供标准化的质量报告
    
    Args:
        quality_threshold: 质量阈值(默认0.8)，仅用于报告对比
    """

    def __init__(
        self,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    ):
        """初始化数据质量验证器
        
        Args:
            quality_threshold: 质量阈值(0-1)
        """
        if quality_threshold < 0 or quality_threshold > 1:
            raise ValueError(f"quality_threshold必须在0-1之间,当前值: {quality_threshold}")
        
        self.quality_threshold = quality_threshold
        
        logger.info(
            f"DataQualityEnhancer初始化: 质量阈值={quality_threshold} "
            "(仅用于报告对比)"
        )

    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """验证数据质量
        
        评分维度:
        - 完整性(30%):缺失值比例
        - 一致性(30%):数据类型一致性
        - 准确性(20%):负价格, 极端价格检测
        - 异常值(20%):IQR方法检测异常值
        
        Args:
            data: 待验证的DataFrame
        
        Returns:
            DataQualityReport对象
        """
        # 空数据处理
        if data is None or data.empty:
            logger.debug("数据为空,返回零分质量报告")
            return DataQualityReport(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                outliers_detected=0,
                total_rows=0,
                missing_values=0,
                overall_score=0.0
            )
        # 计算各维度评分
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)
        missing_values = data.isnull().sum().sum()
        
        completeness_score = (
            1.0 - (missing_values / total_cells)
            if total_cells > 0 else 0.0
        )
        consistency_score = self._calculate_consistency_score(data)
        accuracy_score = self._calculate_accuracy_score(data)
        outliers_detected = self._detect_outliers(data)
        
        # 综合评分(加权平均)
        outlier_penalty = min(1.0, outliers_detected / max(1, total_rows))
        overall_score = (
            0.3 * completeness_score +
            0.3 * consistency_score +
            0.2 * accuracy_score +
            0.2 * (1.0 - outlier_penalty)
        )
        
        report = DataQualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            outliers_detected=int(outliers_detected),
            total_rows=int(total_rows),
            missing_values=int(missing_values),
            overall_score=overall_score
        )
        
        logger.debug(
            f"质量评分: 总分={overall_score:.3f}, "
            f"完整性={completeness_score:.3f}, "
            f"一致性={consistency_score:.3f}, "
            f"准确性={accuracy_score:.3f}, "
            f"异常值={outliers_detected}"
        )
        
        return report
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算数据一致性评分
        
        检查数值列的类型一致性
        
        Args:
            data: 待检查的DataFrame
        
        Returns:
            一致性评分(0-1)
        """
        if data.empty:
            return 0.0
        numeric_columns = data.select_dtypes(include=['number']).columns
        if len(numeric_columns) == 0:
            return 0.5
        consistency_issues = 0
        for col in numeric_columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    consistency_issues += 1
        consistency_score = 1.0 - (consistency_issues / len(numeric_columns)) if len(numeric_columns) > 0 else 1.0
        return max(0.0, consistency_score)
    
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """计算数据准确性评分
        
        检查项:
        - 负价格(扣0.2 * 比例)
        - 极端价格(超过均值10倍,扣0.1 * 比例)
        
        Args:
            data: 待检查的DataFrame
        
        Returns:
            准确性评分(0-1)
        """
        if data.empty:
            return 0.0
        accuracy_score = 1.0
        if 'close' in data.columns:
            close_prices = data['close']
            negative_prices = (close_prices < 0).sum()
            if negative_prices > 0:
                accuracy_score -= 0.2 * (negative_prices / len(close_prices))
            if len(close_prices) > 0:
                mean_price = close_prices.mean()
                if mean_price > 0:
                    extreme_prices = (close_prices > mean_price * 10).sum()
                    if extreme_prices > 0:
                        accuracy_score -= 0.1 * (extreme_prices / len(close_prices))
        return max(0.0, min(1.0, accuracy_score))
    
    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """检测异常值(IQR方法)
        
        使用四分位距(IQR)检测异常值:
        - 异常值定义:< Q1 - 1.5*IQR 或 > Q3 + 1.5*IQR
        
        Args:
            data: 待检测的DataFrame
        
        Returns:
            异常值总数
        """
        if data.empty:
            return 0
        outliers = 0
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            series = data[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outliers += col_outliers
        return outliers
