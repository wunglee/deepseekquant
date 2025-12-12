"""
数据质量工具函数
提供通用的数据质量评估函数，避免重复实现

强类型设计原则（2025-12-11）:
- 优先使用PriceData替代pd.DataFrame
- 提供向后兼容的DataFrame接口
- 实现强类型约束，提高代码健壮性
"""

import pandas as pd
import numpy as np
from typing import Tuple
from core_bak_refactored.core.data.providers.protocols import PriceData
from core_bak_refactored.core.data.quality.quality_types import DataQualityReport


def calculate_consistency_score(data: pd.DataFrame) -> float:
    """计算数据一致性评分
    
    检查数值列的类型一致性
    
    Args:
        data: 待检查的DataFrame
        
    Returns:
        一致性评分(0-1)
    """
    if data.empty:
        return 0.0
    
    # 检查数值列的数据类型一致性
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        return 0.5  # 如果没有数值列，给中等分数
        
    # 检查是否有混合类型的数据
    consistency_issues = 0
    for col in numeric_columns:
        if data[col].dtype == 'object':
            # 尝试转换为数值
            try:
                pd.to_numeric(data[col], errors='raise')
            except (ValueError, TypeError):
                consistency_issues += 1
    
    # 计算一致性评分
    consistency_score = 1.0 - (consistency_issues / len(numeric_columns)) if len(numeric_columns) > 0 else 1.0
    return max(0.0, consistency_score)  # 确保不为负数


def calculate_accuracy_score(data: pd.DataFrame) -> float:
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
    
    # 检查价格数据是否合理
    if 'close' in data.columns:
        close_prices = data['close']
        # 检查是否有负价格
        negative_prices = (close_prices < 0).sum()
        if negative_prices > 0:
            accuracy_score -= 0.2 * (negative_prices / len(close_prices))
        
        # 检查是否有异常大的价格
        if len(close_prices) > 0:
            mean_price = close_prices.mean()
            if mean_price > 0:
                # 检查超过均值10倍的价格
                extreme_prices = (close_prices > mean_price * 10).sum()
                if extreme_prices > 0:
                    accuracy_score -= 0.1 * (extreme_prices / len(close_prices))
    
    # 确保评分在0-1范围内
    return max(0.0, min(1.0, accuracy_score))


def detect_outliers(data: pd.DataFrame) -> int:
    """检测异常值数量（使用IQR方法）
    
    使用四分位距(IQR)检测异常值:
    - 异常值定义: < Q1 - 1.5*IQR 或 > Q3 + 1.5*IQR
    
    Args:
        data: 待检测的DataFrame
        
    Returns:
        异常值总数
    """
    if data.empty:
        return 0
        
    outliers = 0
    
    # 对数值列进行异常值检测
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        series = data[col]
        # 使用IQR方法检测异常值
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        outliers += col_outliers
        
    return outliers


def validate_data_quality(price_data: PriceData) -> DataQualityReport:
    """
    数据质量验证报告（强类型版本，使用PriceData）
    
    从yahoo_provider.py、tushare_provider.py和akshare_provider.py迁移合并而来，统一实现。
    
    评分维度：
    - 完整性（30%）：基于缺失值比例
    - 一致性（30%）：数据类型一致性检查
    - 准确性（20%）：价格合理性检查（负价格、极端价格）
    - 异常值（20%）：IQR方法检测异常值
    
    Args:
        price_data: PriceData对象
        
    Returns:
        DataQualityReport: 数据质量报告（包含overall_score综合评分）
    """
    # 将PriceData转换为DataFrame进行计算
    data = price_data.to_dataframe()
    
    if data is None or data.empty:
        return DataQualityReport(
            completeness_score=0.0,
            consistency_score=0.0,
            accuracy_score=0.0,
            outliers_detected=0,
            total_rows=0,
            missing_values=0
        )
    
    total_rows = len(data)
    
    # 计算缺失值
    missing_values = data.isnull().sum().sum()
    
    # 完整性评分 (基于缺失值比例)
    completeness_score = 1.0 - (missing_values / (total_rows * len(data.columns))) if total_rows > 0 else 0.0
    
    # 一致性评分
    consistency_score = calculate_consistency_score(data)
    
    # 准确性评分
    accuracy_score = calculate_accuracy_score(data)
    
    # 异常值检测
    outliers_detected = detect_outliers(data)
    
    # 创建报告（会自动计算overall_score）
    report = DataQualityReport(
        completeness_score=completeness_score,
        consistency_score=consistency_score,
        accuracy_score=accuracy_score,
        outliers_detected=outliers_detected,
        total_rows=total_rows,
        missing_values=missing_values
    )
    
    return report