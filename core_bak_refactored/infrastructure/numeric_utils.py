"""
数值转换和处理工具
从risk模块57处float()和12处float(abs())调用中提炼的安全转换器
"""

import numpy as np
import pandas as pd
from typing import Any, Union, Optional
import logging

logger = logging.getLogger('DeepSeekQuant.NumericUtils')


class SafeNumericConverter:
    """安全数值转换器（统一57处float转换）"""
    
    @staticmethod
    def to_float(
        value: Any,
        default: float = 0.0,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> float:
        """
        安全转换为float
        
        Args:
            value: 输入值
            default: 转换失败或无效时的默认值
            allow_nan: 是否允许NaN
            allow_inf: 是否允许Inf
        
        Returns:
            float值或默认值
        
        示例:
            value = SafeNumericConverter.to_float(raw_value, default=0.0)
        """
        try:
            result = float(value)
            
            # NaN检查
            if not allow_nan and np.isnan(result):
                logger.debug(f"转换结果为NaN，返回默认值{default}")
                return default
            
            # Inf检查
            if not allow_inf and np.isinf(result):
                logger.debug(f"转换结果为Inf，返回默认值{default}")
                return default
            
            return result
        except (TypeError, ValueError) as e:
            logger.debug(f"转换为float失败({value}): {e}，返回默认值{default}")
            return default
    
    @staticmethod
    def to_positive_float(
        value: Any,
        default: float = 0.0
    ) -> float:
        """
        安全转换为正float（自动取绝对值）
        
        统一12处 float(abs(...)) 模式
        
        Args:
            value: 输入值
            default: 转换失败时的默认值
        
        Returns:
            正float值或默认值
        
        示例:
            var_value = SafeNumericConverter.to_positive_float(calculated_var, 0.0)
        """
        try:
            result = float(abs(value))
            
            if np.isnan(result) or np.isinf(result):
                logger.debug(f"转换结果无效({result})，返回默认值{default}")
                return default
            
            return result
        except (TypeError, ValueError) as e:
            logger.debug(f"转换为正float失败({value}): {e}，返回默认值{default}")
            return default
    
    @staticmethod
    def to_bounded_float(
        value: Any,
        min_value: float,
        max_value: float,
        default: Optional[float] = None
    ) -> float:
        """
        转换为有界float（自动clip）
        
        Args:
            value: 输入值
            min_value: 最小值
            max_value: 最大值
            default: 转换失败时的默认值（None则使用min_value）
        
        Returns:
            [min_value, max_value]范围内的float
        
        示例:
            ratio = SafeNumericConverter.to_bounded_float(
                raw_ratio,
                min_value=0.0,
                max_value=1.0
            )
        """
        if default is None:
            default = min_value
        
        try:
            result = float(value)
            
            if np.isnan(result) or np.isinf(result):
                logger.debug(f"转换结果无效({result})，返回默认值{default}")
                return default
            
            return float(np.clip(result, min_value, max_value))
        except (TypeError, ValueError) as e:
            logger.debug(f"转换为有界float失败({value}): {e}，返回默认值{default}")
            return default
    
    @staticmethod
    def to_int(
        value: Any,
        default: int = 0
    ) -> int:
        """
        安全转换为int
        
        Args:
            value: 输入值
            default: 转换失败时的默认值
        
        Returns:
            int值或默认值
        
        示例:
            days = SafeNumericConverter.to_int(calculated_days, default=1)
        """
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            logger.debug(f"转换为int失败({value}): {e}，返回默认值{default}")
            return default


class NumericCleaner:
    """数值清洗工具"""
    
    @staticmethod
    def remove_outliers(
        data: Union[np.ndarray, pd.Series],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Union[np.ndarray, pd.Series]:
        """
        移除异常值
        
        Args:
            data: 数据
            method: 方法 ('iqr' | 'zscore')
            threshold: 阈值（IQR倍数或Z分数）
        
        Returns:
            清洗后的数据
        
        示例:
            clean_returns = NumericCleaner.remove_outliers(
                returns,
                method='iqr',
                threshold=1.5
            )
        """
        is_series = isinstance(data, pd.Series)
        values = data.values if is_series else data
        
        if method == 'iqr':
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (values >= lower) & (values <= upper)
        elif method == 'zscore':
            mean = np.mean(values)
            std = np.std(values)
            z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros_like(values)
            mask = z_scores <= threshold
        else:
            raise ValueError(f"未知方法: {method}")
        
        if is_series:
            return data[mask]
        return values[mask]
    
    @staticmethod
    def winsorize(
        data: Union[np.ndarray, pd.Series],
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0
    ) -> Union[np.ndarray, pd.Series]:
        """
        缩尾处理（替代截断）
        
        Args:
            data: 数据
            lower_percentile: 下分位数
            upper_percentile: 上分位数
        
        Returns:
            缩尾后的数据
        
        示例:
            winsorized = NumericCleaner.winsorize(
                returns,
                lower_percentile=1.0,
                upper_percentile=99.0
            )
        """
        is_series = isinstance(data, pd.Series)
        values = data.values if is_series else data.copy()
        
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        
        values = np.clip(values, lower, upper)
        
        if is_series:
            return pd.Series(values, index=data.index)
        return values
    
    @staticmethod
    def fill_missing(
        data: pd.Series,
        method: str = 'forward',
        limit: Optional[int] = None
    ) -> pd.Series:
        """
        填充缺失值
        
        Args:
            data: 数据
            method: 方法 ('forward' | 'backward' | 'mean' | 'median' | 'zero')
            limit: 最大填充数量
        
        Returns:
            填充后的数据
        
        示例:
            filled = NumericCleaner.fill_missing(
                price_series,
                method='forward',
                limit=3
            )
        """
        if method == 'forward':
            return data.fillna(method='ffill', limit=limit)
        elif method == 'backward':
            return data.fillna(method='bfill', limit=limit)
        elif method == 'mean':
            return data.fillna(data.mean())
        elif method == 'median':
            return data.fillna(data.median())
        elif method == 'zero':
            return data.fillna(0.0)
        else:
            raise ValueError(f"未知方法: {method}")


class StatisticalNormalizer:
    """统计标准化工具"""
    
    @staticmethod
    def z_score_normalize(
        data: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        """
        Z分数标准化
        
        Args:
            data: 数据
        
        Returns:
            标准化后的数据 (均值0,标准差1)
        
        示例:
            normalized = StatisticalNormalizer.z_score_normalize(returns)
        """
        is_series = isinstance(data, pd.Series)
        values = data.values if is_series else data
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            logger.warning("标准差为0，返回全0数组")
            normalized = np.zeros_like(values)
        else:
            normalized = (values - mean) / std
        
        if is_series:
            return pd.Series(normalized, index=data.index)
        return normalized
    
    @staticmethod
    def min_max_normalize(
        data: Union[np.ndarray, pd.Series],
        feature_range: tuple = (0.0, 1.0)
    ) -> Union[np.ndarray, pd.Series]:
        """
        最小-最大标准化
        
        Args:
            data: 数据
            feature_range: 目标范围
        
        Returns:
            标准化后的数据
        
        示例:
            normalized = StatisticalNormalizer.min_max_normalize(
                data,
                feature_range=(0, 1)
            )
        """
        is_series = isinstance(data, pd.Series)
        values = data.values if is_series else data
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == min_val:
            logger.warning("最大值等于最小值，返回中间值数组")
            normalized = np.full_like(values, (feature_range[0] + feature_range[1]) / 2)
        else:
            # 先标准化到[0,1]
            normalized = (values - min_val) / (max_val - min_val)
            # 再缩放到目标范围
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        if is_series:
            return pd.Series(normalized, index=data.index)
        return normalized
    
    @staticmethod
    def robust_normalize(
        data: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        """
        鲁棒标准化（基于中位数和IQR）
        
        对异常值不敏感
        
        Args:
            data: 数据
        
        Returns:
            标准化后的数据
        
        示例:
            normalized = StatisticalNormalizer.robust_normalize(returns)
        """
        is_series = isinstance(data, pd.Series)
        values = data.values if is_series else data
        
        median = np.median(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        
        if iqr == 0:
            logger.warning("IQR为0，返回全0数组")
            normalized = np.zeros_like(values)
        else:
            normalized = (values - median) / iqr
        
        if is_series:
            return pd.Series(normalized, index=data.index)
        return normalized


class RatioCalculator:
    """比率计算器（防止除零）"""
    
    @staticmethod
    def safe_ratio(
        numerator: Union[float, np.ndarray],
        denominator: Union[float, np.ndarray],
        default: float = 0.0,
        min_denominator: float = 1e-8
    ) -> Union[float, np.ndarray]:
        """
        安全比率计算
        
        Args:
            numerator: 分子
            denominator: 分母
            default: 除零时的默认值
            min_denominator: 最小分母阈值
        
        Returns:
            比率或默认值
        
        示例:
            ratio = RatioCalculator.safe_ratio(a, b, default=1.0)
        """
        is_scalar = np.isscalar(denominator)
        
        if is_scalar:
            if abs(denominator) < min_denominator:
                return default
            result = numerator / denominator
            return result if np.isfinite(result) else default
        else:
            # 数组版本
            result = np.full_like(denominator, default, dtype=float)
            mask = np.abs(denominator) >= min_denominator
            result[mask] = numerator[mask] / denominator[mask] if isinstance(numerator, np.ndarray) else numerator / denominator[mask]
            result[~np.isfinite(result)] = default
            return result
    
    @staticmethod
    def percentage_change(
        current: Union[float, np.ndarray],
        baseline: Union[float, np.ndarray],
        default: float = 0.0
    ) -> Union[float, np.ndarray]:
        """
        计算百分比变化
        
        Args:
            current: 当前值
            baseline: 基线值
            default: 基线为0时的默认值
        
        Returns:
            百分比变化
        
        示例:
            pct_change = RatioCalculator.percentage_change(new_value, old_value)
        """
        return RatioCalculator.safe_ratio(
            current - baseline,
            baseline,
            default=default
        )
