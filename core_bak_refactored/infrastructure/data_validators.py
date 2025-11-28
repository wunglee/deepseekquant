"""
统一数据验证工具
从risk模块61处长度检查和85处类型检查中提炼的通用验证器
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger('DeepSeekQuant.DataValidators')


class LengthValidator:
    """数据长度验证器（统一61处len检查）"""
    
    @staticmethod
    def validate_min_length(
        data: Union[List, np.ndarray, pd.Series, pd.DataFrame],
        min_length: int,
        data_name: str = "数据"
    ) -> bool:
        """
        验证最小长度
        
        Args:
            data: 数据
            min_length: 最小长度
            data_name: 数据名称（用于日志）
        
        Returns:
            是否满足最小长度
        
        示例:
            if LengthValidator.validate_min_length(returns, 20, "收益率"):
                # 执行计算
        """
        try:
            actual_length = len(data)
            is_valid = actual_length >= min_length
            
            if not is_valid:
                logger.debug(
                    f"{data_name}长度不足: 需要{min_length}, 实际{actual_length}"
                )
            
            return is_valid
        except Exception:
            logger.warning(f"{data_name}长度验证失败")
            return False
    
    @staticmethod
    def get_valid_data_or_none(
        data: Union[List, np.ndarray, pd.Series, pd.DataFrame],
        min_length: int,
        data_name: str = "数据"
    ) -> Optional[Union[List, np.ndarray, pd.Series, pd.DataFrame]]:
        """
        获取有效数据，无效则返回None
        
        Args:
            data: 输入数据
            min_length: 最小长度
            data_name: 数据名称
        
        Returns:
            有效数据或None
        
        示例:
            returns = LengthValidator.get_valid_data_or_none(raw_returns, 20)
            if returns is None:
                return default_value
        """
        if LengthValidator.validate_min_length(data, min_length, data_name):
            return data
        return None
    
    @staticmethod
    def validate_length_range(
        data: Union[List, np.ndarray, pd.Series],
        min_length: int,
        max_length: Optional[int] = None,
        data_name: str = "数据"
    ) -> bool:
        """
        验证长度范围
        
        Args:
            data: 数据
            min_length: 最小长度
            max_length: 最大长度（可选）
            data_name: 数据名称
        
        Returns:
            是否在范围内
        """
        try:
            actual_length = len(data)
            
            if actual_length < min_length:
                logger.debug(f"{data_name}长度低于最小值: {actual_length} < {min_length}")
                return False
            
            if max_length is not None and actual_length > max_length:
                logger.debug(f"{data_name}长度超过最大值: {actual_length} > {max_length}")
                return False
            
            return True
        except Exception:
            return False


class TypeValidator:
    """类型验证器（统一85处isinstance检查）"""
    
    @staticmethod
    def ensure_series(
        data: Union[List, np.ndarray, pd.Series],
        name: str = "data"
    ) -> pd.Series:
        """
        确保数据是pandas Series
        
        Args:
            data: 输入数据
            name: 数据名称
        
        Returns:
            pandas Series
        
        示例:
            returns = TypeValidator.ensure_series(raw_data, "returns")
        """
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        else:
            logger.warning(f"{name}类型无效: {type(data)}, 尝试转换为Series")
            return pd.Series(data)
    
    @staticmethod
    def ensure_dataframe(
        data: Union[Dict, pd.DataFrame, np.ndarray],
        name: str = "data"
    ) -> pd.DataFrame:
        """
        确保数据是pandas DataFrame
        
        Args:
            data: 输入数据
            name: 数据名称
        
        Returns:
            pandas DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            logger.warning(f"{name}类型无效: {type(data)}, 尝试转换为DataFrame")
            return pd.DataFrame(data)
    
    @staticmethod
    def ensure_numeric_array(
        data: Union[List, np.ndarray, pd.Series],
        name: str = "data"
    ) -> np.ndarray:
        """
        确保数据是numpy数组（数值类型）
        
        Args:
            data: 输入数据
            name: 数据名称
        
        Returns:
            numpy ndarray
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        else:
            logger.warning(f"{name}类型无效: {type(data)}, 尝试转换为ndarray")
            return np.array(data)
    
    @staticmethod
    def validate_dict_structure(
        data: Dict,
        required_keys: List[str],
        data_name: str = "字典"
    ) -> bool:
        """
        验证字典结构
        
        Args:
            data: 字典数据
            required_keys: 必需的键列表
            data_name: 数据名称
        
        Returns:
            是否包含所有必需键
        
        示例:
            if TypeValidator.validate_dict_structure(
                market_data, 
                ['prices', 'volumes'],
                "市场数据"
            ):
                # 处理数据
        """
        if not isinstance(data, dict):
            logger.debug(f"{data_name}不是字典类型: {type(data)}")
            return False
        
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            logger.debug(f"{data_name}缺少键: {missing_keys}")
            return False
        
        return True


class NumericValidator:
    """数值有效性验证器"""
    
    @staticmethod
    def is_valid_numeric(
        value: Any,
        allow_nan: bool = False,
        allow_inf: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """
        验证数值有效性
        
        Args:
            value: 数值
            allow_nan: 是否允许NaN
            allow_inf: 是否允许Inf
            min_value: 最小值限制
            max_value: 最大值限制
        
        Returns:
            是否有效
        
        示例:
            if NumericValidator.is_valid_numeric(ratio, min_value=0.0, max_value=1.0):
                # 使用ratio
        """
        try:
            # 类型检查
            if not isinstance(value, (int, float, np.number)):
                return False
            
            # NaN检查
            if not allow_nan and np.isnan(value):
                return False
            
            # Inf检查
            if not allow_inf and np.isinf(value):
                return False
            
            # 范围检查
            if min_value is not None and value < min_value:
                return False
            
            if max_value is not None and value > max_value:
                return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def clean_numeric_array(
        data: Union[np.ndarray, pd.Series],
        remove_nan: bool = True,
        remove_inf: bool = True,
        clip_range: Optional[tuple] = None
    ) -> Union[np.ndarray, pd.Series]:
        """
        清洗数值数组
        
        Args:
            data: 数据数组
            remove_nan: 是否移除NaN
            remove_inf: 是否移除Inf
            clip_range: 截断范围 (min, max)
        
        Returns:
            清洗后的数据
        
        示例:
            clean_returns = NumericValidator.clean_numeric_array(
                returns,
                remove_nan=True,
                clip_range=(-0.5, 0.5)
            )
        """
        result = data.copy()
        
        # 移除NaN
        if remove_nan:
            if isinstance(result, pd.Series):
                result = result[~result.isna()]
            else:
                result = result[~np.isnan(result)]
        
        # 移除Inf
        if remove_inf:
            if isinstance(result, pd.Series):
                result = result[~result.isin([np.inf, -np.inf])]
            else:
                result = result[~np.isinf(result)]
        
        # 截断
        if clip_range is not None:
            min_val, max_val = clip_range
            result = np.clip(result, min_val, max_val)
        
        return result
    
    @staticmethod
    def safe_division(
        numerator: float,
        denominator: float,
        default: float = 0.0,
        min_denominator: float = 1e-8
    ) -> float:
        """
        安全除法（防止除零）
        
        Args:
            numerator: 分子
            denominator: 分母
            default: 除零时返回值
            min_denominator: 最小分母阈值
        
        Returns:
            除法结果或默认值
        
        示例:
            ratio = NumericValidator.safe_division(a, b, default=1.0)
        """
        try:
            if abs(denominator) < min_denominator:
                logger.debug(f"分母过小({denominator})，返回默认值{default}")
                return default
            
            result = numerator / denominator
            
            if not np.isfinite(result):
                logger.debug(f"除法结果无效({result})，返回默认值{default}")
                return default
            
            return float(result)
        except Exception:
            return default


class DataQualityValidator:
    """数据质量综合验证器"""
    
    @staticmethod
    def validate_timeseries_quality(
        data: pd.Series,
        min_length: int = 20,
        max_missing_ratio: float = 0.1,
        check_monotonic: bool = False
    ) -> Dict[str, Any]:
        """
        验证时间序列数据质量
        
        Args:
            data: 时间序列数据
            min_length: 最小长度
            max_missing_ratio: 最大缺失比例
            check_monotonic: 是否检查单调性
        
        Returns:
            验证结果字典
        
        示例:
            quality = DataQualityValidator.validate_timeseries_quality(
                price_series,
                min_length=50,
                max_missing_ratio=0.05
            )
            if quality['is_valid']:
                # 使用数据
        """
        result = {
            'is_valid': True,
            'length': len(data),
            'missing_count': 0,
            'missing_ratio': 0.0,
            'issues': []
        }
        
        # 长度检查
        if len(data) < min_length:
            result['is_valid'] = False
            result['issues'].append(f"长度不足: {len(data)} < {min_length}")
        
        # 缺失值检查
        missing_count = data.isna().sum()
        missing_ratio = missing_count / len(data) if len(data) > 0 else 1.0
        result['missing_count'] = int(missing_count)
        result['missing_ratio'] = float(missing_ratio)
        
        if missing_ratio > max_missing_ratio:
            result['is_valid'] = False
            result['issues'].append(
                f"缺失比例过高: {missing_ratio:.2%} > {max_missing_ratio:.2%}"
            )
        
        # 单调性检查
        if check_monotonic:
            is_monotonic = data.is_monotonic_increasing or data.is_monotonic_decreasing
            result['is_monotonic'] = is_monotonic
            if not is_monotonic:
                result['issues'].append("数据不单调")
        
        return result
