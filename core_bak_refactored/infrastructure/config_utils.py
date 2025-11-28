"""
配置管理工具
从risk模块373处.get()调用中提炼的统一配置提取器
"""

from typing import Any, Dict, List, Optional, Union, TypeVar
import logging

logger = logging.getLogger('DeepSeekQuant.ConfigUtils')

T = TypeVar('T')


class ConfigExtractor:
    """配置提取器（统一373处.get()调用）"""
    
    @staticmethod
    def get_nested(
        config: Dict[str, Any],
        path: str,
        default: Any = None,
        separator: str = '.'
    ) -> Any:
        """
        提取嵌套配置
        
        Args:
            config: 配置字典
            path: 配置路径（用separator分隔）
            default: 默认值
            separator: 路径分隔符
        
        Returns:
            配置值或默认值
        
        示例:
            alpha = ConfigExtractor.get_nested(
                config,
                'market_configs.CN.price_impact_alpha',
                default=0.4
            )
        """
        try:
            keys = path.split(separator)
            value = config
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                    if value is None:
                        logger.debug(f"配置路径'{path}'在'{key}'处中断，使用默认值{default}")
                        return default
                else:
                    logger.debug(f"配置路径'{path}'在'{key}'处类型错误，使用默认值{default}")
                    return default
            
            return value
        except Exception as e:
            logger.debug(f"配置提取失败'{path}': {e}，使用默认值{default}")
            return default
    
    @staticmethod
    def get_typed(
        config: Dict[str, Any],
        key: str,
        expected_type: type,
        default: T
    ) -> T:
        """
        提取指定类型的配置
        
        Args:
            config: 配置字典
            key: 配置键
            expected_type: 期望类型
            default: 默认值
        
        Returns:
            指定类型的配置值或默认值
        
        示例:
            threshold = ConfigExtractor.get_typed(
                config,
                'threshold',
                float,
                0.95
            )
        """
        try:
            value = config.get(key, default)
            
            if not isinstance(value, expected_type):
                logger.debug(
                    f"配置'{key}'类型错误: 期望{expected_type.__name__}, "
                    f"实际{type(value).__name__}, 使用默认值{default}"
                )
                return default
            
            return value
        except Exception as e:
            logger.debug(f"配置提取失败'{key}': {e}，使用默认值{default}")
            return default
    
    @staticmethod
    def get_market_config(
        config: Dict[str, Any],
        market_type: str,
        param_name: str,
        default: Any
    ) -> Any:
        """
        提取市场特定配置（简化常见模式）
        
        Args:
            config: 配置字典
            market_type: 市场类型（CN/US/HK等）
            param_name: 参数名
            default: 默认值
        
        Returns:
            市场配置值或默认值
        
        示例:
            alpha = ConfigExtractor.get_market_config(
                config,
                'CN',
                'price_impact_alpha',
                0.4
            )
        """
        return ConfigExtractor.get_nested(
            config,
            f'market_configs.{market_type}.{param_name}',
            default=default
        )
    
    @staticmethod
    def get_with_fallback(
        config: Dict[str, Any],
        keys: List[str],
        default: Any
    ) -> Any:
        """
        按优先级获取配置（fallback链）
        
        Args:
            config: 配置字典
            keys: 配置键列表（按优先级）
            default: 默认值
        
        Returns:
            第一个存在的配置值或默认值
        
        示例:
            # 先尝试advanced_var_enabled，再尝试var_enabled
            enabled = ConfigExtractor.get_with_fallback(
                config,
                ['advanced_var_enabled', 'var_enabled'],
                default=False
            )
        """
        for key in keys:
            if key in config:
                return config[key]
        
        logger.debug(f"配置键{keys}均不存在，使用默认值{default}")
        return default
    
    @staticmethod
    def get_numeric_range(
        config: Dict[str, Any],
        key: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        提取带范围限制的数值配置
        
        Args:
            config: 配置字典
            key: 配置键
            default: 默认值
            min_value: 最小值限制
            max_value: 最大值限制
        
        Returns:
            范围内的配置值或默认值
        
        示例:
            confidence = ConfigExtractor.get_numeric_range(
                config,
                'var_confidence_level',
                default=0.95,
                min_value=0.5,
                max_value=0.999
            )
        """
        try:
            value = float(config.get(key, default))
            
            if min_value is not None and value < min_value:
                logger.warning(
                    f"配置'{key}'={value}低于最小值{min_value}，使用最小值"
                )
                return min_value
            
            if max_value is not None and value > max_value:
                logger.warning(
                    f"配置'{key}'={value}超过最大值{max_value}，使用最大值"
                )
                return max_value
            
            return value
        except Exception as e:
            logger.debug(f"配置提取失败'{key}': {e}，使用默认值{default}")
            return default


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_required_keys(
        config: Dict[str, Any],
        required_keys: Dict[str, type],
        config_name: str = "配置"
    ) -> tuple[bool, List[str]]:
        """
        验证必需配置键
        
        Args:
            config: 配置字典
            required_keys: {键名: 期望类型}
            config_name: 配置名称
        
        Returns:
            (是否有效, 错误消息列表)
        
        示例:
            valid, errors = ConfigValidator.validate_required_keys(
                config,
                {
                    'market_type': str,
                    'var_confidence_level': float
                },
                "风险配置"
            )
        """
        errors = []
        
        for key, expected_type in required_keys.items():
            if key not in config:
                errors.append(f"{config_name}缺少必需键'{key}'")
                continue
            
            if not isinstance(config[key], expected_type):
                errors.append(
                    f"{config_name}键'{key}'类型错误: "
                    f"期望{expected_type.__name__}, 实际{type(config[key]).__name__}"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"{config_name}验证失败: {'; '.join(errors)}")
        
        return is_valid, errors
    
    @staticmethod
    def fill_defaults(
        config: Dict[str, Any],
        defaults: Dict[str, Any],
        deep_merge: bool = False
    ) -> Dict[str, Any]:
        """
        填充默认配置
        
        Args:
            config: 原始配置
            defaults: 默认配置
            deep_merge: 是否深度合并（嵌套字典）
        
        Returns:
            合并后的配置
        
        示例:
            config = ConfigValidator.fill_defaults(
                user_config,
                DEFAULT_RISK_CONFIG,
                deep_merge=True
            )
        """
        result = config.copy()
        
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
                logger.debug(f"填充默认配置: {key} = {default_value}")
            elif deep_merge and isinstance(default_value, dict) and isinstance(result[key], dict):
                # 递归合并
                result[key] = ConfigValidator.fill_defaults(
                    result[key],
                    default_value,
                    deep_merge=True
                )
        
        return result


class ThresholdManager:
    """阈值管理器（统一阈值配置和动态调整）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化阈值管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self._cache = {}
    
    def get_threshold(
        self,
        threshold_name: str,
        market_type: Optional[str] = None,
        default: float = 1.0
    ) -> float:
        """
        获取阈值（支持市场特定阈值）
        
        Args:
            threshold_name: 阈值名称
            market_type: 市场类型（可选，如果提供则查找市场特定阈值）
            default: 默认值
        
        Returns:
            阈值
        
        示例:
            threshold = manager.get_threshold(
                'normal_vol_max',
                market_type='CN',
                default=1.2
            )
        """
        # 构建缓存键
        cache_key = f"{threshold_name}_{market_type}" if market_type else threshold_name
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 尝试获取市场特定阈值
        if market_type:
            value = ConfigExtractor.get_market_config(
                self.config,
                market_type,
                threshold_name,
                default=None
            )
            if value is not None:
                self._cache[cache_key] = float(value)
                return float(value)
        
        # 回退到通用阈值
        value = ConfigExtractor.get_nested(
            self.config,
            f'thresholds.{threshold_name}',
            default=default
        )
        
        self._cache[cache_key] = float(value)
        return float(value)
    
    def get_dynamic_threshold(
        self,
        threshold_name: str,
        adjustment_factor: float = 1.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        获取动态调整的阈值
        
        Args:
            threshold_name: 阈值名称
            adjustment_factor: 调整因子
            min_value: 最小值限制
            max_value: 最大值限制
        
        Returns:
            动态阈值
        
        示例:
            # 在压力期调高阈值
            threshold = manager.get_dynamic_threshold(
                'leverage_limit',
                adjustment_factor=0.8,  # 降低20%
                min_value=1.0
            )
        """
        base_value = self.get_threshold(threshold_name)
        adjusted_value = base_value * adjustment_factor
        
        if min_value is not None:
            adjusted_value = max(adjusted_value, min_value)
        
        if max_value is not None:
            adjusted_value = min(adjusted_value, max_value)
        
        return adjusted_value
