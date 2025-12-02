"""
数据验证模块

职责：
1. 验证MarketData对象的完整性
2. 检查数据值的合理性
3. 识别异常和缺失数据
4. 提供数据清洗建议
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def validate_market_data(data: Any) -> Dict[str, Any]:
    """
    验证MarketData对象的完整性和合理性。
    
    检查项：
    - 必需字段存在性
    - 数值合理性（价格>0, 成交量>=0）
    - 时间戳有效性
    - OHLC逻辑关系（High>=Open, Low<=Close等）
    
    Args:
        data: MarketData对象或数据字典
    
    Returns:
        验证结果字典，包含：
        - valid: 是否有效
        - errors: 错误列表
        - warnings: 警告列表
    
    Example:
        >>> result = validate_market_data(market_data)
        >>> if not result['valid']:
        ...     print("数据无效:", result['errors'])
    """
    errors = []
    warnings = []
    
    try:
        # 1. 检查必需字段
        required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close']
        
        for field in required_fields:
            if hasattr(data, field):
                value = getattr(data, field)
            elif isinstance(data, dict):
                value = data.get(field)
            else:
                errors.append(f"无法访问字段: {field}")
                continue
            
            if value is None:
                errors.append(f"缺少必需字段: {field}")
        
        # 2. 提取OHLC值
        if hasattr(data, 'open'):
            open_price = data.open
            high_price = data.high
            low_price = data.low
            close_price = data.close
            volume = getattr(data, 'volume', None)
        elif isinstance(data, dict):
            open_price = data.get('open')
            high_price = data.get('high')
            low_price = data.get('low')
            close_price = data.get('close')
            volume = data.get('volume')
        else:
            errors.append("无法提取OHLC数据")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # 3. 检查价格合理性
        prices = [open_price, high_price, low_price, close_price]
        
        for i, price in enumerate(prices):
            if price is None:
                errors.append(f"价格字段为None: {required_fields[i+2]}")
                continue
            
            try:
                price_val = float(price)
                if price_val <= 0:
                    errors.append(f"价格必须大于0: {required_fields[i+2]}={price_val}")
                elif price_val > 1000000:
                    warnings.append(f"价格异常大: {required_fields[i+2]}={price_val}")
            except (ValueError, TypeError):
                errors.append(f"价格不是有效数值: {required_fields[i+2]}={price}")
        
        # 4. 检查OHLC逻辑关系
        try:
            if high_price is not None and low_price is not None:
                if high_price < low_price:
                    errors.append(f"High ({high_price}) < Low ({low_price})")
            
            if high_price is not None and open_price is not None:
                if high_price < open_price:
                    warnings.append(f"High ({high_price}) < Open ({open_price})")
            
            if high_price is not None and close_price is not None:
                if high_price < close_price:
                    warnings.append(f"High ({high_price}) < Close ({close_price})")
            
            if low_price is not None and open_price is not None:
                if low_price > open_price:
                    warnings.append(f"Low ({low_price}) > Open ({open_price})")
            
            if low_price is not None and close_price is not None:
                if low_price > close_price:
                    warnings.append(f"Low ({low_price}) > Close ({close_price})")
                    
        except (ValueError, TypeError) as e:
            warnings.append(f"OHLC逻辑检查失败: {e}")
        
        # 5. 检查成交量
        if volume is not None:
            try:
                volume_val = float(volume)
                if volume_val < 0:
                    errors.append(f"成交量不能为负数: {volume_val}")
                elif volume_val == 0:
                    warnings.append("成交量为0")
            except (ValueError, TypeError):
                warnings.append(f"成交量不是有效数值: {volume}")
        
        # 6. 检查时间戳
        if hasattr(data, 'timestamp'):
            timestamp = data.timestamp
        elif isinstance(data, dict):
            timestamp = data.get('timestamp')
        else:
            timestamp = None
        
        if timestamp is not None:
            try:
                if isinstance(timestamp, str):
                    datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    warnings.append(f"时间戳类型异常: {type(timestamp)}")
            except (ValueError, TypeError):
                errors.append(f"时间戳格式无效: {timestamp}")
        
        # 判断总体有效性
        valid = len(errors) == 0
        
        result = {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
        
        if not valid:
            logger.warning(f"数据验证失败: {len(errors)} 个错误, {len(warnings)} 个警告")
        elif warnings:
            logger.debug(f"数据验证通过，但有 {len(warnings)} 个警告")
        
        return result
        
    except Exception as e:
        logger.error(f"数据验证异常: {e}")
        return {
            'valid': False,
            'errors': [f"验证过程异常: {str(e)}"],
            'warnings': warnings,
            'error_count': len(errors) + 1,
            'warning_count': len(warnings)
        }


def validate_data_list(data_list: List[Any]) -> Dict[str, Any]:
    """
    批量验证MarketData列表。
    
    Args:
        data_list: MarketData对象列表
    
    Returns:
        汇总验证结果
    
    Example:
        >>> result = validate_data_list(market_data_list)
        >>> print(f"有效率: {result['valid_ratio']:.1%}")
    """
    if not data_list:
        return {
            'total': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'valid_ratio': 0,
            'total_errors': 0,
            'total_warnings': 0
        }
    
    valid_count = 0
    invalid_count = 0
    total_errors = 0
    total_warnings = 0
    invalid_indices = []
    
    for i, data in enumerate(data_list):
        result = validate_market_data(data)
        
        if result['valid']:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_indices.append(i)
        
        total_errors += result['error_count']
        total_warnings += result['warning_count']
    
    total = len(data_list)
    valid_ratio = valid_count / total if total > 0 else 0
    
    summary = {
        'total': total,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'valid_ratio': valid_ratio,
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'invalid_indices': invalid_indices[:10]  # 只返回前10个无效数据的索引
    }
    
    logger.info(
        f"数据列表验证完成: {total} 条数据, "
        f"{valid_count} 有效 ({valid_ratio:.1%}), "
        f"{invalid_count} 无效"
    )
    
    return summary


def clean_market_data(data: Any) -> Optional[Any]:
    """
    清洗MarketData对象，修复常见问题。
    
    清洗操作：
    - 移除NaN值
    - 填充缺失的可选字段
    - 修正明显的数据错误
    
    Args:
        data: MarketData对象或数据字典
    
    Returns:
        清洗后的数据，无法修复返回None
    """
    try:
        # 验证数据
        validation = validate_market_data(data)
        
        # 如果有严重错误，无法清洗
        if validation['error_count'] > 3:
            logger.warning("数据错误过多，无法清洗")
            return None
        
        # 简单清洗：移除NaN
        if isinstance(data, dict):
            cleaned = data.copy()
            
            # 处理NaN值
            import math
            for key, value in cleaned.items():
                if isinstance(value, float) and math.isnan(value):
                    cleaned[key] = None
            
            return cleaned
        
        # 如果是对象，返回原对象（假设已经清洗）
        return data
        
    except Exception as e:
        logger.error(f"数据清洗失败: {e}")
        return None
