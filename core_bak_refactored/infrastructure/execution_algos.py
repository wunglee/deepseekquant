"""
执行算法库 - 基础设施层
从 core_bak/execution_engine.py 拆分
职责: 提供通用的订单执行算法（TWAP、VWAP、POV等）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.ExecutionAlgos')


class ExecutionAlgorithms:
    """执行算法库"""
    
    @staticmethod
    def twap_schedule(total_quantity: float,
                      duration_minutes: int,
                      interval_minutes: int = 5) -> List[Dict]:
        """
        TWAP (时间加权平均价格) 拆单计划
        
        Args:
            total_quantity: 总数量
            duration_minutes: 执行时长（分钟）
            interval_minutes: 拆单间隔（分钟）
            
        Returns:
            拆单计划列表
        """
        num_slices = int(duration_minutes / interval_minutes)
        quantity_per_slice = total_quantity / num_slices
        
        schedule = []
        for i in range(num_slices):
            schedule.append({
                'slice': i + 1,
                'quantity': quantity_per_slice,
                'delay_minutes': i * interval_minutes
            })
        
        return schedule
    
    @staticmethod
    def vwap_schedule(total_quantity: float,
                      volume_profile: pd.Series) -> List[Dict]:
        """
        VWAP (成交量加权平均价格) 拆单计划
        
        Args:
            total_quantity: 总数量
            volume_profile: 历史成交量分布
            
        Returns:
            拆单计划列表
        """
        if len(volume_profile) == 0:
            return []
        
        # 归一化成交量分布
        normalized_volume = volume_profile / volume_profile.sum()
        
        schedule = []
        for i, ratio in enumerate(normalized_volume):
            schedule.append({
                'slice': i + 1,
                'quantity': total_quantity * ratio,
                'volume_ratio': ratio
            })
        
        return schedule
    
    @staticmethod
    def pov_schedule(total_quantity: float,
                     participation_rate: float = 0.1,
                     market_volume_forecast: pd.Series = None) -> List[Dict]:
        """
        POV (成交量百分比) 拆单计划
        
        Args:
            total_quantity: 总数量
            participation_rate: 参与率
            market_volume_forecast: 市场成交量预测
            
        Returns:
            拆单计划列表
        """
        if market_volume_forecast is None or len(market_volume_forecast) == 0:
            # 简化版本：均匀拆分
            return ExecutionAlgorithms.twap_schedule(total_quantity, 60, 5)
        
        schedule = []
        remaining = total_quantity
        
        for i, forecast_volume in enumerate(market_volume_forecast):
            target_qty = forecast_volume * participation_rate
            slice_qty = min(target_qty, remaining)
            
            schedule.append({
                'slice': i + 1,
                'quantity': slice_qty,
                'market_volume': forecast_volume,
                'participation': participation_rate
            })
            
            remaining -= slice_qty
            if remaining <= 0:
                break
        
        return schedule
    
    @staticmethod
    def iceberg_schedule(total_quantity: float,
                         display_quantity: float,
                         min_quantity: Optional[float] = None) -> List[Dict]:
        """
        冰山订单拆单计划
        
        Args:
            total_quantity: 总数量
            display_quantity: 显示数量
            min_quantity: 最小数量
            
        Returns:
            拆单计划列表
        """
        if min_quantity is None:
            min_quantity = display_quantity
        
        schedule = []
        remaining = total_quantity
        slice_num = 1
        
        while remaining > 0:
            qty = min(display_quantity, remaining)
            if qty < min_quantity and remaining < display_quantity:
                qty = remaining
            
            schedule.append({
                'slice': slice_num,
                'quantity': qty,
                'is_hidden': True
            })
            
            remaining -= qty
            slice_num += 1
        
        return schedule
    
    @staticmethod
    def adaptive_schedule(total_quantity: float,
                          market_conditions: Dict,
                          aggressiveness: float = 0.5) -> List[Dict]:
        """
        自适应拆单计划
        
        Args:
            total_quantity: 总数量
            market_conditions: 市场条件
            aggressiveness: 激进程度 (0-1)
            
        Returns:
            拆单计划列表
        """
        # 根据市场条件调整拆单策略
        volatility = market_conditions.get('volatility', 0.02)
        liquidity = market_conditions.get('liquidity', 1.0)
        
        # 高波动或低流动性 -> 更保守（更多拆单）
        # 低波动或高流动性 -> 更激进（更少拆单）
        num_slices = int(10 * (1 + volatility) / liquidity / aggressiveness)
        num_slices = max(2, min(num_slices, 100))  # 限制在2-100之间
        
        return ExecutionAlgorithms.twap_schedule(
            total_quantity,
            duration_minutes=num_slices * 5,
            interval_minutes=5
        )
