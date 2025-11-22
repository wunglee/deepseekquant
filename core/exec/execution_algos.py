"""
执行算法模块
从 core_bak/execution_engine.py 提取的真实业务逻辑实现
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import copy


@dataclass
class AlgoExecutionContext:
    """算法执行上下文"""
    symbol: str
    total_quantity: float
    side: str  # 'buy' or 'sell'
    target_price: Optional[float] = None
    time_horizon_seconds: int = 300
    market_data: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class AlgoSlice:
    """算法切片订单"""
    slice_id: str
    quantity: float
    limit_price: Optional[float] = None
    execute_at: str = field(default_factory=lambda: datetime.now().isoformat())
    urgency: str = "medium"


class ExecutionAlgorithms:
    """执行算法集合"""

    @staticmethod
    def twap(context: AlgoExecutionContext, num_slices: int = 5) -> List[AlgoSlice]:
        """
        时间加权平均价格算法：将订单均匀分布在时间窗口内
        从 core_bak/execution_engine.py:_execute_twap 提取 (line 1210-1277)
        """
        slice_quantity = context.total_quantity / num_slices
        slice_interval = context.time_horizon_seconds / num_slices
        
        slices: List[AlgoSlice] = []  
        for slice_num in range(num_slices):
            # 计算每个切片的执行时间
            execute_time = datetime.now().timestamp() + slice_num * slice_interval
            
            slices.append(AlgoSlice(
                slice_id=f"{context.symbol}_twap_slice_{slice_num}",
                quantity=round(slice_quantity, 6),
                execute_at=datetime.fromtimestamp(execute_time).isoformat(),
                urgency="medium"
            ))
        
        return slices

    @staticmethod
    def vwap(context: AlgoExecutionContext, volume_weights: Optional[List[float]] = None) -> List[AlgoSlice]:
        """
        成交量加权平均价格算法：根据历史成交量分布切分订单
        从 core_bak/execution_engine.py:_execute_vwap 提取 (line 1279-1370)
        """
        # 获取成交量数据
        volume_data = context.market_data.get('volume_profile', volume_weights)
        if not volume_data:
            # 回退到TWAP
            return ExecutionAlgorithms.twap(context, num_slices=5)
        
        # 计算成交量分布权重
        total_volume = sum(volume_data)
        if total_volume <= 0:
            return ExecutionAlgorithms.twap(context, num_slices=5)
        
        volume_weights_normalized = [v / total_volume for v in volume_data]
        
        # 确定执行时间段
        time_horizon = context.time_horizon_seconds
        slices = min(time_horizon, len(volume_data))
        
        # 计算每个时间片的执行量
        slice_quantities = []
        for i in range(slices):
            slice_qty = context.total_quantity * volume_weights_normalized[i]
            slice_quantities.append(slice_qty)
        
        # 生成切片订单
        slices_result: List[AlgoSlice] = []
        for slice_num in range(slices):
            if slice_quantities[slice_num] > 0:
                slices_result.append(AlgoSlice(
                    slice_id=f"{context.symbol}_vwap_slice_{slice_num}",
                    quantity=round(slice_quantities[slice_num], 6),
                    execute_at=datetime.now().isoformat(),
                    urgency="medium"
                ))
        
        return slices_result

    @staticmethod
    def pov(context: AlgoExecutionContext, 
            participation_rate: float = 0.1,
            max_participation: float = 0.2,
            min_participation: float = 0.05) -> List[AlgoSlice]:
        """
        参与率(POV)执行算法：按市场成交量的固定比例执行
        从 core_bak/execution_engine.py:_execute_pov 提取 (line 1372-1465)
        """
        # 限制参与率范围
        participation_rate = max(min(participation_rate, max_participation), min_participation)
        
        # 获取市场成交量数据
        volume_data = context.market_data.get('market_volumes', [])
        if not volume_data:
            # 回退到TWAP
            return ExecutionAlgorithms.twap(context, num_slices=5)
        
        remaining_quantity = context.total_quantity
        time_slices = min(len(volume_data), 10)  # 限制时间片数量
        
        slices: List[AlgoSlice] = []
        for slice_num in range(time_slices):
            if remaining_quantity <= 0:
                break
            
            # 计算当前市场成交量
            current_volume = volume_data[slice_num] if slice_num < len(volume_data) else 0
            if current_volume <= 0:
                continue
            
            # 计算当前切片执行量
            slice_qty = min(remaining_quantity, current_volume * participation_rate)
            if slice_qty <= 0:
                continue
            
            slices.append(AlgoSlice(
                slice_id=f"{context.symbol}_pov_slice_{slice_num}",
                quantity=round(slice_qty, 6),
                execute_at=datetime.now().isoformat(),
                urgency="medium"
            ))
            
            remaining_quantity -= slice_qty
        
        return slices

    @staticmethod
    def iceberg(context: AlgoExecutionContext, display_quantity: float = 100.0) -> List[AlgoSlice]:
        """
        冰山订单：每次仅显示部分数量，隐藏总订单规模
        TODO：补充了冰山订单算法实现，待确认（core_bak中未找到完整独立实现）
        参考 core_bak/execution_engine.py 订单参数中的iceberg字段 (line 760-766)
        """
        # 计算需要的切片数量
        num_slices = int(context.total_quantity / display_quantity)
        if context.total_quantity % display_quantity > 0:
            num_slices += 1
        
        slices: List[AlgoSlice] = []
        remaining_quantity = context.total_quantity
        
        for slice_num in range(num_slices):
            slice_qty = min(remaining_quantity, display_quantity)
            if slice_qty <= 0:
                break
            
            slices.append(AlgoSlice(
                slice_id=f"{context.symbol}_iceberg_slice_{slice_num}",
                quantity=round(slice_qty, 6),
                execute_at=datetime.now().isoformat(),
                urgency="low"  # 冰山订单通常不急
            ))
            
            remaining_quantity -= slice_qty
        
        return slices

    @staticmethod
    def implementation_shortfall(context: AlgoExecutionContext, 
                                risk_aversion: float = 0.5,
                                urgency: str = "medium",
                                benchmark: str = "arrival_price") -> List[AlgoSlice]:
        """
        执行差额算法：平衡市场冲击与时机风险
        从 core_bak/execution_engine.py:_execute_implementation_shortfall 提取 (line 1467-1539)
        """
        # 计算urgency权重
        urgency_weights = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'urgent': 0.9
        }
        urgency_weight = urgency_weights.get(urgency, 0.5)
        
        # 根据风险厌恶和紧急度决定执行策略
        # 风险厌恶越高，切片越多；紧急度越高，切片越少
        base_slices = 10
        risk_adjustment = int((1 - risk_aversion) * 5)  # -5到0的调整
        urgency_adjustment = int((urgency_weight - 0.5) * 10)  # -5到5的调整
        
        num_slices = max(2, base_slices + risk_adjustment - urgency_adjustment)
        
        # 前期执行较多（降低市场冲击），后期执行较少（降低时机风险）
        weights = [1.0 / (i + 1) for i in range(num_slices)]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        slices: List[AlgoSlice] = []
        for slice_num, weight in enumerate(normalized_weights):
            slice_qty = context.total_quantity * weight
            
            # 计算执行时间
            time_fraction = slice_num / num_slices
            execute_time = datetime.now().timestamp() + time_fraction * context.time_horizon_seconds
            
            slices.append(AlgoSlice(
                slice_id=f"{context.symbol}_is_slice_{slice_num}",
                quantity=round(slice_qty, 6),
                execute_at=datetime.fromtimestamp(execute_time).isoformat(),
                urgency=urgency
            ))
        
        return slices
