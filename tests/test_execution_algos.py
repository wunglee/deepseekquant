"""
执行算法模块测试
"""
import pytest
from datetime import datetime
from core.exec.execution_algos import (
    ExecutionAlgorithms,
    AlgoExecutionContext,
    AlgoSlice
)


class TestExecutionAlgorithms:
    """执行算法测试类"""
    
    def test_twap_basic(self):
        """测试TWAP基本功能"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy",
            time_horizon_seconds=300
        )
        
        slices = ExecutionAlgorithms.twap(context, num_slices=5)
        
        assert len(slices) == 5
        assert all(isinstance(s, AlgoSlice) for s in slices)
        # 每个切片数量应该相等
        assert all(s.quantity == 200.0 for s in slices)
        # 验证总量
        total_qty = sum(s.quantity for s in slices)
        assert total_qty == 1000.0
    
    def test_vwap_with_volume_profile(self):
        """测试VWAP with成交量分布"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy",
            time_horizon_seconds=300,
            market_data={'volume_profile': [10, 20, 30, 20, 10, 10]}
        )
        
        slices = ExecutionAlgorithms.vwap(context)
        
        assert len(slices) > 0
        # 验证总量
        total_qty = sum(s.quantity for s in slices)
        assert abs(total_qty - 1000.0) < 0.01  # 允许小的舍入误差
        # 最大切片应该对应最大成交量
        max_slice = max(slices, key=lambda s: s.quantity)
        assert max_slice.quantity == pytest.approx(300.0, rel=0.01)
    
    def test_vwap_fallback_to_twap(self):
        """测试VWAP在没有成交量数据时回退到TWAP"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy",
            time_horizon_seconds=300
        )
        
        slices = ExecutionAlgorithms.vwap(context)
        
        # 应该回退到TWAP，5个切片
        assert len(slices) == 5
        assert all(s.quantity == 200.0 for s in slices)
    
    def test_pov_basic(self):
        """测试POV基本功能"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy",
            market_data={'market_volumes': [100, 200, 300, 200, 100, 100, 200, 150, 100, 50]}
        )
        
        slices = ExecutionAlgorithms.pov(context, participation_rate=0.1)
        
        assert len(slices) > 0
        # 验证每个切片不超过市场成交量的10%
        market_volumes = context.market_data['market_volumes']
        for i, slice_obj in enumerate(slices):
            if i < len(market_volumes):
                assert slice_obj.quantity <= market_volumes[i] * 0.1
    
    def test_iceberg_basic(self):
        """测试冰山订单基本功能"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy"
        )
        
        slices = ExecutionAlgorithms.iceberg(context, display_quantity=100.0)
        
        # 验证切片数量
        assert len(slices) == 10
        # 每个切片不超过显示数量
        assert all(s.quantity <= 100.0 for s in slices)
        # 验证总量
        total_qty = sum(s.quantity for s in slices)
        assert total_qty == 1000.0
    
    def test_implementation_shortfall_urgency_impact(self):
        """测试Implementation Shortfall不同紧急度的影响"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy",
            time_horizon_seconds=300
        )
        
        # 低紧急度应该有更多切片
        slices_low = ExecutionAlgorithms.implementation_shortfall(
            context, risk_aversion=0.5, urgency="low"
        )
        
        # 高紧急度应该有更少切片
        slices_high = ExecutionAlgorithms.implementation_shortfall(
            context, risk_aversion=0.5, urgency="urgent"
        )
        
        assert len(slices_low) > len(slices_high)
        # 验证总量
        assert abs(sum(s.quantity for s in slices_low) - 1000.0) < 0.01
        assert abs(sum(s.quantity for s in slices_high) - 1000.0) < 0.01
    
    def test_implementation_shortfall_weight_distribution(self):
        """测试Implementation Shortfall权重分布（前重后轻）"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy",
            time_horizon_seconds=300
        )
        
        slices = ExecutionAlgorithms.implementation_shortfall(context, urgency="medium")
        
        # 第一个切片应该比最后一个切片大
        assert slices[0].quantity > slices[-1].quantity
        # 验证递减趋势（允许小的波动）
        for i in range(len(slices) - 1):
            # 前面的切片一般应该大于或接近后面的切片
            assert slices[i].quantity >= slices[i+1].quantity * 0.8
    
    def test_algo_slice_structure(self):
        """测试AlgoSlice数据结构"""
        context = AlgoExecutionContext(
            symbol="AAPL",
            total_quantity=1000.0,
            side="buy"
        )
        
        slices = ExecutionAlgorithms.twap(context, num_slices=3)
        
        for slice_obj in slices:
            assert hasattr(slice_obj, 'slice_id')
            assert hasattr(slice_obj, 'quantity')
            assert hasattr(slice_obj, 'execute_at')
            assert hasattr(slice_obj, 'urgency')
            assert isinstance(slice_obj.slice_id, str)
            assert isinstance(slice_obj.quantity, float)
            assert isinstance(slice_obj.execute_at, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
