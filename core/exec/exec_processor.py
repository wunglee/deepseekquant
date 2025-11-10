from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from core.base_processor import BaseProcessor
from common import OrderType, OrderStatus, TradeDirection
from infrastructure.interfaces import InfrastructureProvider

@dataclass
class Order:
    order_id: str
    symbol: str
    quantity: float
    side: TradeDirection
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    price: Optional[float] = None
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class ExecProcessor(BaseProcessor):
    def _initialize_core(self) -> bool:
        self.logger = InfrastructureProvider.get('logging').get_logger('DeepSeekQuant.Exec')
        self.event_bus = InfrastructureProvider.get('event_bus')
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        order_data = kwargs.get('order', {})
        if not order_data:
            return {'status': 'error', 'message': '缺少订单信息'}
        
        order = Order(
            order_id=order_data.get('order_id', f"ord-{datetime.now().timestamp()}"),
            symbol=order_data.get('symbol', 'TEST'),
            quantity=order_data.get('quantity', 1),
            side=order_data.get('side', TradeDirection.LONG),
            order_type=order_data.get('order_type', OrderType.MARKET),
            price=order_data.get('price')
        )
        # 模拟执行：直接标记为已成交
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = order.price if order.price else 0.0
        
        # 计算手续费与滑点成本
        commission_rate = float(kwargs.get('commission', 0.001))
        slippage_rate = float(kwargs.get('slippage', 0.0005))
        fill_price = order.average_fill_price
        notional = fill_price * order.filled_quantity
        costs = round(notional * (commission_rate + slippage_rate), 6)
        execution_report = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'filled_quantity': order.filled_quantity,
            'fill_price': fill_price,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'costs': costs,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"订单执行完成: {order.order_id}, {order.symbol}, {order.quantity}, costs={costs}")
        try:
            self.event_bus.publish('order.filled', {'order_id': order.order_id, 'symbol': order.symbol, 'filled_quantity': order.filled_quantity, 'fill_price': fill_price, 'costs': costs})
        except Exception:
            pass
        
        result_dict = order.__dict__.copy()
        result_dict['status'] = order.status.value  # 转换枚举为字符串
        result_dict['side'] = order.side.value
        result_dict['order_type'] = order.order_type.value
        return {'status': 'success', 'order': result_dict, 'execution_report': execution_report}

    def _cleanup_core(self):
        pass
